import os
import json
import google.generativeai as genai
import pandas as pd
import numpy as np
import dotenv
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

dotenv.load_dotenv()
_GEMINI_API_KEY = (os.environ.get("GEMINI_API_KEY") or "").strip()
genai.configure(api_key=_GEMINI_API_KEY)

# ── RAG INDEX STATE ────────────────────────────────────────────────────────
_RAG_INDEX = {
    "vocabulary": [],
    "vectorizer": None,
    "embeddings": None,
    "signature": None,
}

def reset_categorical_index():
    """Clears the in-memory TF-IDF index so it can be rebuilt for new data."""
    global _RAG_INDEX
    _RAG_INDEX = {
        "vocabulary": [],
        "vectorizer": None,
        "embeddings": None,
        "signature": None,
    }

def build_categorical_index(df: pd.DataFrame):
    """Embeds unique categorical values using local TF-IDF semantic lookup."""
    global _RAG_INDEX
    cols_to_index = ["Training Name", "Department", "Business Unit", "Employee Name", "Designation"]
    signature_parts = []
    for c in cols_to_index:
        if c in df.columns:
            try:
                signature_parts.append((c, int(df[c].nunique(dropna=True))))
            except Exception:
                signature_parts.append((c, None))
    signature = (len(df), tuple(signature_parts))

    if _RAG_INDEX["embeddings"] is not None and _RAG_INDEX.get("signature") == signature:
        return

    _RAG_INDEX["signature"] = signature
    all_values = set()
    for c in cols_to_index:
        if c in df.columns:
            for val in df[c].dropna().unique():
                val_str = str(val).strip()
                if val_str:
                    all_values.add(val_str)

    vocab = list(all_values)
    _RAG_INDEX["vocabulary"] = vocab
    
    try:
        # Character n-grams correctly map semantic intent for names and buzzwords even with typos
        vectorizer = TfidfVectorizer(analyzer='char_wb', ngram_range=(2, 4))
        embeddings = vectorizer.fit_transform(vocab)
        _RAG_INDEX["vectorizer"] = vectorizer
        _RAG_INDEX["embeddings"] = embeddings
        print(f"[RAG Index Built] Indexed {len(vocab)} unique terms locally.")
    except Exception as e:
        print(f"[RAG Local Build Error] {e}")

def retrieve_relevant_categories(user_query: str, top_k=5) -> list:
    """Retrieves top matching exact vocabulary strings using local vector similarity."""
    global _RAG_INDEX
    if _RAG_INDEX["embeddings"] is None or len(_RAG_INDEX["vocabulary"]) == 0:
        return []

    try:
        vectorizer = _RAG_INDEX["vectorizer"]
        query_emb = vectorizer.transform([user_query])
        
        sims = cosine_similarity(query_emb, _RAG_INDEX["embeddings"])[0]
        
        top_indices = np.argsort(sims)[-top_k:][::-1]
        
        # Return terms with at least some similarity threshold
        return [_RAG_INDEX["vocabulary"][i] for i in top_indices if sims[i] > 0.15]
    except Exception as e:
        print(f"[RAG Retrieval Error] {e}")
        return []

def _normalize_plan(plan: dict, schema_dict: dict) -> dict:
    """
    Normalizes a Gemini-produced plan into a safer, executor-friendly shape.
    """
    if not isinstance(plan, dict):
        plan = {}

    normalized = {
        "intent": plan.get("intent") or "filter",
        "deduplicate": bool(plan.get("deduplicate", False)),
        "columns": plan.get("columns") if isinstance(plan.get("columns"), list) else [],
        "filters": plan.get("filters") if isinstance(plan.get("filters"), list) else [],
        "group_by": plan.get("group_by") if isinstance(plan.get("group_by"), list) else [],
        "aggregation": plan.get("aggregation") if isinstance(plan.get("aggregation"), dict) else {"column": "", "operation": "none"},
        "sort": plan.get("sort") if isinstance(plan.get("sort"), dict) else {"column": "", "ascending": True},
        "limit": plan.get("limit", None),
        "response_text": plan.get("response_text", ""),
    }

    allowed_cols = set(schema_dict.keys())
    normalized["columns"] = [c for c in normalized["columns"] if isinstance(c, str) and c in allowed_cols]
    normalized["group_by"] = [c for c in normalized["group_by"] if isinstance(c, str) and c in allowed_cols]

    cleaned_filters = []
    for f in normalized["filters"]:
        if not isinstance(f, dict):
            continue
        col = f.get("column")
        op = f.get("operator")
        val = f.get("value")
        if not isinstance(col, str) or col not in allowed_cols:
            continue
        if op not in ("==", ">", "<", ">=", "<=", "contains"):
            continue
        if val is None:
            continue
        if isinstance(val, str):
            val = val.strip()
        cleaned_filters.append({"column": col, "operator": op, "value": val})
    normalized["filters"] = cleaned_filters

    agg = normalized["aggregation"]
    if agg.get("operation") not in ("sum", "mean", "count", "none"):
        agg["operation"] = "none"
    if not isinstance(agg.get("column", ""), str) or agg.get("column", "") not in allowed_cols:
        agg["column"] = ""

    sort = normalized["sort"]
    if not isinstance(sort.get("column", ""), str) or sort.get("column", "") not in allowed_cols:
        sort["column"] = ""
    sort["ascending"] = bool(sort.get("ascending", True))

    lim = normalized["limit"]
    if isinstance(lim, float) and lim.is_integer():
        lim = int(lim)
    if isinstance(lim, int) and lim > 0:
        normalized["limit"] = lim
    else:
        normalized["limit"] = None

    return normalized




def generate_query_plan(user_query: str, schema_dict: dict) -> dict:
    """
    Uses Gemini to convert a natural language query into a structured
    pandas operation plan (JSON). Returns a parsed dict.
    """
    prompt = f"""You are a precise data query planner for a Learning & Development Dashboard.
You are given ONLY the dataset schema (column names and their data types).
You DO NOT have access to actual data values.

Your job: Convert the user query into a structured JSON for pandas operations.

═══════════════════════════════════════════════
INTENT RULES — pick EXACTLY one intent:
═══════════════════════════════════════════════

"unique"      → User wants distinct/unique/deduplicated values from a column.
                Examples: "list all training names", "show unique departments",
                "what training programs exist", "list trainings no duplicate",
                "how many types of training", "what courses are available".
                HOW: put the target column in "columns", set deduplicate=true.

"filter"      → User wants rows matching conditions.
                Examples: "show employees in HR", "trainings over 10 hours".

"aggregation" → User wants a single computed number.
                Examples: "total hours", "average training duration", "count employees".

"groupby"     → User wants aggregation broken down by a category.
                Examples: "total hours per department", "count of trainings by type".

"top_n"       → User wants highest/lowest/top/bottom N rows.
                Examples: "top 5 employees by hours", "lowest 3 departments".

"summary"     → User wants a broad overview with no specific filter.

"chat"        → User is greeting or chatting, NOT asking for data.
                Examples: "Hi", "Thanks", "How are you?".

═══════════════════════════════════════════════
STRICT RULES:
═══════════════════════════════════════════════
- Only return valid JSON. No text or explanation outside JSON.
- Use ONLY column names that appear EXACTLY in the schema below (case-sensitive).
- Do NOT assume or invent filter values not stated by the user.
- For "unique" intent: ALWAYS set deduplicate=true, put only the target column in "columns".
- For "top_n": set sort.ascending=false for highest/top, true for lowest/bottom.
  Set limit to N (default 10 if unspecified).
- For aggregation/groupby: populate "aggregation.operation". Otherwise set it to "none".
- Filters must ONLY contain conditions explicitly stated by the user.
- For text "==" filters, use the value exactly as the user stated it.

Schema (column_name: dtype):
{json.dumps(schema_dict, indent=2)}

{"" if not retrieve_relevant_categories(user_query) else "RAG Hints - the following exact strings exist in the DB and are semantically related to the user's intent. Do NOT hallucinate names, use these instead if they fit:\n" + str(retrieve_relevant_categories(user_query))}

User Query: "{user_query}"

Output JSON — return this exact structure, no extra keys:
{{
  "intent": "unique | filter | aggregation | groupby | top_n | summary | chat",
  "deduplicate": false,
  "columns": [],
  "filters": [
    {{"column": "", "operator": "== | > | < | >= | <= | contains", "value": ""}}
  ],
  "group_by": [],
  "aggregation": {{
    "column": "",
    "operation": "sum | mean | count | none"
  }},
  "sort": {{
    "column": "",
    "ascending": true
  }},
  "limit": null,
  "response_text": "Brief natural language explanation of what this query computes."
}}
"""
    try:
        if not _GEMINI_API_KEY:
            return {
                "intent": "error",
                "deduplicate": False,
                "response_text": "Missing GEMINI_API_KEY. Please set it in your environment or .env file.",
                "columns": [],
                "filters": [],
                "group_by": [],
                "aggregation": {"column": "", "operation": "none"},
                "sort": {"column": "", "ascending": True},
                "limit": None,
            }
        model = genai.GenerativeModel("gemini-2.5-flash")
        response = model.generate_content(
            prompt,
            generation_config=genai.GenerationConfig(
                response_mime_type="application/json",
                temperature=0.0,
            ),
        )
        plan = json.loads(response.text)
        return _normalize_plan(plan, schema_dict)
    except Exception as e:
        print(f"[Query Plan Error] {e}")
        return {
            "intent": "error",
            "deduplicate": False,
            "response_text": f"Error parsing query: {str(e)}",
            "columns": [],
            "filters": [],
            "group_by": [],
            "aggregation": {"column": "", "operation": "none"},
            "sort": {"column": "", "ascending": True},
            "limit": None,
        }


def execute_query_plan(df: pd.DataFrame, plan: dict) -> list:
    """
    Executes a structured query plan against a pandas DataFrame.
    Returns a list of clean, JSON-serializable dicts.

    Intent routing:
      unique      → drop_duplicates on selected column(s), sorted alphabetically
      aggregation → scalar result (sum/mean/count), returned immediately
      groupby     → grouped aggregation
      filter/top_n/summary → filtered rows + optional column select + sort + limit
      chat/error  → empty list (handled upstream by generate_rag_response)
    """
    intent = plan.get("intent", "filter")

    if intent in ("error", "chat"):
        return []

    try:
        result = df.copy()

        # ── Step 1: Apply Filters (shared by all intents) ─────────────────────
        for f in plan.get("filters", []):
            col = f.get("column", "")
            op  = f.get("operator", "")
            raw = f.get("value", "")

            if not col or col not in result.columns:
                continue

            cast_val = raw
            try:
                if result[col].dtype in ["int64", "float64"]:
                    cast_val = float(raw)
            except (ValueError, TypeError):
                pass

            if op == "==":
                if result[col].dtype in ["int64", "float64"]:
                    result = result[result[col] == cast_val]
                else:
                    result = result[
                        result[col].astype(str).str.lower() == str(raw).lower()
                    ]
            elif op == ">":
                result = result[pd.to_numeric(result[col], errors="coerce") > cast_val]
            elif op == "<":
                result = result[pd.to_numeric(result[col], errors="coerce") < cast_val]
            elif op == ">=":
                result = result[pd.to_numeric(result[col], errors="coerce") >= cast_val]
            elif op == "<=":
                result = result[pd.to_numeric(result[col], errors="coerce") <= cast_val]
            elif op == "contains":
                result = result[
                    result[col].astype(str).str.contains(str(raw), case=False, na=False)
                ]

        # ── Step 2: UNIQUE — deduplicate and return sorted distinct values ─────
        if intent == "unique" or plan.get("deduplicate") is True:
            cols = [c for c in plan.get("columns", []) if c in result.columns]
            if not cols:
                return [{"error": "No valid column specified for unique query."}]
            unique_result = (
                result[cols]
                .drop_duplicates()
                .dropna(subset=cols)
                .sort_values(by=cols[0])
                .reset_index(drop=True)
            )
            return unique_result.fillna("").to_dict(orient="records")

        # ── Step 3: SCALAR AGGREGATION (no group_by) ──────────────────────────
        group_by_cols = [c for c in plan.get("group_by", []) if c in result.columns]
        agg           = plan.get("aggregation", {})
        agg_col       = agg.get("column", "")
        agg_op        = agg.get("operation", "none")

        if not group_by_cols and agg_col in result.columns and agg_op in ("sum", "mean", "count"):
            series = result[agg_col]
            if agg_op == "sum":
                numeric_series = pd.to_numeric(series, errors="coerce")
                agg_value = round(float(numeric_series.sum()), 2)
            elif agg_op == "mean":
                numeric_series = pd.to_numeric(series, errors="coerce")
                agg_value = round(float(numeric_series.mean()), 2)
            else:
                agg_value = int(series.notna().sum())
            # Return immediately — column-select below would wipe this
            return [{agg_col: agg_value}]

        # ── Step 4: GROUPED AGGREGATION ───────────────────────────────────────
        if group_by_cols and agg_col in result.columns and agg_op in ("sum", "mean", "count"):
            if agg_op == "sum":
                result[agg_col] = pd.to_numeric(result[agg_col], errors="coerce")
                result = result.groupby(group_by_cols)[agg_col].sum().reset_index()
            elif agg_op == "mean":
                result[agg_col] = pd.to_numeric(result[agg_col], errors="coerce")
                result = result.groupby(group_by_cols)[agg_col].mean().round(2).reset_index()
            elif agg_op == "count":
                result = result.groupby(group_by_cols)[agg_col].apply(lambda s: int(s.notna().sum())).reset_index(name=agg_col)

        # ── Step 5: COLUMN SELECTION (filter / top_n / summary) ───────────────
        else:
            cols = [c for c in plan.get("columns", []) if c in result.columns]
            if cols:
                result = result[cols]

        # ── Step 6: SORT ──────────────────────────────────────────────────────
        sort     = plan.get("sort", {})
        sort_col = sort.get("column", "")
        if sort_col and sort_col in result.columns:
            result = result.sort_values(
                by=sort_col, ascending=sort.get("ascending", True)
            )

        # ── Step 7: LIMIT ─────────────────────────────────────────────────────
        limit = plan.get("limit")
        if isinstance(limit, int) and limit > 0:
            result = result.head(limit)
        elif intent == "top_n":
            result = result.head(10)          # default for top_n with no limit
        elif len(result) > 100:
            result = result.head(100)         # safety cap for all other intents

        return result.fillna("").to_dict(orient="records")

    except Exception as e:
        print(f"[Executor Error] {e}")
        return [{"error": f"Execution failed: {str(e)}"}]


def generate_rag_response(user_query: str, data_result: list, plan: dict) -> str:
    """
    Generates a concise, human-readable answer from the executed data result.

    For unique/list queries: builds a formatted bullet list directly (no LLM needed).
    For aggregations and small results: calls Gemini RAG.
    For large filtered results: formats inline without an LLM call.
    """
    intent = plan.get("intent", "filter")

    if intent == "chat":
        return plan.get("response_text", "Hello! How can I help you with the dashboard?")

    if not data_result:
        return (
            "I couldn't find any data matching your query. "
            "Try rephrasing or broadening your filters."
        )

    # ── UNIQUE: format as a clean HTML list directly ─────────────
    if intent == "unique" or plan.get("deduplicate") is True:
        cols = plan.get("columns", [])
        col  = cols[0] if cols else (list(data_result[0].keys())[0] if data_result else None)
        if col:
            names   = [str(row.get(col, "")).strip() for row in data_result if row.get(col)]
            count   = len(names)
            formatted = "".join(f"<li class='ml-4'>{n}</li>" for n in names)
            return (
                f"<div class='mb-2'>There are <strong>{count}</strong> unique values for <strong>{col}</strong>:</div>"
                f"<ol class='list-decimal list-outside text-sm space-y-1'>{formatted}</ol>"
            )

    # ── SCALAR AGGREGATION: format directly ──────────────────────────────────
    if intent == "aggregation" and len(data_result) == 1:
        row = data_result[0]
        if "error" not in row:
            key, val = next(iter(row.items()))
            return f"The **{key}** is **{val}**."

    # ── LARGE RESULT SETS: format inline, skip Gemini call ───────────────────
    if len(data_result) > 25:
        cols = list(data_result[0].keys()) if data_result else []
        primary_col = cols[0] if cols else None
        if primary_col:
            items = [str(row.get(primary_col, "")).strip() for row in data_result]
            return (
                f"Found **{len(items)}** results:\n\n"
                + "\n".join(f"• {item}" for item in items if item)
            )

    # ── SMALL RESULTS: use Gemini RAG for a natural language answer ───────────
    prompt = f"""You are a helpful analytics assistant for a Learning & Development Dashboard.
The user asked a question. We queried the dataset and retrieved the result below.

User Question: "{user_query}"
Query Summary: {plan.get("response_text", "")}

Data Result (JSON):
{json.dumps(data_result, indent=2)}

Instructions:
- Answer the user's question directly and conversationally.
- Use ONLY the numbers and values from the data above. Do NOT invent any figures.
- Be concise. Do not output raw JSON or markdown tables.
- If the result is a single number, state it clearly with helpful context.
- If the result has multiple rows, summarize the key highlights.
"""
    try:
        model = genai.GenerativeModel("gemini-2.5-flash")
        response = model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        print(f"[RAG Error] {e}")
        return "Here is the data: " + json.dumps(data_result)