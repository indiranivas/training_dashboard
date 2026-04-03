import app
from genai_helper import retrieve_relevant_categories

app.load_data() # This reads excel and builds index

print("\n--- Testing RAG Retrieval ---")
query1 = "Can you show me employees who took sales related courses?"
print("Query:", query1)
print("Matches:", retrieve_relevant_categories(query1))

query2 = "what did people in Human Resources or PR do?"
print("\nQuery:", query2)
print("Matches:", retrieve_relevant_categories(query2))
