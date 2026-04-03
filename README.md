# Training Analytics Dashboard

A fully-dynamic, full-stack **Training Analytics Dashboard** built in Flask. Designed to process organizational learning and development metrics, evaluate comprehensive employee performance across departments, and natively aggregate training volumes cleanly via a modern, data-driven web interface.

## 🚀 Key Features

*   **Dynamic Interactive Dashboards**: Real-time KPI aggregation showcasing total session hours, learning coverage, and completion densities cleanly mapped with `Chart.js`.
*   **Centralized Employee Tracking**: A master `/employees` route natively identifying exact employee participation loads, completely stripping multi-session duplications to accurately group their total hours and courses.
*   **Granular Metrics & Analytics**: Evaluate precise Month-Over-Month training enrollments vs. completions, filtering success metrics vertically across 'Business Units' and 'Departments' natively from URL parameters.
*   **Data Lake Telemetry**: Live metric evaluation inside the settings page cleanly exposing DataFrame schemas directly imported from internal datasets.
*   **Production UI**: A seamless dark/light responsive interface leveraging Tailwind CSS, Material Icons, and Jinja2 loops injecting complex dataframes cleanly.

---

## 🛠 Tech Stack

*   **Backend**: Python, Flask
*   **Data Processing**: Pandas
*   **Frontend Template System**: Jinja2
*   **UI Components**: Tailwind CSS, Google Material Symbols
*   **Visualizations**: Chart.js 

---

## 📂 Project Structure

```text
├── app.py                # Main backend configuration, route routing, and Pandas data aggregations
├── data.xlsx             # Primary data source (Ensure your export maps to the columns listed below)
├── README.md             # This documentation
└── templates/            # Frontend Views mapped via Jinja2
    ├── Dashboard.html    # Primary KPI metrics & high-level charts
    ├── Employees.html    # Granular Employee logs with aggregation
    ├── Analytics.html    # Deep-dive Month-Over-Month & Departmental visualizations
    └── settings.html     # Internal System connection properties and dataset diagnostics
```

---

## ⚙️ Expected Dataset Layout

Ensure that your `data.xlsx` document possesses the following structured columns so that the Pandas engine properly reads and filters it:
*   `Emp ID` / `Employee Name`
*   `Department` / `Business Unit`
*   `Overall Training Duration (Planned Hrs)`
*   `Training Name`
*   `Training Status`
*   `Training Start Month` / `Start Date`

---

## 📥 Installation & Setup

1. **Clone the Repository** (or navigate to the workspace directory):
   ```bash
   cd work_with_excel
   ```

2. **Install Requirements**:
   Make sure you have Python 3 installed. Then, install the required packages.
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the Application Locally**:
   Spin up the built-in development server natively from the python binary.
   ```bash
   python app.py
   ```
   *The server will boot up via `http://127.0.0.1:5000`.*

## 🔐 Environment variables

- **Do not commit `.env`**. Put your key in a local `.env` file:

```bash
GEMINI_API_KEY=your_key_here
```

---

## 🖥 Navigation Guide

Once booted, interact with the platform natively:
*   **`/` (Dashboard)**: Gain a high-level view of core KPI evaluations and interactive graph structures evaluating learning coverage.
*   **`/employees`**: Filter aggregated employee lists directly.
*   **`/analytics`**: Identify deep chronological training deployments and structural gaps.
*   **`/settings`**: Check exact file ingestions, active column parameters, and system metrics.
