# PredictAgent
### A Multi-Agent Framework for Predictive Maintenance Using Deep Learning

PredictAgent is a predictive maintenance research prototype that combines machine learning, deep learning, anomaly detection, and a multi-agent architecture to support industrial equipment monitoring and maintenance decision-making.

The system uses the MetroPT air compressor dataset for historical sensor analysis and coordinates specialized agents to propose maintenance actions.

## Key Features

- Historical sensor monitoring and visualization
- State-aware statistical anomaly detection
- Machine learning and deep learning experiments
- Multi-agent maintenance coordination
- Spare-part availability checks
- Technician availability checks
- Weather-aware maintenance recommendations
- Human-in-the-loop approval requirements
- Interactive Streamlit dashboard
- REST APIs using FastAPI

## Multi-Agent Architecture

PredictAgent coordinates six specialized agents:

1. **Diagnostics Agent:** Analyzes sensor readings and identifies statistical anomalies.
2. **Maintenance Planner Agent:** Identifies suitable maintenance windows.
3. **Parts Availability Agent:** Checks simulated spare-part inventory.
4. **Technician Availability Agent:** Checks simulated technician schedules and skills.
5. **Weather Agent:** Retrieves weather information to inform maintenance decisions.
6. **Alert Agent:** Generates simulated maintenance notifications.

A central orchestrator coordinates these agents to produce maintenance recommendations.

## Dataset

**MetroPT — Metro Air Compressor Dataset**

The dataset contains historical industrial air compressor measurements, including:

- TP2 and TP3 pressures
- H1
- DV pressure
- Reservoir pressure
- Oil temperature
- Motor current

The system uses historical measurements rather than a live industrial sensor feed.

**Dataset setup:** The dataset is not included in this GitHub repository. To run historical sensor analysis, obtain the MetroPT dataset separately and prepare the processed data using the scripts in `src/data/`.

The local processed dataset is expected at:

`data/processed/metropt_cleaned.csv`

The data directories and CSV files are excluded from Git to avoid uploading large datasets.

## Machine Learning and Deep Learning

The project includes experiments with:

- Logistic Regression
- Random Forest
- XGBoost
- Long Short-Term Memory (LSTM)
- Transformer-based models

The LSTM and Transformer implementations use **PyTorch**.

The current operational Diagnostics Agent uses state-aware z-score anomaly screening. The experimental ML/DL models are not yet integrated as validated production failure predictors.

## Technology Stack

| Component | Technology |
|---|---|
| Programming language | Python |
| Backend | FastAPI |
| Dashboard | Streamlit |
| Data processing | Pandas, NumPy |
| Machine learning | Scikit-learn, XGBoost |
| Deep learning | PyTorch |
| API communication | HTTP/REST |

## API Endpoints

| Endpoint | Purpose |
|---|---|
| `/` | API root |
| `/health` | Check backend health |
| `/diagnostics/analyze` | Analyze historical sensor readings |
| `/maintenance/plan` | Generate a provisional maintenance plan |
| `/predictagent/assess` | Run the coordinated assessment |
| `/predictagent/simulate-anomaly` | Demonstrate synthetic anomaly handling |
| `/sensors/history` | Retrieve historical sensor measurements |

## Dashboard

The Streamlit dashboard provides visualizations and interfaces for historical sensor monitoring, diagnostics, anomaly simulation, maintenance recommendations, and the multi-agent workflow.

## Safety and Limitations

- Anomaly detection is exploratory and does not confirm equipment failure.
- Failure prediction accuracy has not been validated for production use.
- Maintenance recommendations require human review and approval.
- Operations, inventory, and technician information are simulated.
- No actual maintenance is booked.
- No spare parts are reserved.
- Generated alerts are not sent to real recipients.
- Current weather observations do not represent forecasts for proposed maintenance windows.

## Project Status

**Research prototype — local integration testing completed.**

The backend, agent orchestration, synthetic anomaly workflow, and dashboard have been tested locally.

Further work is needed for validated failure prediction, real-world system integration, and production deployment.

## Installation and Setup

### 1. Clone the Repository

```bash
git clone <YOUR_REPOSITORY_URL>
cd PredictAgent
```

Replace `<YOUR_REPOSITORY_URL>` with the actual GitHub repository URL after publishing.

### 2. Create a Virtual Environment

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install Dependencies

```bash
python -m pip install -r requirements.txt
```

The project has been developed locally using Python 3.13. A clean installation of the pinned dependencies has not yet been independently verified.

### 4. Prepare the Dataset

Obtain the MetroPT dataset separately and use the project's data preparation scripts to generate the processed dataset.

Expected local file:

```text
data/processed/metropt_cleaned.csv
```

Historical monitoring and diagnostics features require this file.

### 5. Start the Backend

From the project root, start the FastAPI application:

```bash
uvicorn src.api.main:app --reload --port 8000
```

The API documentation will be available at:

http://127.0.0.1:8000/docs

### 6. Start the Dashboard

Open a second terminal, activate the virtual environment, and run:

```bash
streamlit run dashboard/app.py
```

The dashboard will be available at:

http://localhost:8501

### 7. Verify the Backend

```bash
curl http://127.0.0.1:8000/health
```

The health endpoint should return a successful response when the backend is running correctly.

## Running the Microservices

PredictAgent uses four supporting FastAPI microservices.

Open four separate VS Code terminals in the project root. Activate the virtual environment in each terminal.

### Operations Service — Port 8001

```bash
uvicorn services.operations_service.main:app --reload --port 8001
```

### Inventory Service — Port 8002

```bash
uvicorn services.inventory_service.main:app --reload --port 8002
```

### Technician Service — Port 8003

```bash
uvicorn services.technician_service.main:app --reload --port 8003
```

### Notification Service — Port 8004

```bash
uvicorn services.notification_service.main:app --reload --port 8004
```

### Complete System

For the full multi-agent workflow, run:

- Main FastAPI backend on port 8000
- Operations service on port 8001
- Inventory service on port 8002
- Technician service on port 8003
- Notification service on port 8004
- Streamlit dashboard on port 8501

Each application runs in its own terminal.

**Note:** These are local development commands. The supporting services use simulated operational data, and the system does not execute real-world maintenance actions.

## Future Improvements

- Validate failure prediction models against independent test data
- Integrate trained deep learning models into operational diagnostics
- Connect real industrial sensor feeds
- Integrate real maintenance management and inventory systems
- Implement authenticated supervisor approval workflows
- Add automated testing and deployment pipelines
- Deploy the system to cloud infrastructure
