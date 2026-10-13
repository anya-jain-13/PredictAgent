
# Importing FastAPI for building the main application API
from fastapi import FastAPI, HTTPException

# Importing the Central Orchestrator
from src.agents.orchestrator import PredictAgentOrchestrator

# Importing the Diagnostics Agent
from src.agents.diagnostics_agent import DiagnosticsAgent


# Creating the PredictAgent FastAPI application
app = FastAPI(
    title="PredictAgent API",
    description="Multi-Agent Predictive Maintenance System",
    version="1.0.0"
)


# Creating the home endpoint
@app.get("/")
def home():

    # Returning application information
    return {
        "application": "PredictAgent",
        "status": "running",
        "description": "Multi-Agent Predictive Maintenance System"
    }


# Creating the health-check endpoint
@app.get("/health")
def health_check():

    # Returning the API health status
    return {
        "status": "healthy",
        "service": "PredictAgent Main API"
    }


# Creating an endpoint for maintenance planning
@app.get("/maintenance/plan")
def generate_maintenance_plan(
    equipment_id: str = "COMP-001",
    part_id: str = "FILTER-001",
    required_quantity: int = 1,
    duration_minutes: int = 90
):

    # Validating maintenance requirements
    if required_quantity < 1 or duration_minutes < 1:
        raise HTTPException(
            status_code=400,
            detail="Quantity and duration must be positive."
        )

    # Initializing the Central Orchestrator
    orchestrator = PredictAgentOrchestrator()

    # Coordinating maintenance planning across agents
    result = orchestrator.coordinate_maintenance(
        equipment_id=equipment_id,
        part_id=part_id,
        required_quantity=required_quantity,
        duration_minutes=duration_minutes
    )

    # Returning the complete maintenance recommendation
    return result


# Creating an endpoint for historical sensor diagnostics
@app.get("/diagnostics/analyze")
def analyze_sensor_data():

    # Initializing the Central Orchestrator
    orchestrator = PredictAgentOrchestrator()

    # Loading historical MetroPT sensor data
    diagnostic_agent = DiagnosticsAgent()

    # Selecting the latest 360 historical readings
    recent_data = diagnostic_agent.data.tail(360).copy()

    # Reusing the loaded Diagnostics Agent
    orchestrator.diagnostics_agent = diagnostic_agent

    # Running diagnostics through the Central Orchestrator
    result = orchestrator.run_diagnostics(recent_data)

    # Returning the diagnostic findings
    return {
        "equipment_id": "COMP-001",
        "data_source": "Historical MetroPT dataset",
        "sensor_readings_analyzed": len(recent_data),
        "result": result
    }


# Creating an endpoint for the complete multi-agent workflow
@app.get("/predictagent/assess")
def assess_equipment(
    equipment_id: str = "COMP-001",
    part_id: str = "FILTER-001"
):

    # Initializing the Central Orchestrator
    orchestrator = PredictAgentOrchestrator()

    # Loading the historical MetroPT dataset
    diagnostic_agent = DiagnosticsAgent()

    # Selecting the latest 360 historical sensor readings
    recent_data = diagnostic_agent.data.tail(360).copy()

    # Reusing the initialized Diagnostics Agent
    orchestrator.diagnostics_agent = diagnostic_agent

    # Running diagnostics-driven maintenance assessment
    result = orchestrator.assess_and_plan(
        recent_data=recent_data,
        equipment_id=equipment_id,
        part_id=part_id
    )

    # Returning the complete assessment
    return {
        "data_source": "Historical MetroPT dataset",
        "sensor_readings_analyzed": len(recent_data),
        "assessment": result
    }


# Creating an endpoint for demonstrating anomaly-driven maintenance
@app.get("/predictagent/simulate-anomaly")
def simulate_anomaly():

    # Initializing the Central Orchestrator
    orchestrator = PredictAgentOrchestrator()

    # Loading historical compressor sensor readings
    diagnostic_agent = DiagnosticsAgent()
    orchestrator.diagnostics_agent = diagnostic_agent

    # Selecting historical readings in the LOADED operating state
    data = diagnostic_agent.data

    loaded_data = data[
        (data["Motor_current"] >= 1.0) &
        (data["DV_eletric"] == 1)
    ].tail(360).copy()

    # Checking whether suitable readings are available
    if loaded_data.empty:
        raise HTTPException(
            status_code=404,
            detail="No suitable sensor readings found."
        )

    # Retrieving the historical temperature reference
    reference = diagnostic_agent.reference["LOADED"]

    # Injecting a synthetic oil-temperature anomaly
    loaded_data.loc[
        loaded_data.index[-1],
        "Oil_temperature"
    ] = (
        reference["means"]["Oil_temperature"]
        + 10 * reference["stds"]["Oil_temperature"]
    )

    # Running the complete multi-agent workflow
    result = orchestrator.assess_and_plan(
        recent_data=loaded_data,
        equipment_id="COMP-001",
        part_id="FILTER-001"
    )

    # Returning the simulation results
    return {
        "simulation": True,
        "data_source": "Historical MetroPT data with synthetic anomaly",
        "sensor_readings_analyzed": len(loaded_data),
        "assessment": result,
        "warning": (
            "Synthetic demonstration only. "
            "No validated failure prediction or real maintenance booking."
        )
    }

# Creating an endpoint for historical sensor visualization
@app.get("/sensors/history")
def get_sensor_history(limit: int = 500):

    # Validating the requested number of readings
    if limit < 1 or limit > 5000:
        raise HTTPException(
            status_code=400,
            detail="Limit must be between 1 and 5000."
        )

    # Loading historical MetroPT sensor readings
    import pandas as pd

    # Selecting the required sensor columns
    sensor_columns = [
        "timestamp",
        "Oil_temperature",
        "Motor_current",
        "TP2",
        "TP3",
        "H1",
        "DV_pressure",
        "Reservoirs"
    ]

    # Reading the selected columns from the dataset
    data = pd.read_csv(
        "data/processed/metropt_cleaned.csv",
        usecols=sensor_columns
    )

    # Selecting the latest historical readings
    recent_data = data.tail(limit).copy()

    # Converting readings into JSON-compatible records
    records = recent_data.to_dict(orient="records")

    # Returning the sensor history
    return {
        "equipment_id": "COMP-001",
        "data_source": "Historical MetroPT dataset",
        "total_readings": len(records),
        "sensors": sensor_columns[1:],
        "readings": records
    }
