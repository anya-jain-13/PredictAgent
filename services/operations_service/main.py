
# Importing FastAPI for creating the Operations API
from fastapi import FastAPI, HTTPException

# Importing datetime utilities for generating maintenance schedules
from datetime import datetime, timedelta


# Creating the Operations API application
app = FastAPI(
    title="PredictAgent Operations API",
    description="Simulated operational schedules for maintenance planning",
    version="1.0.0"
)


# Creating simulated equipment information
EQUIPMENT = {
    "COMP-001": {
        "equipment_name": "Metro Air Compressor 001",
        "location": "Metro Maintenance Depot A",
        "operational_status": "ACTIVE",
        "criticality": "HIGH",
        "restrictions": [
            "Maintenance requires supervisor approval",
            "Equipment must be isolated before repair"
        ]
    },
    "COMP-002": {
        "equipment_name": "Metro Air Compressor 002",
        "location": "Metro Maintenance Depot B",
        "operational_status": "STANDBY",
        "criticality": "MEDIUM",
        "restrictions": [
            "Maintenance requires equipment isolation"
        ]
    }
}


# Generating future maintenance windows automatically
def generate_maintenance_windows(equipment_id):

    # Getting the current local date and time
    now = datetime.now()

    # Creating a list for storing maintenance windows
    windows = []

    # Selecting maintenance timings based on equipment
    if equipment_id == "COMP-001":
        schedules = [(1, 0, 120), (0, 30, 210)]
    else:
        schedules = [(2, 0, 180)]

    # Generating maintenance windows on future calendar days
    for day_offset, (hour, minute, duration) in enumerate(schedules, start=1):

        # Calculating the future maintenance date
        future_date = now.date() + timedelta(days=day_offset)

        # Creating the maintenance starting time
        start = datetime.combine(
            future_date,
            datetime.min.time()
        ).replace(hour=hour, minute=minute)

        # Calculating the maintenance ending time
        end = start + timedelta(minutes=duration)

        # Storing the maintenance window
        windows.append({
            "start": start.isoformat(),
            "end": end.isoformat(),
            "service_disruption": "LOW"
        })

    return windows


# Creating an endpoint for checking API availability
@app.get("/health")
def health_check():

    # Returning the current API status
    return {
        "service": "operations",
        "status": "healthy",
        "data_source": "simulated"
    }


# Creating an endpoint for listing all available equipment
@app.get("/equipment")
def list_equipment():

    # Returning the identifiers of simulated equipment
    return {
        "equipment_ids": list(EQUIPMENT.keys())
    }


# Creating an endpoint for retrieving equipment information
@app.get("/equipment/{equipment_id}")
def get_equipment(equipment_id: str):

    # Checking whether the requested equipment exists
    if equipment_id not in EQUIPMENT:
        raise HTTPException(
            status_code=404,
            detail="Equipment not found"
        )

    # Returning the requested equipment details
    return {
        "equipment_id": equipment_id,
        "data_source": "simulated",
        **EQUIPMENT[equipment_id],
        "maintenance_windows": generate_maintenance_windows(equipment_id)
    }


# Creating an endpoint for retrieving maintenance windows
@app.get("/maintenance-windows/{equipment_id}")
def get_maintenance_windows(equipment_id: str):

    # Checking whether the equipment identifier exists
    if equipment_id not in EQUIPMENT:
        raise HTTPException(
            status_code=404,
            detail="Equipment not found"
        )

    # Generating the future maintenance windows
    windows = generate_maintenance_windows(equipment_id)

    # Validating the maintenance window durations
    for window in windows:

        start = datetime.fromisoformat(window["start"])
        end = datetime.fromisoformat(window["end"])

        if end <= start:
            raise HTTPException(
                status_code=500,
                detail="Invalid maintenance window configuration"
            )

    # Returning the maintenance windows and restrictions
    return {
        "equipment_id": equipment_id,
        "data_source": "simulated",
        "maintenance_windows": windows,
        "restrictions": EQUIPMENT[equipment_id]["restrictions"]
    }
