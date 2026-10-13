
# Importing FastAPI for creating the Workforce API
from fastapi import FastAPI, HTTPException

# Importing datetime utilities for managing technician shifts
from datetime import datetime, timedelta


# Creating the Workforce API application
app = FastAPI(title="PredictAgent Workforce API")


# Creating simulated technician information
technicians = {
    "TECH-001": {
        "name": "Rahul Sharma",
        "skills": ["AIR_COMPRESSOR", "MECHANICAL"],
        "available": True,
        "shift_start_hour": 0,
        "shift_end_hour": 6
    },
    "TECH-002": {
        "name": "Priya Mehta",
        "skills": ["ELECTRICAL", "AIR_COMPRESSOR"],
        "available": True,
        "shift_start_hour": 1,
        "shift_end_hour": 8
    },
    "TECH-003": {
        "name": "Arjun Patel",
        "skills": ["MECHANICAL", "HVAC"],
        "available": False,
        "shift_start_hour": 0,
        "shift_end_hour": 8
    }
}


# Generating simulated technician shifts for a requested date
def generate_shift(technician, shift_date):

    # Creating the beginning of the requested date
    midnight = datetime.combine(
        shift_date,
        datetime.min.time()
    )

    # Calculating the technician's shift start
    shift_start = midnight + timedelta(
        hours=technician["shift_start_hour"]
    )

    # Calculating the technician's shift end
    shift_end = midnight + timedelta(
        hours=technician["shift_end_hour"]
    )

    # Returning the generated shift times
    return shift_start, shift_end


# Checking whether the Workforce API is running
@app.get("/health")
def health():
    return {
        "status": "running",
        "service": "workforce",
        "data_source": "simulated"
    }


# Returning all registered technicians
@app.get("/technicians")
def get_technicians():

    # Returning simulated technician records
    return {
        "data_source": "simulated",
        "technicians": technicians
    }


# Retrieving information about a particular technician
@app.get("/technicians/{technician_id}")
def get_technician(technician_id: str):

    # Checking whether the technician exists
    if technician_id not in technicians:
        raise HTTPException(
            status_code=404,
            detail="Technician not found"
        )

    # Returning the technician information
    return {
        "technician_id": technician_id,
        **technicians[technician_id],
        "data_source": "simulated"
    }


# Finding technicians available during a maintenance window
@app.get("/available-technicians")
def available_technicians(
    start_time: str,
    end_time: str,
    required_skill: str = "AIR_COMPRESSOR"
):

    # Converting the requested maintenance timestamps
    try:
        maintenance_start = datetime.fromisoformat(start_time)
        maintenance_end = datetime.fromisoformat(end_time)
    except ValueError:
        raise HTTPException(
            status_code=400,
            detail="Invalid datetime format"
        )

    # Validating the requested maintenance duration
    if maintenance_end <= maintenance_start:
        raise HTTPException(
            status_code=400,
            detail="End time must be after start time"
        )

    # Rejecting timezone-aware timestamps for this local-time simulation
    if maintenance_start.tzinfo or maintenance_end.tzinfo:
        raise HTTPException(
            status_code=400,
            detail="Use local timestamps without timezone offsets"
        )

    # Rejecting maintenance requests in the past
    if maintenance_start < datetime.now():
        raise HTTPException(
            status_code=400,
            detail="Maintenance start time must be in the future"
        )

    # Creating a list for storing matching technicians
    suitable_technicians = []

    # Checking each technician's skills and shift
    for technician_id, technician in technicians.items():

        # Generating a shift for the maintenance date
        shift_start, shift_end = generate_shift(
            technician,
            maintenance_start.date()
        )

        # Checking whether the required skill is present
        has_skill = required_skill in technician["skills"]

        # Checking whether the technician is marked available
        is_available = technician["available"]

        # Checking whether the entire maintenance fits inside the shift
        shift_matches = (
            shift_start <= maintenance_start
            and shift_end >= maintenance_end
        )

        # Selecting technicians satisfying every requirement
        if has_skill and is_available and shift_matches:
            suitable_technicians.append({
                "technician_id": technician_id,
                "name": technician["name"],
                "skills": technician["skills"],
                "shift_start": shift_start.isoformat(),
                "shift_end": shift_end.isoformat()
            })

    # Returning technician availability results
    return {
        "required_skill": required_skill,
        "maintenance_start": start_time,
        "maintenance_end": end_time,
        "available_technicians": suitable_technicians,
        "count": len(suitable_technicians),
        "status": (
            "TECHNICIANS_AVAILABLE"
            if suitable_technicians
            else "NO_TECHNICIANS_AVAILABLE"
        ),
        "data_source": "simulated"
    }
