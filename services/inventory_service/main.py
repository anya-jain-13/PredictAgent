
# Importing FastAPI for creating the Inventory API
from fastapi import FastAPI, HTTPException

# Creating the Inventory API application
app = FastAPI(title="PredictAgent Inventory API")

# Creating simulated spare-parts inventory data
inventory = {
    "FILTER-001": {
        "name": "Air Compressor Filter",
        "quantity": 8,
        "unit": "pieces"
    },
    "VALVE-001": {
        "name": "Air Compressor Valve",
        "quantity": 3,
        "unit": "pieces"
    },
    "SEAL-001": {
        "name": "Compressor Seal Kit",
        "quantity": 0,
        "unit": "kits"
    },
    "BELT-001": {
        "name": "Compressor Drive Belt",
        "quantity": 5,
        "unit": "pieces"
    }
}


# Checking whether the Inventory API is running
@app.get("/health")
def health():
    return {
        "status": "running",
        "service": "inventory",
        "data_source": "simulated"
    }


# Returning all spare parts in the inventory
@app.get("/parts")
def get_parts():
    return {
        "data_source": "simulated",
        "parts": inventory
    }


# Checking the availability of a requested spare part
@app.get("/availability/{part_id}")
def check_availability(part_id: str, required_quantity: int = 1):

    # Validating the requested quantity
    if required_quantity < 1:
        raise HTTPException(
            status_code=400,
            detail="Required quantity must be at least 1"
        )

    # Checking whether the requested part exists
    if part_id not in inventory:
        raise HTTPException(
            status_code=404,
            detail="Part not found"
        )

    # Retrieving the spare-part information
    part = inventory[part_id]

    # Checking whether sufficient stock is available
    available = part["quantity"] >= required_quantity

    # Returning the inventory availability result
    return {
        "part_id": part_id,
        "part_name": part["name"],
        "required_quantity": required_quantity,
        "available_quantity": part["quantity"],
        "available": available,
        "status": "IN_STOCK" if available else "OUT_OF_STOCK",
        "data_source": "simulated"
    }
