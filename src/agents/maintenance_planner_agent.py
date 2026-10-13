
# Importing requests for communicating with the external Operations API
import requests

# Importing datetime for checking maintenance schedules
from datetime import datetime


# Setting the address of the separately running Operations API
OPERATIONS_API_URL = "http://127.0.0.1:8001"


# Creating the Maintenance Planner Agent
class MaintenancePlannerAgent:

    # Initializing the agent with the Operations API address
    def __init__(self, api_url=OPERATIONS_API_URL):
        self.api_url = api_url.rstrip("/")

    # Fetching maintenance windows from the external Operations API
    def get_maintenance_windows(self, equipment_id):

        # Creating the endpoint for the requested equipment
        url = f"{self.api_url}/maintenance-windows/{equipment_id}"

        # Sending an HTTP request to the Operations API
        response = requests.get(url, timeout=10)

        # Checking whether the API request was successful
        response.raise_for_status()

        # Converting the received JSON response into a Python dictionary
        return response.json()

    # Finding maintenance windows that satisfy the required duration
    def plan_maintenance(self, equipment_id, duration_minutes=60):

        # Validating the requested maintenance duration
        if duration_minutes <= 0:
            raise ValueError("Maintenance duration must be positive.")

        # Retrieving operational information from the external API
        operations_data = self.get_maintenance_windows(equipment_id)

        # Creating a list for storing suitable maintenance windows
        suitable_windows = []

        # Checking every maintenance window returned by the API
        for window in operations_data["maintenance_windows"]:

            # Converting the window timestamps into datetime objects
            start = datetime.fromisoformat(window["start"])
            end = datetime.fromisoformat(window["end"])

            # Calculating the available maintenance duration in minutes
            available_minutes = (
                end - start
            ).total_seconds() / 60

            # Selecting windows with sufficient available time
            if available_minutes >= duration_minutes:

                # Storing the suitable maintenance window
                suitable_windows.append({
                    "start": window["start"],
                    "end": window["end"],
                    "available_minutes": available_minutes,
                    "service_disruption": window["service_disruption"]
                })

        # Determining whether suitable maintenance windows were found
        status = (
            "WINDOWS_FOUND"
            if suitable_windows
            else "NO_SUITABLE_WINDOWS"
        )

        # Returning the proposed maintenance plan
        return {
            "equipment_id": equipment_id,
            "required_duration_minutes": duration_minutes,
            "suitable_windows": suitable_windows,
            "restrictions": operations_data["restrictions"],
            "data_source": operations_data["data_source"],
            "status": status,
            "approval_status": "NOT_REQUESTED"
        }


# Running a demonstration of the Maintenance Planner Agent
if __name__ == "__main__":

    # Creating the Maintenance Planner Agent
    agent = MaintenancePlannerAgent()

    try:

        # Requesting a 90-minute maintenance plan for COMP-001
        result = agent.plan_maintenance(
            equipment_id="COMP-001",
            duration_minutes=90
        )

        # Displaying the maintenance planning results
        print("\nMaintenance Planner Agent Result:")

        for key, value in result.items():
            print(f"{key}: {value}")

    # Handling errors when the Operations API cannot be reached
    except requests.exceptions.RequestException as error:
        print("\nOperations API request failed:")
        print(error)
