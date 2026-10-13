
# Importing requests for communicating with the external Technician API
import requests


# Setting the address of the Technician API
TECHNICIAN_API_URL = "http://127.0.0.1:8003"


# Creating the Technician Availability Agent
class TechnicianAvailabilityAgent:

    # Initializing the agent with the Technician API address
    def __init__(self, api_url=TECHNICIAN_API_URL):
        self.api_url = api_url.rstrip("/")

    # Fetching technicians available for a maintenance window
    def check_availability(
        self,
        start_time,
        end_time,
        required_skill="AIR_COMPRESSOR"
    ):

        # Creating the external API endpoint
        url = f"{self.api_url}/available-technicians"

        # Preparing the maintenance requirements
        parameters = {
            "start_time": start_time,
            "end_time": end_time,
            "required_skill": required_skill
        }

        # Sending the request to the external Technician API
        response = requests.get(
            url,
            params=parameters,
            timeout=10
        )

        # Checking whether the API request was successful
        response.raise_for_status()

        # Converting the JSON response into a Python dictionary
        technician_data = response.json()

        # Returning the technician availability result
        return {
            "required_skill": technician_data["required_skill"],
            "maintenance_start": technician_data["maintenance_start"],
            "maintenance_end": technician_data["maintenance_end"],
            "available_technicians": technician_data["available_technicians"],
            "count": technician_data["count"],
            "status": technician_data["status"],
            "data_source": technician_data["data_source"]
        }


# Testing the Technician Availability Agent
if __name__ == "__main__":

    # Creating the Technician Availability Agent
    agent = TechnicianAvailabilityAgent()

    try:

        # Checking technician availability for a 90-minute maintenance window
        result = agent.check_availability(
            start_time="2026-10-09T01:00:00",
            end_time="2026-10-09T02:30:00",
            required_skill="AIR_COMPRESSOR"
        )

        # Displaying the technician availability result
        print("\nTechnician Availability Agent Result:")

        for key, value in result.items():
            print(f"{key}: {value}")

    # Handling external API connection errors
    except requests.exceptions.RequestException as error:
        print("\nTechnician API request failed:")
        print(error)
