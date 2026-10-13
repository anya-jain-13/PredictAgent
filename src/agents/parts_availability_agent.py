
# Importing requests for communicating with the external Inventory API
import requests


# Setting the Inventory API address
INVENTORY_API_URL = "http://127.0.0.1:8002"


# Creating the Parts Availability Agent
class PartsAvailabilityAgent:

    # Initializing the agent with the Inventory API address
    def __init__(self, api_url=INVENTORY_API_URL):
        self.api_url = api_url.rstrip("/")

    # Fetching spare-part availability from the external Inventory API
    def check_part(self, part_id, required_quantity=1):

        # Creating the inventory availability endpoint
        url = f"{self.api_url}/availability/{part_id}"

        # Sending an HTTP request with the required quantity
        response = requests.get(
            url,
            params={"required_quantity": required_quantity},
            timeout=10
        )

        # Checking whether the API request was successful
        response.raise_for_status()

        # Converting the JSON response into a Python dictionary
        inventory_data = response.json()

        # Returning the spare-part availability decision
        return {
            "part_id": inventory_data["part_id"],
            "part_name": inventory_data["part_name"],
            "required_quantity": inventory_data["required_quantity"],
            "available_quantity": inventory_data["available_quantity"],
            "available": inventory_data["available"],
            "status": inventory_data["status"],
            "data_source": inventory_data["data_source"]
        }


# Testing the Parts Availability Agent
if __name__ == "__main__":

    # Creating the Parts Availability Agent
    agent = PartsAvailabilityAgent()

    try:
        # Checking whether the required filter is available
        result = agent.check_part(
            part_id="FILTER-001",
            required_quantity=1
        )

        # Displaying the availability result
        print("\nParts Availability Agent Result:")

        for key, value in result.items():
            print(f"{key}: {value}")

    # Handling connection and API errors
    except requests.exceptions.RequestException as error:
        print("\nInventory API request failed:")
        print(error)
