
# Importing requests for communicating with the external Notification API
import requests


# Setting the address of the Notification API
NOTIFICATION_API_URL = "http://127.0.0.1:8004"


# Creating the Alert Agent
class AlertAgent:

    # Initializing the agent with the Notification API address
    def __init__(self, api_url=NOTIFICATION_API_URL):
        self.api_url = api_url.rstrip("/")

    # Creating and submitting a maintenance alert
    def send_alert(
        self,
        equipment_id,
        message,
        severity="HIGH",
        recipient="maintenance_team"
    ):

        # Creating the external Notification API endpoint
        url = f"{self.api_url}/notifications"

        # Preparing the maintenance alert information
        alert_data = {
            "equipment_id": equipment_id,
            "message": message,
            "severity": severity,
            "recipient": recipient
        }

        # Sending the alert to the external Notification API
        response = requests.post(
            url,
            json=alert_data,
            timeout=10
        )

        # Checking whether the API request was successful
        response.raise_for_status()

        # Converting the JSON response into a Python dictionary
        notification_data = response.json()

        # Returning the recorded notification details
        return {
            "notification_id": notification_data["notification_id"],
            "equipment_id": notification_data["equipment_id"],
            "message": notification_data["message"],
            "severity": notification_data["severity"],
            "recipient": notification_data["recipient"],
            "created_at": notification_data["created_at"],
            "status": notification_data["status"],
            "delivery_status": notification_data["delivery_status"]
        }


# Testing the Alert Agent
if __name__ == "__main__":

    # Creating the Alert Agent
    agent = AlertAgent()

    try:

        # Creating a demonstration maintenance alert
        result = agent.send_alert(
            equipment_id="COMP-001",
            message="Maintenance inspection recommended for air compressor.",
            severity="HIGH",
            recipient="maintenance_team"
        )

        # Displaying the recorded alert information
        print("\nAlert Agent Result:")

        for key, value in result.items():
            print(f"{key}: {value}")

    # Handling errors when communicating with the Notification API
    except requests.exceptions.RequestException as error:
        print("\nNotification API request failed:")
        print(error)
