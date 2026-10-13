
# Importing requests for communicating with the external Weather API
import requests


# Setting the external Open-Meteo API address
WEATHER_API_URL = "https://api.open-meteo.com/v1/forecast"


# Creating the Weather Agent
class WeatherAgent:

    # Initializing the agent with the Weather API address
    def __init__(self, api_url=WEATHER_API_URL):
        self.api_url = api_url

    # Fetching current weather conditions for an equipment location
    def get_weather(self, latitude, longitude):

        # Preparing the weather API parameters
        parameters = {
            "latitude": latitude,
            "longitude": longitude,
            "current": [
                "temperature_2m",
                "relative_humidity_2m",
                "precipitation",
                "wind_speed_10m"
            ],
            "timezone": "auto"
        }

        # Sending an HTTP request to the external Weather API
        response = requests.get(
            self.api_url,
            params=parameters,
            timeout=15
        )

        # Checking whether the API request was successful
        response.raise_for_status()

        # Converting the JSON response into a Python dictionary
        weather_data = response.json()

        # Extracting the current weather measurements
        current = weather_data["current"]

        # Checking weather-related maintenance precautions
        precautions = []

        # Flagging heavy precipitation conditions
        if current["precipitation"] >= 5:
            precautions.append("Heavy precipitation: inspect site access and electrical safety.")

        # Flagging high wind conditions
        if current["wind_speed_10m"] >= 40:
            precautions.append("High wind: review outdoor maintenance safety.")

        # Flagging high temperature conditions
        if current["temperature_2m"] >= 40:
            precautions.append("High temperature: review technician heat exposure.")

        # Returning the weather assessment
        return {
            "location": {
                "latitude": latitude,
                "longitude": longitude
            },
            "observation_time": current["time"],
            "temperature_celsius": current["temperature_2m"],
            "humidity_percent": current["relative_humidity_2m"],
            "precipitation_mm": current["precipitation"],
            "wind_speed_kmh": current["wind_speed_10m"],
            "weather_precautions": precautions,
            "status": (
                "PRECAUTIONS_RECOMMENDED"
                if precautions
                else "NO_WEATHER_PRECAUTIONS_FLAGGED"
            ),
            "data_source": "Open-Meteo API"
        }


# Testing the Weather Agent
if __name__ == "__main__":

    # Creating the Weather Agent
    agent = WeatherAgent()

    try:

        # Testing weather retrieval using approximate Mumbai coordinates
        result = agent.get_weather(
            latitude=19.0760,
            longitude=72.8777
        )

        # Displaying the weather assessment
        print("\nWeather Agent Result:")

        for key, value in result.items():
            print(f"{key}: {value}")

    # Handling external Weather API connection errors
    except requests.exceptions.RequestException as error:
        print("\nWeather API request failed:")
        print(error)
