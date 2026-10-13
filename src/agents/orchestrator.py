
# Importing datetime for calculating maintenance duration
from datetime import datetime, timedelta

# Importing the existing PredictAgent components
from src.agents.maintenance_planner_agent import MaintenancePlannerAgent
from src.agents.parts_availability_agent import PartsAvailabilityAgent
from src.agents.technician_availability_agent import TechnicianAvailabilityAgent
from src.agents.weather_agent import WeatherAgent
from src.agents.alert_agent import AlertAgent
# Importing the Diagnostics Agent for sensor anomaly screening
from src.agents.diagnostics_agent import DiagnosticsAgent

# Creating the Central Orchestrator
class PredictAgentOrchestrator:

    # Initializing the individual agents
    def __init__(self):
        self.maintenance_agent = MaintenancePlannerAgent()
        self.parts_agent = PartsAvailabilityAgent()
        self.technician_agent = TechnicianAvailabilityAgent()
        self.weather_agent = WeatherAgent()
        self.alert_agent = AlertAgent()
        # Preparing the Diagnostics Agent for lazy initialization
        self.diagnostics_agent = None

        # Running sensor diagnostics through the Diagnostics Agent
    def run_diagnostics(self, recent_data):

        # Initializing the Diagnostics Agent only when required
        if self.diagnostics_agent is None:
            self.diagnostics_agent = DiagnosticsAgent()

        # Analyzing the supplied sensor readings
        diagnostic_result = self.diagnostics_agent.analyze(
            recent_data
        )

        # Determining whether the readings require investigation
        if diagnostic_result["anomaly_detected"]:
            recommendation = "SENSOR_ANOMALY_REVIEW_REQUIRED"
        else:
            recommendation = "NO_SENSOR_ANOMALY_DETECTED"

        # Returning the diagnostic findings and recommendation
        return {
            "diagnostics": diagnostic_result,
            "recommendation": recommendation,
            "maintenance_automatically_scheduled": False,
            "warning": (
                "Anomaly screening is exploratory. "
                "A detected anomaly does not confirm equipment failure."
            )
        }

        # Coordinating diagnostics-driven maintenance recommendations
    def assess_and_plan(
        self,
        recent_data,
        equipment_id,
        part_id,
        required_quantity=1,
        duration_minutes=90
    ):

        # Running diagnostics on the supplied sensor readings
        diagnostic_assessment = self.run_diagnostics(recent_data)

        # Selecting the diagnostic findings
        diagnostics = diagnostic_assessment["diagnostics"]

        # Checking whether unusual sensor readings were detected
        if not diagnostics["anomaly_detected"]:

            # Returning findings without initiating maintenance planning
            return {
                "equipment_id": equipment_id,
                "decision": "NO_SENSOR_ANOMALY_DETECTED",
                "diagnostics": diagnostics,
                "maintenance": None,
                "requires_human_review": False,
                "warning": (
                    "No anomaly detected by exploratory screening. "
                    "This does not guarantee equipment health."
                )
            }

        # Generating a provisional plan for human review
        maintenance_result = self.coordinate_maintenance(
            equipment_id=equipment_id,
            part_id=part_id,
            required_quantity=required_quantity,
            duration_minutes=duration_minutes
        )

        # Returning the diagnostic findings and proposed maintenance
        return {
            "equipment_id": equipment_id,
            "decision": "SENSOR_ANOMALY_REVIEW_REQUIRED",
            "diagnostics": diagnostics,
            "maintenance": maintenance_result,
            "requires_human_review": True,
            "warning": (
                "An anomaly is not proof of impending failure. "
                "Any maintenance proposal requires supervisor approval."
            )
        }
    
    # Coordinating the maintenance planning workflow
    def coordinate_maintenance(
        self,
        equipment_id,
        part_id,
        required_quantity=1,
        duration_minutes=90,
        latitude=19.0760,
        longitude=72.8777
    ):

        # Requesting available maintenance windows
        maintenance = self.maintenance_agent.plan_maintenance(
            equipment_id=equipment_id,
            duration_minutes=duration_minutes
        )

        # Checking replacement-part availability
        parts = self.parts_agent.check_part(
            part_id=part_id,
            required_quantity=required_quantity
        )

        # Retrieving current weather conditions
        weather = self.weather_agent.get_weather(
            latitude=latitude,
            longitude=longitude
        )

        # Preparing the technician availability results
        technician_checks = []

        # Preparing the selected maintenance plan
        selected_plan = None

        # Checking maintenance windows only when spare parts are available
        if parts["available"]:

            # Checking each suitable maintenance window
            for window in maintenance["suitable_windows"]:

                # Converting the maintenance start time
                start = datetime.fromisoformat(window["start"])

                # Calculating the required maintenance end time
                end = start + timedelta(minutes=duration_minutes)

                # Requesting available technicians for the proposed time
                technician_result = self.technician_agent.check_availability(
                    start_time=start.isoformat(),
                    end_time=end.isoformat(),
                    required_skill="AIR_COMPRESSOR"
                )

                # Recording the technician availability check
                technician_checks.append(technician_result)

                # Selecting the first window with a suitable technician
                if technician_result["count"] > 0:

                    # Preparing a provisional maintenance recommendation
                    selected_plan = {
                        "equipment_id": equipment_id,
                        "start_time": start.isoformat(),
                        "end_time": end.isoformat(),
                        "part_id": part_id,
                        "technician": technician_result[
                            "available_technicians"
                        ][0],
                        "service_disruption": window["service_disruption"],
                        "approval_status": "NOT_REQUESTED",
                        "status": "PROPOSED"
                    }

                    # Stopping after finding a feasible window
                    break

        # Determining whether a maintenance recommendation was found
        if selected_plan:
            decision = "MAINTENANCE_PLAN_PROPOSED"
            message = (
                f"Proposed maintenance for {equipment_id} "
                f"on {selected_plan['start_time']}. "
                "Supervisor approval is still required."
            )
            severity = "MEDIUM"
        else:
            decision = "MANUAL_REVIEW_REQUIRED"
            message = (
                f"No complete maintenance plan found for {equipment_id}. "
                "Review spare parts, technicians and maintenance windows."
            )
            severity = "HIGH"

        # Recording an alert through the external Notification API
        alert = self.alert_agent.send_alert(
            equipment_id=equipment_id,
            message=message,
            severity=severity,
            recipient="maintenance_team"
        )

        # Returning the coordinated maintenance recommendation
        return {
            "equipment_id": equipment_id,
            "decision": decision,
            "maintenance": maintenance,
            "parts": parts,
            "technician_checks": technician_checks,
            "weather": weather,
            "selected_plan": selected_plan,
            "alert": alert,
            "limitations": [
                "Operations, inventory and technician records are simulated.",
                "Weather data describes current conditions, not the future maintenance window.",
                "Weather precautions are informational and do not currently block scheduling.",
                "No maintenance booking or spare-part reservation has been made.",
                "Diagnostics-based failure prediction is not yet integrated."
            ]
        }


# Testing the Central Orchestrator
if __name__ == "__main__":

    # Creating the Central Orchestrator
    orchestrator = PredictAgentOrchestrator()

    try:

        # Coordinating maintenance for a sample air compressor
        result = orchestrator.coordinate_maintenance(
            equipment_id="COMP-001",
            part_id="FILTER-001",
            required_quantity=1,
            duration_minutes=90
        )

        # Displaying the overall maintenance decision
        print("\nCENTRAL ORCHESTRATOR RESULT")
        print("Equipment:", result["equipment_id"])
        print("Decision:", result["decision"])
        print("Selected plan:", result["selected_plan"])
        print("Parts status:", result["parts"]["status"])
        print("Technician checks:", len(result["technician_checks"]))
        print("Weather status:", result["weather"]["status"])
        print("Alert status:", result["alert"]["delivery_status"])
        print("Limitations:", result["limitations"])

    # Handling errors during the coordinated workflow
    except Exception as error:
        print("\nOrchestrator failed:")
        print(type(error).__name__, error)
