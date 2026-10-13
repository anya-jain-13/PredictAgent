# Importing libraries for the PredictAgent dashboard
import pandas as pd
import requests
import streamlit as st

# Configuring the dashboard
st.set_page_config(page_title="PredictAgent | Operations Console", page_icon="🔧", layout="wide", initial_sidebar_state="expanded")
API_BASE_URL = "http://127.0.0.1:8000"

# Defining a helper for backend requests
def fetch_api(endpoint, params=None):
    # Requesting data from the FastAPI backend
    response = requests.get(f"{API_BASE_URL}{endpoint}", params=params, timeout=120)
    response.raise_for_status()
    return response.json()

# Defining reusable display helpers
def show_error(error, context):
    # Displaying API or response errors
    st.error(f"{context}: {error}")


def show_plan(plan):
    # Displaying a proposed maintenance plan without implying a booking
    if not isinstance(plan, dict):
        st.warning("No complete maintenance proposal was returned. Manual review is required.")
        return
    left, right = st.columns(2)
    with left:
        st.write("**Equipment:**", plan.get("equipment_id", "—"))
        st.write("**Start:**", plan.get("start_time", "—"))
        st.write("**End:**", plan.get("end_time", "—"))
        st.write("**Spare part:**", plan.get("part_id", "—"))
    with right:
        technician = plan.get("technician") or {}
        st.write("**Technician:**", technician.get("name", "—") if isinstance(technician, dict) else str(technician))
        st.write("**Service disruption:**", plan.get("service_disruption", "—"))
        st.write("**Approval:**", plan.get("approval_status", "NOT_REQUESTED"))
        st.write("**Status:**", plan.get("status", "PROPOSED"))
    st.caption("Provisional recommendation only. No approval, technician booking, or parts reservation has been made.")


def show_diagnostics(diagnostics, reading_count=None):
    # Displaying diagnostic summary metrics
    cols = st.columns(3)
    cols[0].metric("Operating state", diagnostics.get("operating_state", "—"))
    cols[1].metric("Flagged sensors", diagnostics.get("anomaly_count", "—"))
    cols[2].metric("Readings analyzed" if reading_count is not None else "Human review", reading_count if reading_count is not None else ("Required" if diagnostics.get("anomaly_detected") else "Not flagged"))
    if diagnostics.get("anomaly_detected"):
        st.warning("Unusual sensor values flagged for human review.")
    else:
        st.info("No anomaly flagged in this sample; this does not establish equipment health.")
    st.write("**Operating-state baseline:**", diagnostics.get("operating_state", "—"))
    st.write("**Method:**", diagnostics.get("diagnostic_method", "State-aware z-score screening"))
    st.write("**Reading timestamp:**", diagnostics.get("timestamp", "—"))
    anomalies = diagnostics.get("anomalies") or {}
    if anomalies:
        st.subheader("Flagged sensor readings")
        st.dataframe(pd.DataFrame.from_dict(anomalies, orient="index"), use_container_width=True)


# Styling the application without changing the backend
st.markdown("""
<style>
.block-container {padding-top: 1.6rem; padding-bottom: 2.5rem; max-width: 1450px;}
[data-testid="stSidebar"] {border-right: 1px solid rgba(125,125,125,.18);}
[data-testid="stMetric"] {border: 1px solid rgba(125,125,125,.2); border-radius: 12px; padding: 14px 16px;}
h1, h2, h3 {letter-spacing: -.025em;}
</style>
""", unsafe_allow_html=True)

# Defining dashboard pages
PAGES = [
    "Overview",
    "Backend Connection",
    "Sensor Diagnostics",
    "Maintenance Planning",
    "Complete Assessment",
    "Anomaly Simulation",
    "Historical Sensor Monitoring",
    "Anomaly Visualization",
]

# Displaying navigation and shared inputs
with st.sidebar:
    st.title("🔧 PredictAgent")
    st.caption("Multi-Agent Maintenance Console")
    page = st.radio("Navigation", PAGES, label_visibility="collapsed")
    st.divider()
    st.subheader("Equipment settings")
    equipment_id = st.text_input("Equipment ID", value="COMP-001")
    part_id = st.text_input("Required spare part", value="FILTER-001")
    duration_minutes = st.number_input("Maintenance duration (minutes)", min_value=30, max_value=480, value=90, step=30)
    st.divider()
    st.caption("Historical MetroPT data • Simulated resource services • No live monitoring")

# Initializing persistent results across sidebar navigation
for state_key in ["health_result", "diagnostics_result", "maintenance_result", "assessment_result", "simulation_result", "sensor_history", "anomaly_chart_data"]:
    if state_key not in st.session_state:
        st.session_state[state_key] = None

# Displaying a consistent header
st.title("PredictAgent")
st.caption("State-aware diagnostics · Multi-agent coordination · Provisional maintenance recommendations")

if page == "Overview":
    st.header("Operations Overview")
    st.write("Explore historical compressor readings, screen unusual behavior, and coordinate simulated maintenance resources from one dashboard.")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Diagnostic method", "Z-score")
    col2.metric("Detection rule", "|Z| ≥ 3")
    col3.metric("Data source", "MetroPT")
    col4.metric("Maintenance", "Proposed only")
    st.subheader("System workflow")
    st.markdown("**Historical sensors** → **Diagnostics Agent** → **Orchestrator** → **Maintenance / Parts / Technician / Weather / Alert Agents** → **Human review**")
    st.subheader("Recent results in this session")
    recent = {
        "Backend check": st.session_state.health_result is not None,
        "Diagnostics": st.session_state.diagnostics_result is not None,
        "Maintenance proposal": st.session_state.maintenance_result is not None,
        "Full assessment": st.session_state.assessment_result is not None,
        "Synthetic simulation": st.session_state.simulation_result is not None,
        "Historical sensor charts": st.session_state.sensor_history is not None,
        "Anomaly visualization": st.session_state.anomaly_chart_data is not None,
    }
    st.dataframe(pd.DataFrame([{"Module": k, "Run this session": "Yes" if v else "Not yet"} for k, v in recent.items()]), hide_index=True, use_container_width=True)
    st.info("This is a demonstration, not a validated failure-probability predictor. Anomaly flags require human interpretation.")

elif page == "Backend Connection":
    st.header("Backend Connection")
    st.write("Checking the FastAPI health endpoint.")
    if st.button("Check API Connection", type="primary"):
        try:
            with st.spinner("Contacting FastAPI..."):
                st.session_state.health_result = fetch_api("/health")
        except requests.exceptions.RequestException as error:
            show_error(error, "Backend connection failed")
    if st.session_state.health_result is not None:
        st.success("FastAPI responded successfully.")
        st.json(st.session_state.health_result)

elif page == "Sensor Diagnostics":
    st.header("Sensor Diagnostics")
    st.write("Screening the latest historical sensor reading against a state-specific baseline.")
    if st.button("Analyze Sensor Data", type="primary"):
        try:
            with st.spinner("Running Diagnostics Agent..."):
                st.session_state.diagnostics_result = fetch_api("/diagnostics/analyze")
        except requests.exceptions.RequestException as error:
            show_error(error, "Diagnostics request failed")
    if st.session_state.diagnostics_result is not None:
        data = st.session_state.diagnostics_result
        try:
            show_diagnostics(data["result"]["diagnostics"], data.get("sensor_readings_analyzed"))
            with st.expander("Complete diagnostic response"):
                st.json(data)
        except (KeyError, TypeError, ValueError) as error:
            show_error(error, "Unexpected diagnostics response")

elif page == "Maintenance Planning":
    st.header("Maintenance Planning")
    st.write("Coordinating simulated operational, spare-parts, technician, weather and alert services.")
    st.caption("Adjust equipment, part, and duration in the sidebar before generating a proposal.")
    if st.button("Generate Maintenance Plan", type="primary"):
        try:
            with st.spinner("Coordinating maintenance agents..."):
                st.session_state.maintenance_result = fetch_api("/maintenance/plan", params={"equipment_id": equipment_id, "part_id": part_id, "duration_minutes": duration_minutes})
        except requests.exceptions.RequestException as error:
            show_error(error, "Maintenance planning failed")
    if st.session_state.maintenance_result is not None:
        result = st.session_state.maintenance_result
        st.subheader("Agent decision")
        st.write(result.get("decision", "No decision returned."))
        st.subheader("Selected proposal")
        show_plan(result.get("selected_plan"))
        tabs = st.tabs(["Parts", "Weather", "Alert", "Full response"])
        with tabs[0]:
            st.json(result.get("parts", {}))
        with tabs[1]:
            st.json(result.get("weather", {}))
        with tabs[2]:
            st.json(result.get("alert", {}))
        with tabs[3]:
            st.json(result)
        st.caption("Resource records are simulated. Weather describes current conditions and may not represent conditions at a proposed future maintenance time.")

elif page == "Complete Assessment":
    st.header("Complete PredictAgent Assessment")
    st.write("Running historical anomaly screening, followed by maintenance coordination only if the orchestrator flags an anomaly.")
    if st.button("Run Complete Assessment", type="primary"):
        try:
            with st.spinner("Running assessment..."):
                st.session_state.assessment_result = fetch_api("/predictagent/assess", params={"equipment_id": equipment_id, "part_id": part_id})
        except requests.exceptions.RequestException as error:
            show_error(error, "Assessment failed")
    if st.session_state.assessment_result is not None:
        data = st.session_state.assessment_result
        try:
            assessment = data["assessment"]
            show_diagnostics(assessment["diagnostics"])
            st.subheader("Assessment decision")
            st.write(assessment.get("decision", "—"))
            st.metric("Human review required", "Yes" if assessment.get("requires_human_review") else "No")
            if assessment.get("maintenance"):
                st.subheader("Maintenance coordination")
                maintenance = assessment["maintenance"]
                st.write(maintenance.get("decision", "—"))
                show_plan(maintenance.get("selected_plan"))
                with st.expander("Supporting agent responses"):
                    st.json(maintenance)
            else:
                st.info("No maintenance coordination was returned for this assessment.")
            with st.expander("Complete assessment response"):
                st.json(data)
        except (KeyError, TypeError, ValueError) as error:
            show_error(error, "Unexpected assessment response")

elif page == "Anomaly Simulation":
    st.header("Synthetic Anomaly Simulation")
    st.write("Injecting an artificial oil-temperature deviation into a historical LOADED-state sample to demonstrate agent coordination.")
    st.warning("Synthetic test only: the injected reading is not a real compressor failure or a validated prediction.")
    if st.button("Simulate Sensor Anomaly", type="primary"):
        try:
            with st.spinner("Running synthetic anomaly simulation..."):
                st.session_state.simulation_result = fetch_api("/predictagent/simulate-anomaly")
        except requests.exceptions.RequestException as error:
            show_error(error, "Simulation failed")
    if st.session_state.simulation_result is not None:
        data = st.session_state.simulation_result
        try:
            assessment = data["assessment"]
            show_diagnostics(assessment["diagnostics"])
            st.subheader("Orchestrator decision")
            st.write(assessment.get("decision", "—"))
            st.metric("Human review required", "Yes" if assessment.get("requires_human_review") else "No")
            maintenance = assessment.get("maintenance")
            if maintenance:
                st.subheader("Provisional maintenance recommendation")
                st.write(maintenance.get("decision", "—"))
                show_plan(maintenance.get("selected_plan"))
                with st.expander("Supporting agent responses"):
                    st.json(maintenance)
            with st.expander("Complete simulation response"):
                st.json(data)
        except (KeyError, TypeError, ValueError) as error:
            show_error(error, "Unexpected simulation response")

elif page == "Historical Sensor Monitoring":
    st.header("Historical Sensor Monitoring")
    st.caption("Historical MetroPT measurements from 2020; not a live sensor feed.")
    sensor_limit = st.slider("Number of historical readings", min_value=50, max_value=1000, value=100, step=50)
    if st.button("Load Sensor History", type="primary"):
        try:
            with st.spinner("Loading historical readings..."):
                st.session_state.sensor_history = fetch_api("/sensors/history", params={"limit": sensor_limit})
        except requests.exceptions.RequestException as error:
            show_error(error, "Sensor history request failed")
    if st.session_state.sensor_history is not None:
        try:
            result = st.session_state.sensor_history
            sensor_df = pd.DataFrame(result["readings"])
            if sensor_df.empty:
                st.warning("No sensor readings returned.")
            else:
                sensor_df["timestamp"] = pd.to_datetime(sensor_df["timestamp"], errors="coerce")
                sensor_df = sensor_df.dropna(subset=["timestamp"]).sort_values("timestamp")
                available = [sensor for sensor in result.get("sensors", []) if sensor in sensor_df.columns]
                st.success(f"Loaded {len(sensor_df)} historical readings.")
                selected = st.multiselect("Select sensors", options=available, default=[s for s in ["Oil_temperature", "Motor_current"] if s in available], key="historical_selected_sensors")
                if not selected:
                    st.info("Select one or more sensors to view their charts.")
                for sensor in selected:
                    chart = sensor_df[["timestamp", sensor]].copy()
                    chart[sensor] = pd.to_numeric(chart[sensor], errors="coerce")
                    chart = chart.dropna(subset=[sensor])
                    st.subheader(sensor.replace("_", " "))
                    if chart.empty:
                        st.warning(f"No valid numeric readings for {sensor}.")
                        continue
                    st.line_chart(chart, x="timestamp", y=sensor, height=310, use_container_width=True)
                    cols = st.columns(3)
                    cols[0].metric("Minimum", f"{chart[sensor].min():.3f}")
                    cols[1].metric("Maximum", f"{chart[sensor].max():.3f}")
                    cols[2].metric("Average", f"{chart[sensor].mean():.3f}")
                with st.expander("Raw historical sensor data"):
                    st.dataframe(sensor_df, use_container_width=True)
        except (KeyError, ValueError, TypeError) as error:
            show_error(error, "Unable to process sensor readings")

elif page == "Anomaly Visualization":
    st.header("Synthetic Anomaly Visualization")
    st.write("Comparing historical oil-temperature readings with an illustrative synthetic anomaly and approximate state-specific ±3σ thresholds.")
    st.caption("The historical chart sample is not necessarily from the simulation's operating state or time window.")
    if st.button("Generate Anomaly Visualization", type="primary"):
        try:
            with st.spinner("Retrieving simulation and historical readings..."):
                simulation = fetch_api("/predictagent/simulate-anomaly")
                history = fetch_api("/sensors/history", params={"limit": 100})
            diagnostics = simulation["assessment"]["diagnostics"]
            oil_anomaly = (diagnostics.get("anomalies") or {}).get("Oil_temperature")
            if not oil_anomaly:
                st.warning("The simulation did not return an oil-temperature anomaly.")
                st.session_state.anomaly_chart_data = None
            else:
                value = float(oil_anomaly["current_value"])
                mean = float(oil_anomaly["reference_mean"])
                z = float(oil_anomaly["z_score"])
                if abs(z) < 1e-9:
                    raise ValueError("The rounded Z-score is zero; cannot estimate reference variation.")
                std = abs((value - mean) / z)
                history_df = pd.DataFrame(history["readings"])
                history_df["timestamp"] = pd.to_datetime(history_df["timestamp"], errors="coerce")
                history_df["Oil_temperature"] = pd.to_numeric(history_df["Oil_temperature"], errors="coerce")
                history_df = history_df.dropna(subset=["timestamp", "Oil_temperature"]).sort_values("timestamp")
                if history_df.empty:
                    raise ValueError("No valid historical oil-temperature readings were returned.")
                chart = history_df[["timestamp", "Oil_temperature"]].rename(columns={"Oil_temperature": "Historical Temperature"}).copy()
                chart["Synthetic Anomaly"] = float("nan")
                synthetic_time = chart["timestamp"].iloc[-1] + pd.Timedelta(seconds=10)
                chart = pd.concat([chart, pd.DataFrame([{"timestamp": synthetic_time, "Historical Temperature": float("nan"), "Synthetic Anomaly": value}])], ignore_index=True)
                chart["Reference Mean"] = mean
                chart["Upper Threshold (+3σ)"] = mean + 3 * std
                chart["Lower Threshold (-3σ)"] = mean - 3 * std
                st.session_state.anomaly_chart_data = {"chart": chart, "value": value, "mean": mean, "z": z, "upper": mean + 3 * std, "lower": mean - 3 * std, "state": diagnostics.get("operating_state"), "detected": diagnostics.get("anomaly_detected")}
        except (requests.exceptions.RequestException, KeyError, ValueError, TypeError) as error:
            show_error(error, "Anomaly visualization failed")
    if st.session_state.anomaly_chart_data is not None:
        v = st.session_state.anomaly_chart_data
        cols = st.columns(3)
        cols[0].metric("Synthetic oil temperature", f"{v['value']:.2f}")
        cols[1].metric("Reference mean", f"{v['mean']:.2f}")
        cols[2].metric("Z-score", f"{v['z']:.2f}")
        cols = st.columns(2)
        cols[0].metric("Approx. upper threshold", f"{v['upper']:.2f}")
        cols[1].metric("Approx. lower threshold", f"{v['lower']:.2f}")
        st.write("**Simulation operating state:**", v["state"])
        st.line_chart(v["chart"], x="timestamp", y=["Historical Temperature", "Synthetic Anomaly", "Reference Mean", "Upper Threshold (+3σ)", "Lower Threshold (-3σ)"], height=450, use_container_width=True)
        if v["detected"]:
            st.warning("Synthetic anomaly flagged for human review.")
        else:
            st.info("The simulation did not flag an anomaly.")
        st.caption("The synthetic point is placed after an unrelated historical sample for illustration. Thresholds are approximated from rounded API Z-scores and are specific to the simulation's operating state; they are not necessarily applicable to every historical point plotted.")
        with st.expander("Visualization data"):
            st.dataframe(v["chart"], use_container_width=True)

# Displaying a consistent footer
st.divider()
st.caption("PredictAgent • Historical MetroPT demonstration • Exploratory anomaly screening • Simulated maintenance resources • Human approval required")

# ---------------------------------------------------------
# SECTION 8: MULTI-AGENT DECISION VISUALIZATION
# ---------------------------------------------------------

st.divider()
st.header("8. Multi-Agent Decision Visualization")

st.write(
    "Visualize how PredictAgent coordinates its agents "
    "to assess equipment conditions and prepare a "
    "provisional maintenance recommendation."
)

st.info(
    "This demonstration uses a synthetic sensor anomaly "
    "and simulated maintenance resources. "
    "No actual maintenance is booked or approved."
)


# Initializing session state for agent decisions
if "agent_decision_results" not in st.session_state:
    st.session_state.agent_decision_results = None


# Creating the multi-agent execution button
if st.button("Run Multi-Agent Workflow", key="run_agent_workflow"):

    try:
        # Requesting the synthetic anomaly workflow
        with st.spinner("Coordinating PredictAgent agents..."):
            workflow_result = fetch_api(
                "/predictagent/simulate-anomaly"
            )

        # Storing results across Streamlit reruns
        st.session_state.agent_decision_results = workflow_result

        st.success("Multi-agent workflow completed.")

    except requests.exceptions.RequestException as error:
        st.error(f"Workflow request failed: {error}")


# Displaying saved agent decisions
if st.session_state.agent_decision_results is not None:

    # Extracting the complete workflow response
    workflow = st.session_state.agent_decision_results

    # Extracting the assessment results
    assessment = workflow.get("assessment", {})

    # Extracting diagnostics
    diagnostics = assessment.get("diagnostics", {})

    # Extracting maintenance results
    maintenance = assessment.get("maintenance") or {}

    # Extracting the selected maintenance plan
    selected_plan = maintenance.get("selected_plan") or {}

    # Displaying the workflow heading
    st.subheader("Multi-Agent Workflow")

    # Displaying the Diagnostics Agent
    with st.container(border=True):

        st.markdown("### 1. Diagnostics Agent")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric(
                "Operating State",
                diagnostics.get("operating_state", "Unknown")
            )

        with col2:
            st.metric(
                "Anomalies Detected",
                diagnostics.get("anomaly_count", 0)
            )

        with col3:
            st.metric(
                "Anomaly Status",
                "Detected"
                if diagnostics.get("anomaly_detected")
                else "Not Detected"
            )

        # Displaying sensor anomalies
        with st.expander("View Sensor Diagnostics"):
            st.json(diagnostics)

    # Displaying the workflow connection
    st.markdown(
        "<h2 style='text-align:center;'>↓</h2>",
        unsafe_allow_html=True
    )

    # Displaying the Maintenance Planner
    with st.container(border=True):

        st.markdown("### 2. Maintenance Planner Agent")

        st.write(
            "**Planning Decision:**",
            maintenance.get("decision", "No maintenance proposal generated")
        )

        if selected_plan:

            col1, col2 = st.columns(2)

            with col1:
                st.write(
                    "**Equipment:**",
                    selected_plan.get("equipment_id", "N/A")
                )

                st.write(
                    "**Start Time:**",
                    selected_plan.get("start_time", "N/A")
                )

                st.write(
                    "**End Time:**",
                    selected_plan.get("end_time", "N/A")
                )

            with col2:
                st.write(
                    "**Spare Part:**",
                    selected_plan.get("part_id", "N/A")
                )

                st.write(
                    "**Plan Status:**",
                    selected_plan.get("status", "N/A")
                )

                st.write(
                    "**Approval Status:**",
                    selected_plan.get("approval_status", "N/A")
                )

        else:
            st.warning(
                "No complete maintenance plan was selected."
            )

    # Displaying supporting agents
    st.markdown(
        "<h2 style='text-align:center;'>↓</h2>",
        unsafe_allow_html=True
    )

    st.subheader("Supporting Agents")

    col1, col2 = st.columns(2)

    # Displaying Parts Availability Agent
    with col1:

        with st.container(border=True):

            st.markdown("### 3. Parts Availability Agent")

            parts = maintenance.get("parts")

            if parts is not None:
                st.json(parts)
            else:
                st.info("No parts availability response.")

    # Displaying Technician Availability Agent
    with col2:

        with st.container(border=True):

            st.markdown("### 4. Technician Availability Agent")

            technician = selected_plan.get("technician")

            if technician:
                st.json(technician)
            else:
                st.info(
                    "No selected technician in the maintenance plan."
                )

    col3, col4 = st.columns(2)

    # Displaying Weather Agent
    with col3:

        with st.container(border=True):

            st.markdown("### 5. Weather Agent")

            weather = maintenance.get("weather")

            if weather is not None:
                st.json(weather)
            else:
                st.info("No weather response.")

    # Displaying Alert Agent
    with col4:

        with st.container(border=True):

            st.markdown("### 6. Alert Agent")

            alert = maintenance.get("alert")

            if alert is not None:
                st.json(alert)
            else:
                st.info("No alert response.")

    # Displaying final workflow connection
    st.markdown(
        "<h2 style='text-align:center;'>↓</h2>",
        unsafe_allow_html=True
    )

    # Displaying the final decision
    with st.container(border=True):

        st.markdown("### Final Maintenance Recommendation")

        st.write(
            "**Assessment Decision:**",
            assessment.get("decision", "Not available")
        )

        st.write(
            "**Human Review Required:**",
            "Yes"
            if assessment.get("requires_human_review")
            else "No"
        )

        if selected_plan:

            st.warning(
                "A provisional maintenance plan has been proposed. "
                "Human approval is required before execution."
            )

        elif diagnostics.get("anomaly_detected"):

            st.error(
                "An anomaly was detected, but no complete "
                "maintenance plan was selected. Manual review is required."
            )

        else:

            st.info(
                "No anomaly-triggered maintenance proposal "
                "was generated."
            )

    # Displaying the full workflow response
    with st.expander("View Complete Multi-Agent Workflow JSON"):
        st.json(workflow)

    st.caption(
        "Agent outputs are displayed from the backend workflow. "
        "Maintenance availability and operational resources are simulated. "
        "This system does not autonomously approve or book maintenance."
    )
