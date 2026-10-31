
import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
import joblib
import plotly.graph_objects as go
from pathlib import Path

# --------------------------------
# PAGE CONFIGURATION
# --------------------------------

st.set_page_config(
    page_title="EV BatteryGuard",
    page_icon="🔋",
    layout="wide"
)

BASE_DIR = Path(__file__).parent

# --------------------------------
# LOAD MODELS AND SCALERS
# --------------------------------

@st.cache_resource
def load_artifacts():

    soh_model = tf.keras.models.load_model(
        BASE_DIR / "models" / "EV_BatteryGuard_SOH.keras"
    )

    rul_model = tf.keras.models.load_model(
        BASE_DIR / "models" / "EV_BatteryGuard_RUL.keras"
    )

    soh_scaler = joblib.load(
        BASE_DIR / "scalers" / "SOH_Scaler.pkl"
    )

    rul_scaler = joblib.load(
        BASE_DIR / "scalers" / "RUL_Scaler.pkl"
    )

    rul_target_scaler = joblib.load(
        BASE_DIR / "scalers" / "RUL_Target_Scaler.pkl"
    )

    return (
        soh_model,
        rul_model,
        soh_scaler,
        rul_scaler,
        rul_target_scaler
    )


# --------------------------------
# APPLICATION HEADER
# --------------------------------

st.title("🔋 EV BatteryGuard")
st.subheader("AI-Powered EV Battery Health Monitoring System")

st.write(
    "Predict battery State of Health (SOH) "
    "and Remaining Useful Life (RUL) using Artificial Neural Networks."
)

st.divider()

# --------------------------------
# LOAD ARTIFACTS
# --------------------------------

try:
    (
        soh_model,
        rul_model,
        soh_scaler,
        rul_scaler,
        rul_target_scaler
    ) = load_artifacts()

    st.success("ANN models and scalers loaded successfully!")

except Exception as e:
    st.error(f"Error loading models: {e}")
    st.stop()


# --------------------------------
# SIDEBAR INPUTS
# --------------------------------

st.sidebar.header("Battery Operating Parameters")

cycle = st.sidebar.number_input(
    "Cycle Number", min_value=1, max_value=300, value=50
)

voltage_mean = st.sidebar.number_input(
    "Mean Voltage", value=3.50
)

voltage_min = st.sidebar.number_input(
    "Minimum Voltage", value=2.50
)

voltage_max = st.sidebar.number_input(
    "Maximum Voltage", value=4.20
)

voltage_std = st.sidebar.number_input(
    "Voltage Standard Deviation", value=0.20
)

current_mean = st.sidebar.number_input(
    "Mean Current", value=-1.50
)

current_std = st.sidebar.number_input(
    "Current Standard Deviation", value=0.10
)

temperature_mean = st.sidebar.number_input(
    "Mean Temperature (°C)", value=32.0
)

temperature_max = st.sidebar.number_input(
    "Maximum Temperature (°C)", value=35.0
)

temperature_std = st.sidebar.number_input(
    "Temperature Standard Deviation", value=1.50
)

discharge_time = st.sidebar.number_input(
    "Discharge Time (seconds)", value=3600.0
)


# --------------------------------
# FEATURE PREPARATION
# --------------------------------

features = pd.DataFrame([[
    cycle,
    voltage_mean,
    voltage_min,
    voltage_max,
    voltage_std,
    current_mean,
    current_std,
    temperature_mean,
    temperature_max,
    temperature_std,
    discharge_time
]], columns=[
    "Cycle",
    "Voltage_Mean",
    "Voltage_Min",
    "Voltage_Max",
    "Voltage_Std",
    "Current_Mean",
    "Current_Std",
    "Temperature_Mean",
    "Temperature_Max",
    "Temperature_Std",
    "Discharge_Time"
])


# --------------------------------
# PREDICTION
# --------------------------------

st.subheader("Battery Health Prediction")

if st.button("Predict Battery Health", type="primary"):

    soh_input = soh_scaler.transform(features)

    soh_prediction = soh_model.predict(
        soh_input, verbose=0
    )[0][0]

    rul_input = rul_scaler.transform(features)

    rul_scaled_prediction = rul_model.predict(
        rul_input, verbose=0
    )

    rul_prediction = rul_target_scaler.inverse_transform(
        rul_scaled_prediction
    )[0][0]

    soh_prediction = float(np.clip(soh_prediction, 0, 100))
    rul_prediction = max(0, float(rul_prediction))

    col1, col2 = st.columns(2)

    with col1:
        st.metric(
            "Predicted State of Health",
            f"{soh_prediction:.2f}%"
        )

    with col2:
        st.metric(
            "Estimated Remaining Useful Life",
            f"{rul_prediction:.1f} cycles"
        )

    # --------------------------------
    # SOH VISUALIZATION
    # --------------------------------

    st.subheader("Battery Health Status")

    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=soh_prediction,
            title={"text": "Battery SOH (%)"},
            gauge={
                "axis": {"range": [0, 100]},
                "threshold": {
                    "line": {"color": "red", "width": 4},
                    "value": 70
                }
            }
        )
    )

    st.plotly_chart(fig, width="stretch")

    if soh_prediction >= 80:
        st.success("Battery health estimate: Good")
    elif soh_prediction >= 70:
        st.warning("Battery health estimate: Moderate")
    else:
        st.error("Battery health estimate: Below the 70% threshold")

    st.caption(
        "Experimental model estimates based on NASA battery "
        "cycling data. Predictions are not a substitute for "
        "a validated battery management system."
    )
