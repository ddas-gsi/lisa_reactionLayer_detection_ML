# Streamlit dashboard to visualize the batch mode predictions of the model in real-time with websocket

import streamlit as st
import asyncio
import pandas as pd
import plotly.express as px
import json
import websockets

st.set_page_config(page_title="Live Layer Prediction", layout="wide")
st.title("🧪 Real-Time Reaction Layer Monitor")

# Streamlit session state
if "paused" not in st.session_state:
    st.session_state.paused = False
if "latest_df" not in st.session_state:
    st.session_state.latest_df = pd.DataFrame()

# Controls
col1, col2 = st.columns([1, 3])
with col1:
    if st.button("⏸️ Pause" if not st.session_state.paused else "▶️ Resume"):
        st.session_state.paused = not st.session_state.paused

with col2:
    if st.button("🔁 Reset Histogram"):
        st.session_state.latest_df = pd.DataFrame()
        st.success("Histograms reset!")

# Live plot area
placeholder = st.empty()

async def listen_to_websocket():
    uri = "ws://localhost:8000/ws/predict"
    async with websockets.connect(uri) as websocket:
        while True:
            if not st.session_state.paused:
                try:
                    message = await websocket.recv()
                    data = json.loads(message)
                    df = pd.DataFrame(data)
                    df["Layer"] = df["prediction"].map({0: "Layer 1", 1: "Layer 2", 999: "No Reaction"})

                    # Accumulate if you want historic data
                    st.session_state.latest_df = pd.concat([st.session_state.latest_df, df], ignore_index=True)

                    with placeholder.container():
                        st.subheader("🔍 Real-Time Histogram")
                        fig = px.histogram(df, x="Layer", color="Layer", title="Live Predicted Layers", nbins=3)
                        st.plotly_chart(fig, use_container_width=True)

                        col1, col2 = st.columns(2)
                        fig_l0 = px.histogram(df, x="x1", color="Layer", title="dE_L0 Distribution", nbins=30)
                        fig_l1 = px.histogram(df, x="x2", color="Layer", title="dE_L1 Distribution", nbins=30)
                        with col1:
                            st.plotly_chart(fig_l0, use_container_width=True)
                        with col2:
                            st.plotly_chart(fig_l1, use_container_width=True)

                        fig_tot = px.histogram(df, x="x3", color="Layer", title="dE_Tot Distribution", nbins=30)
                        st.plotly_chart(fig_tot, use_container_width=True)
                except Exception as e:
                    st.error(f"WebSocket Error: {e}")
            await asyncio.sleep(1)

# Async Streamlit wrapper
async def main():
    await listen_to_websocket()

# Run the async main loop
asyncio.run(main())
