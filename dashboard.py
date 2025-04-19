import streamlit as st
import pandas as pd
import plotly.express as px
import json
import time
import os

st.set_page_config(page_title="Live Reaction Layer Monitor", layout="wide")
st.title("Live Reaction Layer Prediction")

# Pause/Continue toggle
paused = st.checkbox("Pause live updates", value=False)

# Reset button
if st.button("Reset Histogram"):
    if os.path.exists("latest_batch.json"):
        os.remove("latest_batch.json")
        st.success("Histograms reset!")
    else:
        st.warning("No histogram data to reset.")
    st.stop()

# Placeholder to update plots live
placeholder = st.empty()

# Counter for unique keys
update_counter = 0

# Main loop
while True:
    if not paused:
        try:
            with open("latest_batch.json", "r") as f:
                data = json.load(f)
                df = pd.DataFrame(data)
                df["Layer"] = df["prediction"].map({0: "Layer 1", 1: "Layer 2", 999: "No Reaction"})

                with placeholder.container():
                    st.subheader("Live Histogram of Predicted Layers")
                    fig = px.histogram(df, x="Layer", color="Layer", nbins=3, title="Layer Distribution (Updated every second)")
                    st.plotly_chart(fig, use_container_width=True, key=f"main_hist_{update_counter}")

                    col1, col2 = st.columns(2)

                    with col1:
                        fig_l0 = px.histogram(df, x="x1", color="Layer", title="dE_L0 Distribution", nbins=100)
                        # fig_l0 = px.histogram(df, x="x1", color="Layer", title="dE_L0 Distribution")
                        fig_l0.update_layout(xaxis_range=[500, 1500])
                        st.plotly_chart(fig_l0, use_container_width=True, key=f"hist_l0_{update_counter}")

                    with col2:
                        fig_l1 = px.histogram(df, x="x2", color="Layer", title="dE_L1 Distribution", nbins=100)
                        # fig_l1 = px.histogram(df, x="x2", color="Layer", title="dE_L1 Distribution")
                        fig_l1.update_layout(xaxis_range=[500, 1500])
                        st.plotly_chart(fig_l1, use_container_width=True, key=f"hist_l1_{update_counter}")

                    fig_tot = px.histogram(df, x="x3", color="Layer", title="dE_Tot Distribution", nbins=100, range_x=[500,2500])
                    # fig_tot = px.histogram(df, x="x3", color="Layer", title="dE_Tot Distribution")
                    st.plotly_chart(fig_tot, use_container_width=True, key=f"hist_tot_{update_counter}")

                update_counter += 1  # Increment key version

        except FileNotFoundError:
            st.warning("Waiting for data from randomEnergy generator...")

    else:
        st.info("Live updates are paused.")

    time.sleep(1)
