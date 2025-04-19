import streamlit as st
import pandas as pd
import plotly.express as px
import json
import time
import os
import json.decoder

st.set_page_config(page_title="Live Reaction Layer Monitor", layout="wide")
st.title("Live Reaction Layer Monitor")

# Pause/Continue toggle
paused = st.checkbox("Pause live updates", value=False)

# Reset button
if st.button("Reset Histogram"):
    if os.path.exists("latest_batch.json"):
        os.remove("latest_batch.json")
    if os.path.exists("cumulative_data.csv"):
        os.remove("cumulative_data.csv")
    st.success("Histograms and cumulative stats reset!")
    st.stop()

# Placeholder for plot updates
placeholder = st.empty()

# Counter for unique keys
update_counter = 0

# Load existing cumulative data (if any)
if os.path.exists("cumulative_data.csv"):
    all_data = pd.read_csv("cumulative_data.csv")
else:
    all_data = pd.DataFrame()

# Main loop
while True:
    if not paused:
        try:
            with open("latest_batch.json", "r") as f:
                data = json.load(f)

            df = pd.DataFrame(data)
            df["Layer"] = df["prediction"].map({0: "Layer 1", 1: "Layer 2", 999: "No Reaction"})

            # Append to cumulative data
            all_data = pd.concat([all_data, df], ignore_index=True)
            all_data.to_csv("cumulative_data.csv", index=False)

            # Cumulative layer counts
            all_data["Layer"] = all_data["prediction"].map({0: "Layer 1", 1: "Layer 2", 999: "No Reaction"})
            layer_counts = all_data["Layer"].value_counts().reset_index()
            layer_counts.columns = ["Layer", "Total Count"]

            with placeholder.container():

                # Fixed order & color
                layer_order = ["Layer 1", "Layer 2", "No Reaction"]
                layer_color_map = {
                    "Layer 1": "blue",
                    "Layer 2": "green",
                    "No Reaction": "red"
                }
                st.subheader("Live Histogram of Predicted Layers")
                fig = px.histogram(
                    df, x="Layer", color="Layer", nbins=3,
                    category_orders={"Layer": layer_order},
                    # color_discrete_map=layer_color_map,
                    title="Layer Distribution (Updates every second)",
                    labels={"Layer": "Reaction Layer"}
                    )
                st.plotly_chart(fig, use_container_width=True, key=f"main_hist_{update_counter}")

                st.subheader("📊 Live Energy Distributions (Updates every second)")
                col1, col2 = st.columns(2)
                with col1:
                    fig_l0 = px.histogram(df, x="x1", color="Layer", title="dE_L0 Distribution", nbins=100)
                    fig_l0.update_layout(xaxis_range=[750, 1200])
                    st.plotly_chart(fig_l0, use_container_width=True, key=f"hist_l0_{update_counter}")

                with col2:
                    fig_l1 = px.histogram(df, x="x2", color="Layer", title="dE_L1 Distribution", nbins=100)
                    fig_l1.update_layout(xaxis_range=[750, 1200])
                    st.plotly_chart(fig_l1, use_container_width=True, key=f"hist_l1_{update_counter}")

                fig_tot = px.histogram(df, x="x3", color="Layer", title="dE_Tot Distribution", nbins=100, range_x=[1500, 2500])
                st.plotly_chart(fig_tot, use_container_width=True, key=f"hist_tot_{update_counter}")

                st.subheader("🔄 Cumulative Prediction Count")
                st.dataframe(layer_counts)
                fig_cumulative_bar = px.bar(layer_counts, x="Layer", y="Total Count", color="Layer", title="Cumulative Prediction Count")
                st.plotly_chart(fig_cumulative_bar, use_container_width=True, key=f"bar_cumulative_{update_counter}")

                st.subheader("📊 Cumulative Energy Distributions")
                col3, col4 = st.columns(2)
                with col3:
                    fig_cum_l0 = px.histogram(all_data, x="x1", color="Layer", title="Cumulative dE_L0 Distribution", nbins=500, range_x=[750, 1200])
                    st.plotly_chart(fig_cum_l0, use_container_width=True, key=f"cum_l0_{update_counter}")

                with col4:
                    fig_cum_l1 = px.histogram(all_data, x="x2", color="Layer", title="Cumulative dE_L1 Distribution", nbins=500, range_x=[750, 1200])
                    st.plotly_chart(fig_cum_l1, use_container_width=True, key=f"cum_l1_{update_counter}")

                fig_cum_tot = px.histogram(all_data, x="x3", color="Layer", title="Cumulative dE_Tot Distribution", nbins=500, range_x=[1500, 2500])
                st.plotly_chart(fig_cum_tot, use_container_width=True, key=f"cum_tot_{update_counter}")

            update_counter += 1

        except (FileNotFoundError, json.decoder.JSONDecodeError):
            st.warning("Waiting for valid data from randomEnergy generator...")

    else:
        st.info("Live updates are paused.")

    time.sleep(1)
