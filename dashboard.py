import streamlit as st
import requests

st.title("Reaction Layer Prediction")

x1 = st.number_input("dE_L0", value=0.0)
x2 = st.number_input("dE_L1", value=0.0)
# x3 = st.number_input("Input x3", value=0.0)
x3 = x1 + x2  # derived feature
st.write(f"dE_Total: {x3}")


if st.button("Get Reaction Layer"):
    st.write("Fetching prediction from the server...")
    try:
        response = requests.post("http://localhost:8000/predict", json={    # Replace with the actual URL of the server
            "x1": x1,
            "x2": x2,
            "x3": x3
        })
        response.raise_for_status()  # raises an error for non-200 codes
        result = response.json()
        # st.success(f"Prediction: {result['prediction']}")
        if result['prediction'] == 0:
            st.success("Reaction Layer >> 1")
        elif result['prediction'] == 1:
            st.success("Reaction Layer >> 2")
        elif result['prediction'] == 999:
            st.success("No Reaction")

    except requests.exceptions.RequestException as e:
        st.error(f"Request failed: {e}")
        st.text(f"Raw response: {response.text}")
    except ValueError as e:
        st.error("Could not decode JSON response")
        st.text(f"Raw response: {response.text}")


