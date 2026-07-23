import streamlit as st
import joblib
import pandas as pd

st.set_page_config(page_title="California Housing Price Predictor", page_icon="🏠")


@st.cache_resource
def load_artifacts():
    model = joblib.load("model.pkl")
    scaler = joblib.load("scaler.pkl")
    feature_names = joblib.load("feature_names.pkl")
    return model, scaler, feature_names


model, scaler, feature_names = load_artifacts()

st.title("🏠 California Housing Price Predictor")
st.write(
    "Predicts median house value using the sigmoid MLP (2 hidden layers x 3 nodes) "
    "trained in the notebook. Fill in the fields below."
)

# scaler.data_min_ / data_max_ are set automatically by MinMaxScaler.fit() and
# line up with feature_names, so we reuse them to build sensible inputs instead
# of hardcoding column names or ranges.
feature_min = getattr(scaler, "data_min_", None)
feature_max = getattr(scaler, "data_max_", None)

input_values = {}
cols = st.columns(2)
for i, feature in enumerate(feature_names):
    with cols[i % 2]:
        lo = float(feature_min[i]) if feature_min is not None else 0.0
        hi = float(feature_max[i]) if feature_max is not None else 1.0

        if lo == 0.0 and hi == 1.0:
            # A 0/1 range almost always means this came from pd.get_dummies
            # (one-hot encoding) -- a checkbox fits that better than a number box.
            checked = st.checkbox(feature)
            input_values[feature] = 1.0 if checked else 0.0
        else:
            default = round((lo + hi) / 2, 4)
            input_values[feature] = st.number_input(
                feature, value=default, help=f"Training data range: {lo:,.2f} to {hi:,.2f}"
            )

if st.button("Predict Price", type="primary"):
    input_df = pd.DataFrame([input_values], columns=feature_names)
    scaled_input = scaler.transform(input_df)
    prediction = model.predict(scaled_input)[0]
    st.success(f"Predicted Median House Value: **${prediction * 100000:,.2f}**")

st.caption(
    "Model: MLPRegressor (activation='logistic', hidden_layer_sizes=(3, 3)) "
    "trained on the California housing dataset."
)
