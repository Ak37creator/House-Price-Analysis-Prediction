
import streamlit as st
import pandas as pd
import numpy as np
import joblib

from pathlib import Path

# --------------------------------
# 1. Page Configuration
# --------------------------------

st.set_page_config(
    page_title="House Price Predictor",
    page_icon="🏠",
    layout="wide"
)

st.title("🏠 House Price Prediction")
st.write(
    "Enter the property details below to estimate "
    "the house sale price."
)

# --------------------------------
# 2. Load Model and Training Data
# --------------------------------

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "models" / "final_house_price_model.pkl"

DATA_PATH = (
    BASE_DIR / "data" / "processed"
    / "train_feature_engineered.csv"
)

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

@st.cache_data
def load_training_data():
    return pd.read_csv(DATA_PATH)

try:
    model = load_model()
    reference_data = load_training_data()

except FileNotFoundError as error:
    st.error(f"Required file not found: {error}")
    st.stop()

# --------------------------------
# 3. Prepare Feature Defaults
# --------------------------------

X_reference = reference_data.drop(
    columns=["SalePrice", "Id"],
    errors="ignore"
)

# Numeric columns: median
numeric_defaults = X_reference.select_dtypes(
    include=np.number
).median()

# Categorical columns: most frequent value
categorical_defaults = {}

for column in X_reference.select_dtypes(
    include=["object", "str", "category"]
).columns:

    modes = X_reference[column].mode()

    if not modes.empty:
        categorical_defaults[column] = modes.iloc[0]
    else:
        categorical_defaults[column] = "None"

# Create one default property row
default_values = {}

for column in X_reference.columns:

    if column in numeric_defaults.index:
        default_values[column] = numeric_defaults[column]

    else:
        default_values[column] = categorical_defaults[column]

# --------------------------------
# 4. User Input Form
# --------------------------------

st.subheader("Property Details")

col1, col2 = st.columns(2)

with col1:

    neighborhood_options = sorted(
        X_reference["Neighborhood"].dropna().unique().tolist()
    )

    neighborhood = st.selectbox(
        "Neighborhood",
        neighborhood_options
    )

    overall_quality = st.slider(
        "Overall Quality",
        min_value=1,
        max_value=10,
        value=7
    )

    living_area = st.number_input(
        "Ground Living Area (sq ft)",
        min_value=100,
        max_value=10000,
        value=1500,
        step=50
    )

    lot_area = st.number_input(
        "Lot Area (sq ft)",
        min_value=100,
        max_value=100000,
        value=8000,
        step=100
    )

    year_built = st.number_input(
        "Year Built",
        min_value=1800,
        max_value=2026,
        value=2000,
        step=1
    )

with col2:

    bedrooms = st.number_input(
        "Bedrooms Above Ground",
        min_value=0,
        max_value=15,
        value=3
    )

    garage_cars = st.number_input(
        "Garage Capacity (Cars)",
        min_value=0,
        max_value=5,
        value=2
    )

    total_basement = st.number_input(
        "Total Basement Area (sq ft)",
        min_value=0,
        max_value=10000,
        value=800,
        step=50
    )

    first_floor = st.number_input(
        "First Floor Area (sq ft)",
        min_value=100,
        max_value=10000,
        value=1000,
        step=50
    )

    second_floor = st.number_input(
        "Second Floor Area (sq ft)",
        min_value=0,
        max_value=10000,
        value=500,
        step=50
    )

    year_sold = st.number_input(
        "Sale Year",
        min_value=2006,
        max_value=2010,
        value=2010,
        step=1
    )

# Additional inputs
st.subheader("Additional Property Details")

col3, col4 = st.columns(2)

with col3:
    year_remodeled = st.number_input(
        "Year Remodeled",
        min_value=1800,
        max_value=2026,
        value=2005,
        step=1
    )

    full_bath = st.number_input(
        "Full Bathrooms Above Ground",
        min_value=0,
        max_value=10,
        value=2
    )

    half_bath = st.number_input(
        "Half Bathrooms Above Ground",
        min_value=0,
        max_value=10,
        value=1
    )

with col4:
    basement_full_bath = st.number_input(
        "Basement Full Bathrooms",
        min_value=0,
        max_value=5,
        value=0
    )

    basement_half_bath = st.number_input(
        "Basement Half Bathrooms",
        min_value=0,
        max_value=5,
        value=0
    )

# --------------------------------
# 5. Prediction
# --------------------------------

if st.button("Predict House Price", type="primary"):

    # Start with training-data defaults
    input_data = default_values.copy()

    # Update user-provided original features
    user_inputs = {
        "Neighborhood": neighborhood,
        "OverallQual": overall_quality,
        "GrLivArea": living_area,
        "LotArea": lot_area,
        "YearBuilt": year_built,
        "BedroomAbvGr": bedrooms,
        "GarageCars": garage_cars,
        "TotalBsmtSF": total_basement,
        "1stFlrSF": first_floor,
        "2ndFlrSF": second_floor,
        "YrSold": year_sold,
        "YearRemodAdd": year_remodeled,
        "FullBath": full_bath,
        "HalfBath": half_bath,
        "BsmtFullBath": basement_full_bath,
        "BsmtHalfBath": basement_half_bath
    }

    # Check whether all selected columns exist
    missing_columns = [
        column for column in user_inputs
        if column not in input_data
    ]

    if missing_columns:
        st.error(
            "These input columns are missing from the model data: "
            + ", ".join(missing_columns)
        )
        st.stop()

    input_data.update(user_inputs)

    # Calculate engineered features
    input_data["HouseAge"] = year_sold - year_built

    input_data["RemodAge"] = year_sold - year_remodeled

    input_data["TotalArea"] = (
        total_basement + first_floor + second_floor
    )

    input_data["TotalBathrooms"] = (
        full_bath
        + 0.5 * half_bath
        + basement_full_bath
        + 0.5 * basement_half_bath
    )

    # Calculate porch area from default values
    porch_columns = [
        "OpenPorchSF",
        "EnclosedPorch",
        "3SsnPorch",
        "ScreenPorch",
        "WoodDeckSF"
    ]

    input_data["TotalPorchArea"] = sum(
        input_data.get(column, 0)
        for column in porch_columns
    )

    if overall_quality <= 4:
        input_data["QualityCategory"] = "Low"
    elif overall_quality <= 7:
        input_data["QualityCategory"] = "Medium"
    else:
        input_data["QualityCategory"] = "High"

    # Match the exact training feature order
    input_df = pd.DataFrame(
        [input_data],
        columns=X_reference.columns
    )

    try:
        predicted_price = model.predict(input_df)[0]

        st.success("Prediction Completed!")

        st.subheader("Estimated House Price")

        st.metric(
            label="Predicted Sale Price",
            value=f"${predicted_price:,.2f}"
        )

        st.caption(
            "This is an estimated price generated by a "
            "machine learning model, not a guaranteed market value."
        )

        with st.expander("View Model Input Details"):
            st.dataframe(input_df)

    except Exception as error:
        st.error(f"Prediction failed: {error}")


