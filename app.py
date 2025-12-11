# app.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import io

st.set_page_config(page_title="House Price Predictor", layout="centered")

st.title("🏡 House Price Predictor")
st.markdown(
    """
Simple demo app that loads the saved pipeline `models/house_price_model.joblib`
and makes price predictions. Upload a CSV with the same features used for training,
or use the manual input form for a single prediction.
"""
)

@st.cache_resource
def load_model(path="models/house_price_model.joblib"):
    model = joblib.load(path)
    return model

# Load model
try:
    model = load_model()
except FileNotFoundError:
    st.error("Saved model not found. Please ensure `models/house_price_model.joblib` exists in the project.")
    st.stop()

# Try to extract expected column names from the fitted ColumnTransformer
def get_expected_columns(pipe):
    try:
        preproc = pipe.named_steps["preprocess"]
        expected = []
        # For fitted ColumnTransformer, `transformers_` contains (name, transformer, column_list)
        for name, transformer, cols in preproc.transformers_:
            # skip remainder
            if name == "remainder" or cols == "remainder":
                continue
            # If cols is a list/array of names, extend
            if isinstance(cols, (list, tuple, np.ndarray, pd.Index)):
                expected.extend(list(cols))
            else:
                # could be slice or function; skip gracefully
                pass
        # de-duplicate preserving order
        seen = set()
        expected_unique = [x for x in expected if not (x in seen or seen.add(x))]
        return expected_unique
    except Exception:
        return None

expected_cols = get_expected_columns(model)
if expected_cols is None:
    st.warning("Could not automatically determine expected feature columns from the pipeline. CSV upload mode will still work if you provide correct columns.")
else:
    st.info(f"Model expects ~{len(expected_cols)} input columns (auto-detected).")

st.header("Choose input mode")
mode = st.radio("Input mode", ["Upload CSV file", "Manual single-row input"])

def predict_dataframe(df_input):
    """Given a DataFrame with feature columns, run through pipeline and return original-scale predictions."""
    # Ensure all expected columns exist (if we detected expected_cols)
    if expected_cols is not None:
        for c in expected_cols:
            if c not in df_input.columns:
                # fill numeric-like with 0, categorical with "None"
                df_input[c] = 0
    # Keep columns in a stable order: use expected_cols if available else df columns
    try:
        preds_log = model.predict(df_input)
    except Exception as e:
        st.error(f"Prediction failed: {e}")
        return None
    # If model trained on log target, convert back
    preds = np.expm1(preds_log)
    return preds

if mode == "Upload CSV file":
    st.subheader("Upload CSV")
    st.markdown("Upload a CSV with the same features used during training (exclude the target column `SalePrice`). One row per house.")
    uploaded = st.file_uploader("Choose CSV file", type=["csv"])
    if uploaded is not None:
        try:
            df = pd.read_csv(uploaded)
            st.write("Preview of your uploaded data:")
            st.dataframe(df.head())
            if st.button("Run predictions on uploaded CSV"):
                with st.spinner("Predicting..."):
                    preds = predict_dataframe(df.copy())
                    if preds is not None:
                        df_out = df.copy().reset_index(drop=True)
                        df_out["PredictedPrice"] = preds
                        st.success("Predictions complete.")
                        st.dataframe(df_out.head())
                        # allow downloading results
                        csv = df_out.to_csv(index=False).encode("utf-8")
                        st.download_button(
                            label="Download predictions as CSV",
                            data=csv,
                            file_name="predictions.csv",
                            mime="text/csv"
                        )
        except Exception as e:
            st.error(f"Could not read uploaded CSV: {e}")

else:
    st.subheader("Manual Input — single house")
    st.markdown("Fill the fields below for a single prediction. Only shows commonly important features if they exist in model inputs.")
    # define a list of commonly important features; we'll show those if they exist in expected_cols
    common = [
        "OverallQual","GrLivArea","GarageCars","TotalBsmtSF",
        "YearBuilt","YearRemodAdd","FullBath","HalfBath","BsmtFullBath","BsmtHalfBath",
        "1stFlrSF","2ndFlrSF","YrSold","LotArea","Neighborhood"
    ]
    input_data = {}

    # If expected columns known, show intersection of common and expected
    if expected_cols is not None:
        to_show = [c for c in common if c in expected_cols]
        other_numeric = [c for c in expected_cols if c not in to_show and c in df.select_dtypes(include=[np.number]).columns] if 'df' in globals() else []
    else:
        # fallback: show common set
        to_show = common

    # Render widgets for to_show
    for col in to_show:
        # Choose widget type based on name heuristics
        if col in ["Neighborhood"]:
            val = st.selectbox(f"{col}", options=["NA","CollgCr","Veenker","Crawfor","NoRidge","Mitchel","Somerst","NWAmes","OldTown","BrkSide","Sawyer","NridgHt","NAmes","SawyerW","IDOTRR","MeadowV","Edwards","Timber","Gilbert","StoneBr","ClearCr","Blmngtn","BrDale","SWISU","Blueste","NPkVill"])
            input_data[col] = val
        elif any(k in col.lower() for k in ["year","yr","age","area","sf","lot","bsmt","grliv","garage","fullbath","halfbath"]):
            # numeric
            default = 2000 if "Year" in col or "Yr" in col else 1000
            v = st.number_input(f"{col}", value=int(default))
            input_data[col] = v
        else:
            v = st.text_input(f"{col}", value="")
            input_data[col] = v

    if st.button("Predict single house"):
        # build a single-row DataFrame
        row = pd.DataFrame([input_data])
        # ensure all expected columns exist
        if expected_cols is not None:
            for c in expected_cols:
                if c not in row.columns:
                    # default numeric 0, default categorical "None"
                    row[c] = 0
            # reorder
            row = row[expected_cols]
        with st.spinner("Running model..."):
            preds = predict_dataframe(row)
            if preds is not None:
                price = float(preds[0])
                st.success(f"Predicted house price: ${price:,.0f}")
                st.write("Input used for prediction:")
                st.table(row.T)

st.markdown("---")
st.markdown("Tips:\n- If you have the original training CSV, use Upload mode and remove `SalePrice` before uploading.\n- For multi-row predictions, upload a CSV with each sample as a row.")
