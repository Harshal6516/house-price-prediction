import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline

from preprocess import feature_engineering, build_preprocessor

def rmse(y_true, y_pred):
    from sklearn.metrics import mean_squared_error
    return np.sqrt(mean_squared_error(y_true, y_pred))

def train():
    # 1. Load data
    df = pd.read_csv("data/house_prices.csv")

    # 2. Feature engineering
    df = feature_engineering(df)

    # 3. Define target
    y = np.log1p(df["SalePrice"])       # log transform
    X = df.drop(columns=["SalePrice"])

    # 4. Preprocessing pipeline
    preprocessor, num_feats, cat_feats = build_preprocessor(X)

    model = Pipeline(steps=[
        ("preprocess", preprocessor),
        ("model", RandomForestRegressor(
            n_estimators=300,
            random_state=42,
            n_jobs=-1
        ))
    ])

    # 5. Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # 6. Train model
    model.fit(X_train, y_train)

    # 7. Evaluate
    preds = model.predict(X_test)
    score = rmse(y_test, preds)
    print("Model RMSE:", score)

    # 8. Save model
    joblib.dump(model, "models/house_price_model.joblib")
    print("Model saved to models/house_price_model.joblib")

if __name__ == "__main__":
    train()
