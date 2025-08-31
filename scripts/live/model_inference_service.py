import joblib
import pandas as pd
from pathlib import Path

# --- Configuration Paths ---
MODEL_PATH = Path("output/models/supervised_model.joblib")
# In a real system, you might have a feature mapping or a list of expected features
# to ensure the input to the model is consistent.

# --- Global Model Object ---
# The model will be loaded once when this script is imported/run
_model = None
_expected_features = None # To store feature order if needed

def load_model():
    """
    Loads the pre-trained supervised model and its expected features (if available).
    """
    global _model, _expected_features
    if not MODEL_PATH.exists():
        print(f"Error: Model file not found at {MODEL_PATH}. Please ensure 'train_supervised_models.py' has been run.")
        return
    
    print(f"Loading model from {MODEL_PATH}...")
    try:
        # Assume the model is saved directly or as a dict with 'model' and 'features'
        loaded_content = joblib.load(MODEL_PATH)
        if isinstance(loaded_content, dict) and 'model' in loaded_content:
            _model = loaded_content['model']
            _expected_features = loaded_content.get('features')
        else:
            _model = loaded_content
            # If features are not explicitly saved, we'd need to infer or hardcode them
            print("Warning: Model loaded without explicit feature list. Prediction might require manual feature ordering.")
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Error loading model: {e}")

def predict_wallet_risk(wallet_features: dict):
    """
    Makes a fraud prediction for a single wallet based on its updated features.
    Returns the predicted label and the XAI reason code.
    """
    if _model is None:
        print("Error: Model not loaded. Cannot make predictions.")
        return {'prediction': 'unknown', 'xai_reason_code': wallet_features.get('xai_reason_code', 'no_model')}

    # Convert wallet_features dict to DataFrame row suitable for the model
    # This assumes the model expects features in a specific order/format.
    # For a real system, _expected_features would ensure correct order and handle missing features.
    feature_vector = pd.DataFrame([wallet_features])

    # Filter to only features the model expects and ensure order
    if _expected_features:
        missing_features = [f for f in _expected_features if f not in feature_vector.columns]
        for mf in missing_features:
            feature_vector[mf] = 0 # Fill missing features with default (e.g., 0)
        feature_vector = feature_vector[_expected_features]
    else:
        # Fallback if expected features are not defined (less robust)
        print("Warning: No expected features defined for model. Using all available features. This might lead to errors.")

    try:
        # Make prediction
        prediction = _model.predict(feature_vector)[0]
        # Get probability if available
        # prediction_proba = _model.predict_proba(feature_vector)[0][1] # For binary classification

        # Map numeric prediction to human-readable label
        # Assuming 1 for fraud, 0 for normal, or similar
        predicted_label = 'fraud' if prediction == 1 else 'normal' # Adjust based on your model's output
        
        # Get XAI reason code from the updated features (computed by realtime_feature_updater)
        xai_code = wallet_features.get('xai_reason_code', 'no_xai_reason')

        return {'prediction': predicted_label, 'xai_reason_code': xai_code}
    except Exception as e:
        print(f"Error during prediction: {e}")
        return {'prediction': 'error', 'xai_reason_code': wallet_features.get('xai_reason_code', 'prediction_error')}

if __name__ == "__main__":
    print("Running model_inference_service.py directly. Loading model...")
    load_model()
    print("Inference service ready. Intended for import into a live processing script.")

    # Example usage (for direct testing)
    if _model:
        # Create a dummy feature set (replace with actual features from updater)
        dummy_features = {
            'total_transactions': 100,
            'avg_tx_value': 1.5,
            'burst_tx_ratio': 0.98,
            'num_fraud_counterparties': 3,
            'anomaly_iso': 1,
            'xai_reason_code': 'burst_tx|fraud_link|model_anomaly'
            # ... include all features your model expects ...
        }
        print("\n--- Dummy Prediction Test ---")
        result = predict_wallet_risk(dummy_features)
        print(f"Dummy Wallet Prediction: {result['prediction']}")
        print(f"Dummy Wallet XAI Reason: {result['xai_reason_code']}")
    else:
        print("Model not loaded, skipping dummy prediction test.") 