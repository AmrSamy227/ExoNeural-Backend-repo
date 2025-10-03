#!/usr/bin/env python3
"""
ExoNeural Backend API
NASA Space Apps Challenge 2025
Team: ExoNeural

Flask API for exoplanet detection predictions using LightGBM model
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import pandas as pd
import joblib
import logging
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

class ExoplanetPredictor:
    """
    LightGBM model for exoplanet detection with 31 features
    Matches your ML engineer's trained model exactly
    """

    def __init__(self):
        self.model = None
        self.scaler = None
        self.imputer = None
        self.model_loaded = False
        self.load_models()

        # Class names matching your notebook
        self.class_names = ["False Positive", "Candidate Exoplanet", "Confirmed Exoplanet"]

        # Expected feature columns in the correct order
        self.feature_columns = [
            'koi_period', 'koi_prad', 'koi_sma', 'koi_incl', 'koi_teq',
            'koi_insol', 'koi_impact', 'koi_duration', 'koi_depth', 'koi_dor',
            'koi_eccen', 'koi_ror', 'koi_steff', 'koi_slogg', 'koi_smet',
            'koi_srad', 'koi_smass', 'koi_srho', 'koi_num_transits', 'koi_count',
            'koi_model_snr', 'koi_fpflag_nt', 'koi_fpflag_ss', 'koi_fpflag_co',
            'koi_fpflag_ec', 'planet_star_radius_ratio', 'period_duration_ratio',
            'star_density_proxy', 'insol_teq_ratio', 'signal_strength', 'total_fp_flags'
        ]

    def load_models(self):
        """Load the LightGBM model and preprocessing objects"""
        try:
            model_path = "exoplanet_model_multiclass.pkl"
            scaler_path = "scaler.pkl"
            imputer_path = "imputer.pkl"

            if os.path.exists(model_path):
                self.model = joblib.load(model_path)
                logger.info("✅ LightGBM model loaded successfully")

                # Try to load scaler and imputer if they exist
                if os.path.exists(scaler_path):
                    self.scaler = joblib.load(scaler_path)
                    logger.info("✅ Scaler loaded successfully")

                if os.path.exists(imputer_path):
                    self.imputer = joblib.load(imputer_path)
                    logger.info("✅ Imputer loaded successfully")

                self.model_loaded = True
            else:
                logger.warning(f"⚠️ Model file {model_path} not found")
                self.model_loaded = False
        except Exception as e:
            logger.error(f"❌ Error loading models: {str(e)}")
            self.model_loaded = False

    def calculate_derived_features(self, data):
        """
        Calculate the 6 engineered features from the 25 input features
        Matches the feature engineering in your notebook
        """
        try:
            # planet_star_radius_ratio = koi_prad / koi_srad
            data['planet_star_radius_ratio'] = data['koi_prad'] / data['koi_srad']

            # period_duration_ratio = koi_period / koi_duration
            data['period_duration_ratio'] = data['koi_period'] / data['koi_duration']

            # star_density_proxy = koi_smass / (koi_srad ** 3)
            data['star_density_proxy'] = data['koi_smass'] / (data['koi_srad'] ** 3)

            # insol_teq_ratio = koi_insol / koi_teq
            data['insol_teq_ratio'] = data['koi_insol'] / data['koi_teq']

            # signal_strength = koi_depth * koi_model_snr
            data['signal_strength'] = data['koi_depth'] * data['koi_model_snr']

            # total_fp_flags = sum of all false positive flags
            data['total_fp_flags'] = (
                data['koi_fpflag_nt'] +
                data['koi_fpflag_ss'] +
                data['koi_fpflag_co'] +
                data['koi_fpflag_ec']
            )

            return data
        except Exception as e:
            logger.error(f"Error calculating derived features: {str(e)}")
            raise

    def predict(self, input_data):
        """
        Main prediction method
        Expects a dictionary with all 25 parameters
        """
        try:
            if not self.model_loaded or self.model is None:
                return {
                    "prediction": "Error",
                    "confidence": 0.0,
                    "error": "Model not loaded. Please ensure model files exist."
                }

            # Create DataFrame from input data
            df = pd.DataFrame([input_data])

            # Calculate the 6 derived features
            df = self.calculate_derived_features(df)

            # Ensure columns are in the correct order
            df = df[self.feature_columns]

            # Apply preprocessing if available
            if self.imputer is not None:
                df = pd.DataFrame(
                    self.imputer.transform(df),
                    columns=self.feature_columns
                )

            if self.scaler is not None:
                df_scaled = self.scaler.transform(df)
            else:
                df_scaled = df.values

            # Make prediction
            prediction = int(self.model.predict(df_scaled)[0])
            probabilities = self.model.predict_proba(df_scaled)[0]

            return {
                "prediction": self.class_names[prediction],
                "confidence": round(float(probabilities[prediction]), 4),
                "raw_prediction": prediction,
                "probabilities": {
                    "false_positive": round(float(probabilities[0]), 4),
                    "candidate": round(float(probabilities[1]), 4),
                    "confirmed": round(float(probabilities[2]), 4)
                }
            }

        except Exception as e:
            logger.error(f"Prediction error: {str(e)}")
            return {
                "prediction": "Error",
                "confidence": 0.0,
                "error": str(e)
            }

# Initialize the predictor
predictor = ExoplanetPredictor()

@app.route('/', methods=['GET'])
def health_check():
    return jsonify({
        "status": "healthy",
        "service": "ExoNeural API",
        "version": "2.0.0",
        "team": "ExoNeural Team - NASA Space Apps Challenge 2025",
        "model_loaded": predictor.model_loaded
    })

@app.route('/predict', methods=['POST'])
def predict_exoplanet():
    """
    Predict exoplanet classification from 25 input parameters
    """
    try:
        if not request.is_json:
            return jsonify({"error": "Request must be JSON", "status": "error"}), 400

        data = request.get_json()

        # Define all 25 required fields
        required_fields = [
            'koi_period', 'koi_prad', 'koi_sma', 'koi_incl', 'koi_teq',
            'koi_insol', 'koi_impact', 'koi_duration', 'koi_depth', 'koi_dor',
            'koi_eccen', 'koi_ror', 'koi_steff', 'koi_slogg', 'koi_smet',
            'koi_srad', 'koi_smass', 'koi_srho', 'koi_num_transits', 'koi_count',
            'koi_model_snr', 'koi_fpflag_nt', 'koi_fpflag_ss', 'koi_fpflag_co',
            'koi_fpflag_ec'
        ]

        # Check for missing fields
        missing_fields = [f for f in required_fields if f not in data]
        if missing_fields:
            return jsonify({
                "error": f"Missing required fields: {', '.join(missing_fields)}",
                "status": "error"
            }), 400

        # Convert all values to float
        input_data = {}
        for field in required_fields:
            try:
                input_data[field] = float(data[field])
            except (ValueError, TypeError):
                return jsonify({
                    "error": f"Invalid value for field '{field}'. Must be numeric.",
                    "status": "error"
                }), 400

        # Make prediction
        result = predictor.predict(input_data)

        # Add metadata
        result.update({
            "status": "success",
            "model_version": "ExoNeural-v2.0",
            "timestamp": pd.Timestamp.now().isoformat()
        })

        return jsonify(result)

    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        return jsonify({
            "error": "Internal server error",
            "status": "error",
            "details": str(e)
        }), 500

@app.route('/batch_predict', methods=['POST'])
def batch_predict():
    """
    Batch prediction for multiple exoplanet candidates
    """
    try:
        if not request.is_json:
            return jsonify({"error": "Request must be JSON", "status": "error"}), 400

        data = request.get_json()
        if 'data' not in data or not isinstance(data['data'], list):
            return jsonify({
                "error": "Request must contain 'data' array",
                "status": "error"
            }), 400

        results = []
        for i, item in enumerate(data['data']):
            try:
                result = predictor.predict(item)
                result['row_index'] = i
                results.append(result)
            except Exception as e:
                results.append({
                    "row_index": i,
                    "prediction": "Error",
                    "confidence": 0.0,
                    "error": str(e)
                })

        return jsonify({
            "status": "success",
            "results": results,
            "total_processed": len(results)
        })

    except Exception as e:
        logger.error(f"Batch prediction error: {str(e)}")
        return jsonify({"error": "Internal server error", "status": "error"}), 500

if __name__ == '__main__':
    logger.info("🚀 Starting ExoNeural API server...")
    logger.info(f"📊 Model loaded: {predictor.model_loaded}")
    port = int(os.environ.get("PORT", 5000))  # Use Render’s assigned port
    app.run(debug=True, host='0.0.0.0', port=port)


