#!/usr/bin/env python3
"""
Test script to verify model types and functionality
"""
import joblib
import os

def test_model(model_path, model_name):
    """Test a model and report its type and capabilities"""
    print(f"\n--- Testing {model_name} ---")
    print(f"Path: {model_path}")
    
    if not os.path.exists(model_path):
        print("❌ Model file not found!")
        return
    
    try:
        model = joblib.load(model_path)
        model_type = type(model).__name__
        print(f"✅ Model loaded successfully")
        print(f"Model type: {model_type}")
        
        # Check capabilities
        has_predict_proba = hasattr(model, 'predict_proba')
        has_predict = hasattr(model, 'predict')
        
        print(f"Has predict_proba: {'✅' if has_predict_proba else '❌'}")
        print(f"Has predict: {'✅' if has_predict else '❌'}")
        
        if has_predict_proba:
            print("🎯 This is a CLASSIFICATION model (good for match predictions)")
        else:
            print("📊 This is a REGRESSION model (for award predictions)")
            
    except Exception as e:
        print(f"❌ Error loading model: {e}")

def main():
    print("🔍 Testing FIFA World Cup Predictor Models")
    print("=" * 50)
    
    # Test all available models
    models_dir = "models"
    
    model_files = [
        ("match_model.pkl", "Match Prediction Model"),
        ("award_model_goals.pkl", "Goals Award Model"),
        ("award_model_assists.pkl", "Assists Award Model"),
    ]
    
    for filename, description in model_files:
        model_path = os.path.join(models_dir, filename)
        test_model(model_path, description)
    
    print("\n" + "=" * 50)
    print("📋 SUMMARY:")
    print("- Use 'match_model.pkl' for match outcome predictions")
    print("- Use 'award_model_*.pkl' for player award predictions")
    print("- Only classification models have predict_proba method")

if __name__ == "__main__":
    main() 