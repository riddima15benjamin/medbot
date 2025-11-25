"""
Utility functions for AI-CPA system
"""
import os
import json
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Tuple, Any
import joblib


def get_risk_category(risk_score: float) -> str:
    """
    Categorize ADR risk score into Low/Moderate/High
    
    Args:
        risk_score: Predicted probability (0-1)
        
    Returns:
        Risk category string
    """
    if risk_score < 0.3:
        return "Low"
    elif risk_score < 0.7:
        return "Moderate"
    else:
        return "High"


def get_risk_color(risk_category: str) -> str:
    """
    Get color code for risk category
    
    Args:
        risk_category: Low/Moderate/High
        
    Returns:
        Color code
    """
    colors = {
        "Low": "#28a745",      # green
        "Moderate": "#fd7e14",  # orange
        "High": "#dc3545"       # red
    }
    return colors.get(risk_category, "#6c757d")


def load_model(model_path: str = "models/xgb_adr_model.pkl"):
    """
    Load trained XGBoost model
    
    Args:
        model_path: Path to saved model
        
    Returns:
        Loaded model
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at {model_path}. Please train the model first.")
    return joblib.load(model_path)


def load_metadata(metadata_path: str = "models/model_metadata.pkl"):
    """
    Load model metadata (feature names, etc.)
    
    Args:
        metadata_path: Path to metadata file
        
    Returns:
        Metadata dictionary
    """
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"Metadata not found at {metadata_path}")
    return joblib.load(metadata_path)


def save_model(model, metadata: Dict, model_path: str = "models/xgb_adr_model.pkl"):
    """
    Save trained model and metadata
    
    Args:
        model: Trained model
        metadata: Dictionary containing feature names and other metadata
        model_path: Path to save model
    """
    joblib.dump(model, model_path)
    metadata_path = model_path.replace(".pkl", "_metadata.pkl")
    joblib.dump(metadata, metadata_path)
    print(f"Model saved to {model_path}")
    print(f"Metadata saved to {metadata_path}")


def clean_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean DataFrame column names
    
    Args:
        df: Input DataFrame
        
    Returns:
        DataFrame with cleaned column names
    """
    df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_').str.replace('[^a-z0-9_]', '', regex=True)
    return df


def create_age_group(age: int) -> str:
    """
    Create age group category
    
    Args:
        age: Age in years
        
    Returns:
        Age group string
    """
    if age < 40:
        return "<40"
    elif age <= 60:
        return "40-60"
    else:
        return ">60"


def format_timestamp() -> str:
    """
    Get formatted timestamp for reports
    
    Returns:
        Formatted timestamp string
    """
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def generate_report_filename(patient_id: str = None) -> str:
    """
    Generate unique report filename
    
    Args:
        patient_id: Optional patient identifier
        
    Returns:
        Report filename
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if patient_id:
        return f"ADR_Report_{patient_id}_{timestamp}.csv"
    return f"ADR_Report_{timestamp}.csv"


def parse_fhir_patient(fhir_json: Dict) -> Dict:
    """
    Parse FHIR patient resource into model-ready format
    
    Args:
        fhir_json: FHIR patient JSON
        
    Returns:
        Dictionary with extracted features
    """
    try:
        patient_data = {}
        
        # Extract basic demographics
        if 'gender' in fhir_json:
            patient_data['gender'] = fhir_json['gender'].upper()
        
        if 'birthDate' in fhir_json:
            birth_date = pd.to_datetime(fhir_json['birthDate'])
            age = (datetime.now() - birth_date).days // 365
            patient_data['age'] = age
            patient_data['age_group'] = create_age_group(age)
        
        # Extract medications if present
        if 'medicationRequest' in fhir_json:
            medications = []
            for med in fhir_json['medicationRequest']:
                if 'medicationCodeableConcept' in med:
                    med_name = med['medicationCodeableConcept'].get('text', '')
                    if med_name:
                        medications.append(med_name)
            patient_data['medications'] = medications
        
        return patient_data
        
    except Exception as e:
        print(f"Error parsing FHIR: {e}")
        return {}


def normalize_drug_name(drug_name: str) -> str:
    """
    Normalize drug name for matching
    
    Args:
        drug_name: Raw drug name
        
    Returns:
        Normalized drug name
    """
    if pd.isna(drug_name):
        return ""
    return str(drug_name).strip().upper()


def compute_polypharmacy_features(num_drugs: int) -> Dict[str, int]:
    """
    Compute polypharmacy-related features
    
    Args:
        num_drugs: Number of prescribed drugs
        
    Returns:
        Dictionary with polypharmacy features
    """
    return {
        'num_drugs': num_drugs,
        'polypharmacy_flag': 1 if num_drugs >= 5 else 0,
        'major_polypharmacy_flag': 1 if num_drugs >= 10 else 0
    }


def handle_missing_values(df: pd.DataFrame, strategy: str = 'median') -> pd.DataFrame:
    """
    Handle missing values in DataFrame
    
    Args:
        df: Input DataFrame
        strategy: Imputation strategy ('median', 'mean', 'zero')
        
    Returns:
        DataFrame with imputed values
    """
    df_copy = df.copy()
    
    for col in df_copy.columns:
        if df_copy[col].dtype in ['float64', 'int64']:
            if strategy == 'median':
                df_copy[col].fillna(df_copy[col].median(), inplace=True)
            elif strategy == 'mean':
                df_copy[col].fillna(df_copy[col].mean(), inplace=True)
            elif strategy == 'zero':
                df_copy[col].fillna(0, inplace=True)
    
    return df_copy


def validate_input_data(data: Dict, required_fields: List[str]) -> Tuple[bool, str]:
    """
    Validate input data for prediction
    
    Args:
        data: Input data dictionary
        required_fields: List of required field names
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    missing_fields = [field for field in required_fields if field not in data]
    
    if missing_fields:
        return False, f"Missing required fields: {', '.join(missing_fields)}"
    
    return True, ""


def create_prediction_summary(
    risk_score: float,
    top_features: List[Tuple[str, float]],
    patient_data: Dict
) -> Dict[str, Any]:
    """
    Create comprehensive prediction summary
    
    Args:
        risk_score: Predicted ADR risk score
        top_features: List of (feature_name, contribution) tuples
        patient_data: Original patient data
        
    Returns:
        Prediction summary dictionary
    """
    risk_category = get_risk_category(risk_score)
    
    # Extract top contributors
    top_contributors = [f"{feat}: {val:.3f}" for feat, val in top_features[:5]]
    
    # Generate explanation
    if risk_category == "High":
        explanation = f"High ADR risk detected. Primary factors: {', '.join([feat for feat, _ in top_features[:3]])}"
    elif risk_category == "Moderate":
        explanation = f"Moderate ADR risk. Monitor: {', '.join([feat for feat, _ in top_features[:3]])}"
    else:
        explanation = "Low ADR risk. Continue routine monitoring."
    
    return {
        "risk_score": float(risk_score),
        "risk_level": risk_category,
        "risk_color": get_risk_color(risk_category),
        "top_contributors": top_contributors,
        "explanation": explanation,
        "timestamp": format_timestamp(),
        "patient_summary": {
            "age": patient_data.get("age", "N/A"),
            "gender": patient_data.get("gender", "N/A"),
            "num_drugs": patient_data.get("num_drugs", 0)
        }
    }


if __name__ == "__main__":
    # Test utility functions
    print("Testing utility functions...")
    
    # Test risk categorization
    for score in [0.2, 0.5, 0.85]:
        category = get_risk_category(score)
        color = get_risk_color(category)
        print(f"Score: {score} -> Category: {category}, Color: {color}")
    
    # Test age groups
    for age in [25, 55, 75]:
        group = create_age_group(age)
        print(f"Age: {age} -> Group: {group}")
    
    print("\nUtility functions working correctly!")

