"""
SHAP explainability module for ADR predictions
"""
import shap
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
import os
import sys
from typing import List, Tuple, Dict
import warnings
warnings.filterwarnings('ignore')

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.utils import load_model, load_metadata


class SHAPExplainer:
    """
    SHAP-based explainer for XGBoost ADR model
    """
    
    def __init__(self, model=None, model_path: str = "models/xgb_adr_model.pkl"):
        """
        Initialize SHAP explainer
        
        Args:
            model: Trained model (optional)
            model_path: Path to saved model
        """
        if model is None:
            self.model = load_model(model_path)
        else:
            self.model = model
            
        self.explainer = None
        self.feature_names = []
        
        # Load metadata if available
        try:
            metadata_path = model_path.replace(".pkl", "_metadata.pkl")
            metadata = load_metadata(metadata_path)
            self.feature_names = metadata.get('feature_names', [])
        except:
            print("Warning: Could not load feature names from metadata")
    
    def create_explainer(self, X_background: pd.DataFrame = None):
        """
        Create SHAP TreeExplainer
        
        Args:
            X_background: Background data for explainer (optional)
        """
        print("Creating SHAP TreeExplainer...")
        
        if X_background is not None:
            # Use subset of data as background (for speed)
            if len(X_background) > 100:
                background = shap.sample(X_background, 100)
            else:
                background = X_background
            self.explainer = shap.TreeExplainer(self.model, background)
        else:
            self.explainer = shap.TreeExplainer(self.model)
        
        print("✓ SHAP explainer created")
        
    def save_explainer(self, path: str = "models/shap_explainer.pkl"):
        """
        Save SHAP explainer
        
        Args:
            path: Path to save explainer
        """
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump(self.explainer, path)
        print(f"✓ SHAP explainer saved to {path}")
    
    def load_explainer(self, path: str = "models/shap_explainer.pkl"):
        """
        Load saved SHAP explainer
        
        Args:
            path: Path to explainer file
        """
        if os.path.exists(path):
            self.explainer = joblib.load(path)
            print(f"✓ SHAP explainer loaded from {path}")
        else:
            print(f"Warning: Explainer not found at {path}")
    
    def compute_shap_values(self, X: pd.DataFrame) -> shap.Explanation:
        """
        Compute SHAP values for given data
        
        Args:
            X: Input features
            
        Returns:
            SHAP Explanation object
        """
        if self.explainer is None:
            raise ValueError("Explainer not initialized. Call create_explainer() first.")
        
        print(f"Computing SHAP values for {len(X)} samples...")
        shap_values = self.explainer(X)
        print("✓ SHAP values computed")
        
        return shap_values
    
    def get_global_importance(
        self, 
        X: pd.DataFrame, 
        top_n: int = 20,
        save_path: str = None
    ) -> pd.DataFrame:
        """
        Get global feature importance using SHAP
        
        Args:
            X: Input features
            top_n: Number of top features to return
            save_path: Optional path to save plot
            
        Returns:
            DataFrame with feature importances
        """
        print("\nComputing global feature importance...")
        
        # Compute SHAP values
        shap_values = self.compute_shap_values(X)
        
        # Calculate mean absolute SHAP values
        mean_abs_shap = np.abs(shap_values.values).mean(axis=0)
        
        # Create DataFrame
        if len(self.feature_names) == len(mean_abs_shap):
            feature_names = self.feature_names
        else:
            feature_names = X.columns.tolist()
        
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': mean_abs_shap
        }).sort_values('importance', ascending=False)
        
        # Print top features
        print(f"\nTop {top_n} features by SHAP importance:")
        for i, row in importance_df.head(top_n).iterrows():
            print(f"  {row['feature']}: {row['importance']:.4f}")
        
        # Create plot if requested
        if save_path:
            self.plot_global_importance(shap_values, save_path)
        
        return importance_df
    
    def plot_global_importance(
        self, 
        shap_values: shap.Explanation,
        save_path: str = "reports/shap_global_importance.png",
        max_display: int = 20
    ):
        """
        Plot global SHAP summary
        
        Args:
            shap_values: SHAP values
            save_path: Path to save plot
            max_display: Maximum features to display
        """
        print(f"\nCreating global SHAP summary plot...")
        
        plt.figure(figsize=(10, 8))
        shap.summary_plot(
            shap_values.values, 
            shap_values.data,
            feature_names=self.feature_names if self.feature_names else None,
            max_display=max_display,
            show=False
        )
        plt.tight_layout()
        
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Global importance plot saved to {save_path}")
    
    def get_local_explanation(
        self, 
        X_single: pd.DataFrame,
        top_n: int = 10
    ) -> Tuple[List[Tuple[str, float]], shap.Explanation]:
        """
        Get local explanation for single prediction
        
        Args:
            X_single: Single patient features (1 row DataFrame)
            top_n: Number of top contributors to return
            
        Returns:
            Tuple of (top_contributors list, shap_values)
        """
        if len(X_single) != 1:
            raise ValueError("X_single must contain exactly 1 sample")
        
        # Compute SHAP values
        shap_values = self.compute_shap_values(X_single)
        
        # Get feature names
        if self.feature_names:
            feature_names = self.feature_names
        else:
            feature_names = X_single.columns.tolist()
        
        # Create list of (feature, shap_value) tuples
        shap_contributions = list(zip(feature_names, shap_values.values[0]))
        
        # Sort by absolute contribution
        shap_contributions.sort(key=lambda x: abs(x[1]), reverse=True)
        
        # Get top N
        top_contributors = shap_contributions[:top_n]
        
        return top_contributors, shap_values
    
    def plot_local_explanation(
        self,
        shap_values: shap.Explanation,
        save_path: str = None,
        plot_type: str = "waterfall"
    ) -> plt.Figure:
        """
        Plot local explanation for single prediction
        
        Args:
            shap_values: SHAP values for single sample
            save_path: Optional path to save plot
            plot_type: Type of plot ('waterfall' or 'force')
            
        Returns:
            Matplotlib figure
        """
        if plot_type == "waterfall":
            fig = plt.figure(figsize=(10, 6))
            shap.plots.waterfall(shap_values[0], show=False)
            plt.tight_layout()
            
        elif plot_type == "force":
            # Force plot returns HTML, convert to image
            fig = plt.figure(figsize=(12, 3))
            shap.plots.force(shap_values[0], show=False, matplotlib=True)
            plt.tight_layout()
        
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"✓ Local explanation plot saved to {save_path}")
        
        return fig
    
    def explain_prediction(
        self,
        X_single: pd.DataFrame,
        return_plot: bool = True
    ) -> Dict:
        """
        Complete explanation for single prediction
        
        Args:
            X_single: Single patient features
            return_plot: Whether to generate plot
            
        Returns:
            Dictionary with explanation details
        """
        # Get prediction
        pred_proba = self.model.predict_proba(X_single)[0, 1]
        
        # Get SHAP explanation
        top_contributors, shap_values = self.get_local_explanation(X_single, top_n=10)
        
        # Format contributors
        contributors_formatted = []
        for feature, shap_val in top_contributors:
            direction = "↑" if shap_val > 0 else "↓"
            contributors_formatted.append({
                'feature': feature,
                'shap_value': float(shap_val),
                'direction': direction,
                'magnitude': abs(float(shap_val))
            })
        
        # Create explanation text
        top_3 = [c['feature'] for c in contributors_formatted[:3]]
        explanation_text = f"Risk driven by: {', '.join(top_3)}"
        
        result = {
            'prediction': float(pred_proba),
            'top_contributors': contributors_formatted,
            'explanation_text': explanation_text,
            'base_value': float(shap_values.base_values[0])
        }
        
        # Add plot if requested
        if return_plot:
            fig = self.plot_local_explanation(shap_values, plot_type="waterfall")
            result['plot'] = fig
        
        return result
    
    def batch_explain(
        self,
        X: pd.DataFrame,
        save_summary: bool = True,
        summary_path: str = "reports/shap_summary.png"
    ) -> Dict:
        """
        Explain multiple predictions
        
        Args:
            X: Multiple patient features
            save_summary: Whether to save summary plot
            summary_path: Path for summary plot
            
        Returns:
            Dictionary with batch explanation results
        """
        print(f"\nExplaining {len(X)} predictions...")
        
        # Compute SHAP values
        shap_values = self.compute_shap_values(X)
        
        # Get global importance
        importance_df = self.get_global_importance(X, top_n=20, save_path=None)
        
        # Save summary plot if requested
        if save_summary:
            self.plot_global_importance(shap_values, summary_path)
        
        result = {
            'num_samples': len(X),
            'global_importance': importance_df.to_dict('records'),
            'shap_values': shap_values
        }
        
        return result


def main():
    """
    Main function to create and save SHAP explainer
    """
    print("\n" + "="*60)
    print("CREATING SHAP EXPLAINER")
    print("="*60 + "\n")
    
    # Load model
    try:
        model = load_model("models/xgb_adr_model.pkl")
        print("✓ Model loaded")
    except FileNotFoundError:
        print("Error: Model not found. Please train the model first.")
        return
    
    # Load test data for background
    try:
        X_test = pd.read_csv("data/output/X_features.csv")
        print(f"✓ Loaded {len(X_test)} samples for background data")
        
        # Use subset for background
        X_background = X_test.sample(min(100, len(X_test)), random_state=42)
        
    except FileNotFoundError:
        print("Warning: Could not load feature data. Creating explainer without background.")
        X_background = None
    
    # Create explainer
    explainer = SHAPExplainer(model=model)
    explainer.create_explainer(X_background)
    
    # Save explainer
    explainer.save_explainer()
    
    # Compute and save global importance
    if X_background is not None:
        explainer.get_global_importance(
            X_background, 
            top_n=20,
            save_path="reports/shap_global_importance.png"
        )
    
    print("\n" + "="*60)
    print("SHAP EXPLAINER READY")
    print("="*60 + "\n")
    
    return explainer


if __name__ == "__main__":
    main()

