"""
Model evaluation and fairness audit module
"""
import pandas as pd
import numpy as np
from sklearn.metrics import (
    roc_auc_score, precision_score, recall_score, 
    f1_score, accuracy_score, confusion_matrix,
    balanced_accuracy_score, matthews_corrcoef,
    average_precision_score, precision_recall_curve
)
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys
from typing import Dict, List, Tuple

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.utils import load_model, create_age_group


class ModelEvaluator:
    """
    Comprehensive model evaluation including fairness metrics
    """
    
    def __init__(self, model=None, model_path: str = "models/xgb_adr_model.pkl"):
        """
        Initialize evaluator
        
        Args:
            model: Trained model (optional)
            model_path: Path to saved model
        """
        if model is None:
            self.model = load_model(model_path)
        else:
            self.model = model
    
    def compute_metrics(
        self, 
        y_true: np.ndarray, 
        y_pred: np.ndarray, 
        y_pred_proba: np.ndarray
    ) -> Dict:
        """
        Compute comprehensive metrics
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_pred_proba: Predicted probabilities
            
        Returns:
            Dictionary of metrics
        """
        metrics = {
            'auc': roc_auc_score(y_true, y_pred_proba),
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, zero_division=0),
            'recall': recall_score(y_true, y_pred, zero_division=0),
            'f1': f1_score(y_true, y_pred, zero_division=0),
            'n_samples': len(y_true),
            'positive_rate': y_true.mean()
        }
        
        return metrics
    
    def evaluate_fairness_by_group(
        self,
        X: pd.DataFrame,
        y_true: pd.Series,
        group_column: str,
        group_name: str = "Group",
        feature_cols: List[str] = None
    ) -> Dict:
        """
        Evaluate fairness across demographic groups
        
        Args:
            X: Feature DataFrame (must include group column)
            y_true: True labels
            group_column: Column name for grouping
            group_name: Display name for group
            feature_cols: List of feature column names for model prediction
            
        Returns:
            Dictionary with per-group metrics
        """
        print(f"\nEvaluating fairness by {group_name}...")
        
        # Get predictions - only use original feature columns
        if feature_cols is not None:
            X_features = X[feature_cols]
        else:
            # Drop all demographic columns (starting with _)
            X_features = X[[col for col in X.columns if not col.startswith('_')]]
        
        y_pred_proba = self.model.predict_proba(X_features)[:, 1]
        y_pred = (y_pred_proba >= 0.5).astype(int)
        
        # Group-wise evaluation
        group_metrics = {}
        
        for group_value in X[group_column].unique():
            mask = X[group_column] == group_value
            
            if mask.sum() < 10:  # Skip groups with too few samples
                continue
            
            y_true_group = y_true[mask]
            y_pred_group = y_pred[mask]
            y_pred_proba_group = y_pred_proba[mask]
            
            metrics = self.compute_metrics(y_true_group, y_pred_group, y_pred_proba_group)
            group_metrics[str(group_value)] = metrics
            
            print(f"  {group_value}: AUC={metrics['auc']:.3f}, F1={metrics['f1']:.3f}, "
                  f"n={metrics['n_samples']}")
        
        return group_metrics
    
    def compute_fairness_audit(
        self,
        X: pd.DataFrame,
        y_true: pd.Series,
        demographics_df: pd.DataFrame = None
    ) -> Dict:
        """
        Comprehensive fairness audit
        
        Args:
            X: Feature DataFrame
            y_true: True labels
            demographics_df: DataFrame with demographic info (gender, age)
            
        Returns:
            Dictionary with fairness metrics
        """
        print("\n" + "="*60)
        print("FAIRNESS AUDIT")
        print("="*60)
        
        # Overall metrics
        y_pred_proba = self.model.predict_proba(X)[:, 1]
        y_pred = (y_pred_proba >= 0.5).astype(int)
        overall_metrics = self.compute_metrics(y_true, y_pred, y_pred_proba)
        
        print(f"\nOverall Performance:")
        print(f"  AUC: {overall_metrics['auc']:.4f}")
        print(f"  F1: {overall_metrics['f1']:.4f}")
        print(f"  Precision: {overall_metrics['precision']:.4f}")
        print(f"  Recall: {overall_metrics['recall']:.4f}")
        
        fairness_results = {
            'overall': overall_metrics,
            'by_sex': {},
            'by_age_group': {}
        }
        
        # If demographics provided, compute group-wise metrics
        if demographics_df is not None:
            # Ensure indices match
            if len(demographics_df) != len(X):
                print("Warning: Demographics data length mismatch. Skipping fairness analysis.")
                return fairness_results
            
            # Store original feature columns
            original_feature_cols = X.columns.tolist()
            
            # Add demographics to X temporarily
            X_with_demo = X.copy()
            X_with_demo['_gender'] = demographics_df['gender'].values
            X_with_demo['_age'] = demographics_df.get('anchor_age', demographics_df.get('age', 50)).values
            X_with_demo['_age_group'] = X_with_demo['_age'].apply(create_age_group)
            
            # Fairness by sex
            if '_gender' in X_with_demo.columns:
                fairness_results['by_sex'] = self.evaluate_fairness_by_group(
                    X_with_demo, y_true, '_gender', "Sex", feature_cols=original_feature_cols
                )
            
            # Fairness by age group
            if '_age_group' in X_with_demo.columns:
                fairness_results['by_age_group'] = self.evaluate_fairness_by_group(
                    X_with_demo, y_true, '_age_group', "Age Group", feature_cols=original_feature_cols
                )
        
        print("\n" + "="*60)
        print("FAIRNESS AUDIT COMPLETE")
        print("="*60 + "\n")
        
        return fairness_results
    
    def plot_fairness_comparison(
        self,
        fairness_results: Dict,
        metric: str = 'auc',
        save_path: str = "reports/fairness_comparison.png"
    ):
        """
        Plot fairness comparison across groups
        
        Args:
            fairness_results: Results from compute_fairness_audit
            metric: Metric to compare ('auc', 'f1', etc.)
            save_path: Path to save plot
        """
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Plot by sex
        if fairness_results['by_sex']:
            groups = list(fairness_results['by_sex'].keys())
            values = [fairness_results['by_sex'][g][metric] for g in groups]
            
            axes[0].bar(groups, values, color=['#3498db', '#e74c3c'])
            axes[0].axhline(y=fairness_results['overall'][metric], 
                           color='green', linestyle='--', label='Overall')
            axes[0].set_ylabel(metric.upper())
            axes[0].set_title(f'{metric.upper()} by Sex')
            axes[0].legend()
            axes[0].set_ylim(0, 1)
        
        # Plot by age group
        if fairness_results['by_age_group']:
            groups = list(fairness_results['by_age_group'].keys())
            values = [fairness_results['by_age_group'][g][metric] for g in groups]
            
            axes[1].bar(groups, values, color=['#2ecc71', '#f39c12', '#9b59b6'])
            axes[1].axhline(y=fairness_results['overall'][metric], 
                           color='green', linestyle='--', label='Overall')
            axes[1].set_ylabel(metric.upper())
            axes[1].set_title(f'{metric.upper()} by Age Group')
            axes[1].legend()
            axes[1].set_ylim(0, 1)
        
        plt.tight_layout()
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"Fairness comparison plot saved to {save_path}")
    
    def compute_calibration(
        self,
        y_true: np.ndarray,
        y_pred_proba: np.ndarray,
        n_bins: int = 10
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute calibration curve
        
        Args:
            y_true: True labels
            y_pred_proba: Predicted probabilities
            n_bins: Number of calibration bins
            
        Returns:
            Tuple of (predicted_probs, actual_probs)
        """
        bins = np.linspace(0, 1, n_bins + 1)
        bin_indices = np.digitize(y_pred_proba, bins) - 1
        bin_indices = np.clip(bin_indices, 0, n_bins - 1)
        
        predicted_probs = []
        actual_probs = []
        
        for i in range(n_bins):
            mask = bin_indices == i
            if mask.sum() > 0:
                predicted_probs.append(y_pred_proba[mask].mean())
                actual_probs.append(y_true[mask].mean())
        
        return np.array(predicted_probs), np.array(actual_probs)
    
    def plot_calibration_curve(
        self,
        y_true: np.ndarray,
        y_pred_proba: np.ndarray,
        save_path: str = "reports/calibration_curve.png"
    ):
        """
        Plot calibration curve
        
        Args:
            y_true: True labels
            y_pred_proba: Predicted probabilities
            save_path: Path to save plot
        """
        pred_probs, actual_probs = self.compute_calibration(y_true, y_pred_proba)
        
        plt.figure(figsize=(8, 6))
        plt.plot(pred_probs, actual_probs, 'o-', label='Model', linewidth=2, markersize=8)
        plt.plot([0, 1], [0, 1], 'k--', label='Perfect Calibration')
        plt.xlabel('Predicted Probability')
        plt.ylabel('Actual Probability')
        plt.title('Calibration Curve')
        plt.legend()
        plt.grid(alpha=0.3)
        plt.tight_layout()
        
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"Calibration curve saved to {save_path}")
    
    def generate_evaluation_report(
        self,
        X: pd.DataFrame,
        y_true: pd.Series,
        demographics_df: pd.DataFrame = None,
        save_dir: str = "reports"
    ) -> Dict:
        """
        Generate comprehensive evaluation report
        
        Args:
            X: Features
            y_true: True labels
            demographics_df: Demographics for fairness audit
            save_dir: Directory to save reports
            
        Returns:
            Dictionary with all evaluation results
        """
        print("\n" + "="*60)
        print("GENERATING EVALUATION REPORT")
        print("="*60 + "\n")
        
        os.makedirs(save_dir, exist_ok=True)
        
        # Predictions
        y_pred_proba = self.model.predict_proba(X)[:, 1]
        y_pred = (y_pred_proba >= 0.5).astype(int)
        
        # Overall metrics
        overall_metrics = self.compute_metrics(y_true, y_pred, y_pred_proba)
        
        print("Overall Metrics:")
        for key, value in overall_metrics.items():
            if isinstance(value, float):
                print(f"  {key}: {value:.4f}")
        
        # Fairness audit
        fairness_results = self.compute_fairness_audit(X, y_true, demographics_df)
        
        # Calibration
        self.plot_calibration_curve(
            y_true.values, 
            y_pred_proba,
            save_path=os.path.join(save_dir, "calibration_curve.png")
        )
        
        # Fairness comparison plots
        for metric in ['auc', 'f1']:
            self.plot_fairness_comparison(
                fairness_results,
                metric=metric,
                save_path=os.path.join(save_dir, f"fairness_{metric}.png")
            )
        
        # Save results to CSV
        results_df = pd.DataFrame([overall_metrics])
        results_df.to_csv(os.path.join(save_dir, "evaluation_metrics.csv"), index=False)
        
        print("\n" + "="*60)
        print("EVALUATION REPORT COMPLETE")
        print("="*60 + "\n")
        
        return {
            'overall': overall_metrics,
            'fairness': fairness_results
        }


def main():
    """
    Main function to run evaluation
    """
    print("\n" + "="*60)
    print("MODEL EVALUATION & FAIRNESS AUDIT")
    print("="*60 + "\n")
    
    # Get project root
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    # Load model
    try:
        model_path = os.path.join(project_root, "models", "xgb_adr_model.pkl")
        model = load_model(model_path)
        print("Model loaded")
    except FileNotFoundError:
        print("Error: Model not found. Please train the model first.")
        return
    
    # Load test data
    try:
        data_dir = os.path.join(project_root, "data", "output")
        X = pd.read_csv(os.path.join(data_dir, "X_features.csv"))
        y = pd.read_csv(os.path.join(data_dir, "y_target.csv")).squeeze()
        
        # Load merged dataset for demographics
        merged = pd.read_csv(os.path.join(data_dir, "merged_dataset.csv"))
        demographics = merged[['gender', 'anchor_age']] if 'gender' in merged.columns else None
        
        print(f"Loaded {len(X)} samples")
        
    except FileNotFoundError:
        print("Error: Data not found. Please run preprocessing first.")
        return
    
    # Use test split (last 15%)
    test_size = int(0.15 * len(X))
    X_test = X.iloc[-test_size:]
    y_test = y.iloc[-test_size:]
    demographics_test = demographics.iloc[-test_size:] if demographics is not None else None
    
    # Create evaluator
    evaluator = ModelEvaluator(model=model)
    
    # Generate report
    results = evaluator.generate_evaluation_report(
        X_test, 
        y_test, 
        demographics_test,
        save_dir="reports"
    )
    
    return results


if __name__ == "__main__":
    main()

