"""
XGBoost model training module for ADR prediction
"""
import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score, precision_score, recall_score, 
    f1_score, accuracy_score, confusion_matrix,
    classification_report, roc_curve, precision_recall_curve,
    balanced_accuracy_score, matthews_corrcoef,
    average_precision_score, log_loss
)
import joblib
import os
import sys
from typing import Tuple, Dict
import matplotlib.pyplot as plt
import seaborn as sns
from imblearn.over_sampling import SMOTE

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.preprocess import MIMICFAERSPreprocessor
from src.utils import save_model


class ADRModelTrainer:
    """
    Trainer for XGBoost ADR prediction model
    """
    
    def __init__(self, use_smote: bool = True, use_class_weights: bool = True):
        """
        Initialize trainer
        
        Args:
            use_smote: Whether to use SMOTE for oversampling
            use_class_weights: Whether to use class weights in XGBoost
        """
        self.model = None
        self.feature_names = []
        self.metrics = {}
        self.splits = {}
        self.use_smote = use_smote
        self.use_class_weights = use_class_weights
        self.smote = SMOTE(random_state=42) if use_smote else None
        self.class_weights = None
        self.optimal_threshold = 0.5
    
    def compute_class_weights(self, y: pd.Series) -> dict:
        """
        Compute class weights for imbalanced data
        
        Args:
            y: Target labels
            
        Returns:
            Dictionary of class weights
        """
        from sklearn.utils.class_weight import compute_class_weight
        
        classes = np.unique(y)
        weights = compute_class_weight('balanced', classes=classes, y=y)
        class_weights = dict(zip(classes, weights))
        
        print(f"Class distribution: {y.value_counts().to_dict()}")
        print(f"Computed class weights: {class_weights}")
        
        return class_weights
    
    def apply_class_balancing(self, X: pd.DataFrame, y: pd.Series) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Apply class balancing techniques
        
        Args:
            X: Features
            y: Target labels
            
        Returns:
            Balanced (X, y)
        """
        print(f"Original data shape: {X.shape}")
        print(f"Original class distribution: {y.value_counts().to_dict()}")
        
        # Apply SMOTE if enabled
        if self.use_smote and self.smote is not None:
            try:
                X_balanced, y_balanced = self.smote.fit_resample(X, y)
                print(f"After SMOTE - Data shape: {X_balanced.shape}")
                print(f"After SMOTE - Class distribution: {pd.Series(y_balanced).value_counts().to_dict()}")
                
                # Convert back to DataFrames/Series
                X_balanced = pd.DataFrame(X_balanced, columns=X.columns)
                y_balanced = pd.Series(y_balanced, name=y.name)
                
                return X_balanced, y_balanced
            except Exception as e:
                print(f"SMOTE failed: {e}. Using original data.")
                return X, y
        
        return X, y
        
    def load_or_preprocess_data(self) -> Tuple[pd.DataFrame, pd.Series, list]:
        """
        Load preprocessed data or run preprocessing
        
        Returns:
            Tuple of (X, y, feature_names)
        """
        # Check if preprocessed data exists
        feature_path = "data/output/X_features.csv"
        target_path = "data/output/y_target.csv"
        
        if os.path.exists(feature_path) and os.path.exists(target_path):
            print("Loading existing preprocessed data...")
            X = pd.read_csv(feature_path)
            y = pd.read_csv(target_path).squeeze()
            feature_names = X.columns.tolist()
            print(f"Loaded X: {X.shape}, y: {y.shape}")
        else:
            print("Preprocessed data not found. Running preprocessing...")
            preprocessor = MIMICFAERSPreprocessor()
            X, y, feature_names, _ = preprocessor.process_full_pipeline()
        
        return X, y, feature_names
    
    def split_data(
        self, 
        X: pd.DataFrame, 
        y: pd.Series,
        train_size: float = 0.7,
        val_size: float = 0.15,
        random_state: int = 42
    ) -> Dict:
        """
        Split data into train/val/test sets
        
        Args:
            X: Features
            y: Target
            train_size: Proportion for training
            val_size: Proportion for validation
            random_state: Random seed
            
        Returns:
            Dictionary with split data
        """
        print(f"\nSplitting data: {train_size:.0%} train, {val_size:.0%} val, "
              f"{1-train_size-val_size:.0%} test")
        
        # First split: train vs (val + test)
        X_train, X_temp, y_train, y_temp = train_test_split(
            X, y, train_size=train_size, random_state=random_state, stratify=y
        )
        
        # Second split: val vs test
        val_ratio = val_size / (1 - train_size)
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, train_size=val_ratio, random_state=random_state, stratify=y_temp
        )
        
        self.splits = {
            'X_train': X_train, 'y_train': y_train,
            'X_val': X_val, 'y_val': y_val,
            'X_test': X_test, 'y_test': y_test
        }
        
        print(f"Train set: {X_train.shape}, ADR rate: {y_train.mean():.2%}")
        print(f"Val set: {X_val.shape}, ADR rate: {y_val.mean():.2%}")
        print(f"Test set: {X_test.shape}, ADR rate: {y_test.mean():.2%}")
        
        return self.splits
    
    def handle_imbalance(self, X_train: pd.DataFrame, y_train: pd.Series) -> Tuple:
        """
        Handle class imbalance using SMOTE
        
        Args:
            X_train: Training features
            y_train: Training target
            
        Returns:
            Balanced X_train, y_train
        """
        print("\nHandling class imbalance with SMOTE...")
        
        original_dist = y_train.value_counts()
        print(f"Original distribution:\n{original_dist}")
        
        smote = SMOTE(random_state=42, k_neighbors=5)
        X_balanced, y_balanced = smote.fit_resample(X_train, y_train)
        
        balanced_dist = pd.Series(y_balanced).value_counts()
        print(f"Balanced distribution:\n{balanced_dist}")
        
        return X_balanced, y_balanced
    
    def train_model(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame = None,
        y_val: pd.Series = None
    ):
        """
        Train XGBoost model with class balancing
        
        Args:
            X_train: Training features
            y_train: Training target
            X_val: Validation features (optional)
            y_val: Validation target (optional)
        """
        print("\n" + "="*60)
        print("Training XGBoost Model with Class Balancing")
        print("="*60)
        
        # Apply class balancing
        X_train_balanced, y_train_balanced = self.apply_class_balancing(X_train, y_train)
        
        # Compute class weights for XGBoost
        if self.use_class_weights:
            self.class_weights = self.compute_class_weights(y_train)
        
        # Initialize model with class weights
        model_params = {
            'n_estimators': 400,
            'learning_rate': 0.05,
            'max_depth': 6,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'random_state': 42,
            'eval_metric': 'logloss',
            'use_label_encoder': False,
            'early_stopping_rounds': 50 if X_val is not None else None
        }
        
        # Add class weights if available
        if self.use_class_weights and self.class_weights:
            model_params['scale_pos_weight'] = self.class_weights[1] / self.class_weights[0]
            print(f"Using scale_pos_weight: {model_params['scale_pos_weight']:.3f}")
        
        self.model = XGBClassifier(**model_params)
        
        print("\nModel parameters:")
        for key, value in model_params.items():
            print(f"  {key}: {value}")
        
        # Train
        print("\nTraining model...")
        if X_val is not None and y_val is not None:
            self.model.fit(
                X_train_balanced, y_train_balanced,
                eval_set=[(X_val, y_val)],
                verbose=False
            )
        else:
            self.model.fit(X_train_balanced, y_train_balanced)
        
        print("Model training complete!")
        
        # Find optimal threshold on validation set if available
        if X_val is not None and y_val is not None:
            self.optimal_threshold = self.find_optimal_threshold(X_val, y_val)
        else:
            # Use training set for threshold optimization if no validation set
            self.optimal_threshold = self.find_optimal_threshold(X_train, y_train)
        
    def find_optimal_threshold(self, X: pd.DataFrame, y: pd.Series) -> float:
        """
        Find optimal threshold using precision-recall curve
        
        Args:
            X: Features
            y: True labels
            
        Returns:
            Optimal threshold
        """
        y_pred_proba = self.model.predict_proba(X)[:, 1]
        precision, recall, thresholds = precision_recall_curve(y, y_pred_proba)
        
        # Find threshold that maximizes F1 score
        f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)
        optimal_idx = np.argmax(f1_scores)
        
        if optimal_idx < len(thresholds):
            optimal_threshold = thresholds[optimal_idx]
        else:
            optimal_threshold = 0.5
            
        print(f"Optimal threshold: {optimal_threshold:.3f} (F1: {f1_scores[optimal_idx]:.3f})")
        
        return optimal_threshold
    
    def evaluate_model(self, X: pd.DataFrame, y: pd.Series, set_name: str = "Test") -> Dict:
        """
        Evaluate model on given dataset with comprehensive metrics
        
        Args:
            X: Features
            y: True labels
            set_name: Name of dataset (for display)
            
        Returns:
            Dictionary of metrics
        """
        print(f"\nEvaluating on {set_name} set...")
        
        # Predictions
        y_pred_proba = self.model.predict_proba(X)[:, 1]
        
        # Use optimal threshold if available, otherwise use 0.5
        threshold = getattr(self, 'optimal_threshold', 0.5)
        y_pred = (y_pred_proba >= threshold).astype(int)
        
        # Calculate comprehensive metrics
        metrics = {
            # Basic metrics
            'accuracy': accuracy_score(y, y_pred),
            'precision': precision_score(y, y_pred, zero_division=0),
            'recall': recall_score(y, y_pred, zero_division=0),
            'f1': f1_score(y, y_pred, zero_division=0),
            
            # Balanced metrics (important for imbalanced data)
            'balanced_accuracy': balanced_accuracy_score(y, y_pred),
            'matthews_corrcoef': matthews_corrcoef(y, y_pred),
            
            # Probability-based metrics
            'auc_roc': roc_auc_score(y, y_pred_proba),
            'auc_pr': average_precision_score(y, y_pred_proba),
            'log_loss': log_loss(y, y_pred_proba),
            
            # Threshold information
            'threshold_used': threshold,
            
            # Confusion matrix
            'confusion_matrix': confusion_matrix(y, y_pred).tolist(),
            
            # Class distribution
            'class_distribution': y.value_counts().to_dict(),
            'predicted_distribution': pd.Series(y_pred).value_counts().to_dict()
        }
        
        # Print results
        print(f"\n{set_name} Set Metrics:")
        print(f"  Basic Metrics:")
        print(f"    Accuracy: {metrics['accuracy']:.4f}")
        print(f"    Precision: {metrics['precision']:.4f}")
        print(f"    Recall: {metrics['recall']:.4f}")
        print(f"    F1 Score: {metrics['f1']:.4f}")
        
        print(f"  Balanced Metrics (Important for Imbalanced Data):")
        print(f"    Balanced Accuracy: {metrics['balanced_accuracy']:.4f}")
        print(f"    Matthews Correlation Coefficient: {metrics['matthews_corrcoef']:.4f}")
        
        print(f"  Probability-based Metrics:")
        print(f"    AUC-ROC: {metrics['auc_roc']:.4f}")
        print(f"    AUC-PR (Average Precision): {metrics['auc_pr']:.4f}")
        print(f"    Log Loss: {metrics['log_loss']:.4f}")
        
        print(f"  Threshold Used: {metrics['threshold_used']:.3f}")
        
        print(f"\nClass Distribution:")
        print(f"  Actual: {metrics['class_distribution']}")
        print(f"  Predicted: {metrics['predicted_distribution']}")
        
        print(f"\nConfusion Matrix:")
        cm = np.array(metrics['confusion_matrix'])
        print(f"  TN: {cm[0,0]}, FP: {cm[0,1]}")
        print(f"  FN: {cm[1,0]}, TP: {cm[1,1]}")
        
        return metrics
    
    def plot_confusion_matrix(self, cm: np.ndarray, save_path: str = "reports/confusion_matrix.png"):
        """
        Plot confusion matrix
        
        Args:
            cm: Confusion matrix
            save_path: Path to save figure
        """
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['No ADR', 'ADR'],
                   yticklabels=['No ADR', 'ADR'])
        plt.title('Confusion Matrix - ADR Prediction')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Confusion matrix saved to {save_path}")
    
    def plot_feature_importance(self, top_n: int = 20, save_path: str = "reports/feature_importance.png"):
        """
        Plot feature importance
        
        Args:
            top_n: Number of top features to show
            save_path: Path to save figure
        """
        importances = self.model.feature_importances_
        feature_importance = pd.DataFrame({
            'feature': self.feature_names,
            'importance': importances
        }).sort_values('importance', ascending=False)
        
        plt.figure(figsize=(10, 8))
        top_features = feature_importance.head(top_n)
        plt.barh(range(len(top_features)), top_features['importance'])
        plt.yticks(range(len(top_features)), top_features['feature'])
        plt.xlabel('Importance')
        plt.title(f'Top {top_n} Feature Importances')
        plt.gca().invert_yaxis()
        plt.tight_layout()
        
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Feature importance plot saved to {save_path}")
        
        return feature_importance
    
    def plot_roc_curve(self, X: pd.DataFrame, y: pd.Series, save_path: str = "reports/roc_curve.png"):
        """
        Plot ROC curve
        
        Args:
            X: Features
            y: True labels
            save_path: Path to save figure
        """
        y_pred_proba = self.model.predict_proba(X)[:, 1]
        fpr, tpr, thresholds = roc_curve(y, y_pred_proba)
        auc = roc_auc_score(y, y_pred_proba)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {auc:.3f})', linewidth=2)
        plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve - ADR Prediction')
        plt.legend()
        plt.grid(alpha=0.3)
        plt.tight_layout()
        
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"ROC curve saved to {save_path}")
    
    def save_trained_model(self, model_path: str = "models/xgb_adr_model.pkl"):
        """
        Save trained model and metadata
        
        Args:
            model_path: Path to save model
        """
        print(f"\nSaving model to {model_path}...")
        
        metadata = {
            'feature_names': self.feature_names,
            'n_features': len(self.feature_names),
            'metrics': self.metrics,
            'model_type': 'XGBClassifier',
            'train_date': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        save_model(self.model, metadata, model_path)
        
        print("Model and metadata saved successfully!")
    
    def full_training_pipeline(self):
        """
        Execute complete training pipeline
        """
        print("\n" + "="*60)
        print("AI-CPA MODEL TRAINING PIPELINE")
        print("="*60 + "\n")
        
        # Load data
        X, y, self.feature_names = self.load_or_preprocess_data()
        
        # Split data
        splits = self.split_data(X, y)
        
        # Train model with validation set for early stopping and threshold optimization
        self.train_model(
            splits['X_train'], splits['y_train'],
            splits['X_val'], splits['y_val']
        )
        
        # Evaluate on all sets
        print("\n" + "="*60)
        print("MODEL EVALUATION")
        print("="*60)
        
        train_metrics = self.evaluate_model(splits['X_train'], splits['y_train'], "Train")
        val_metrics = self.evaluate_model(splits['X_val'], splits['y_val'], "Validation")
        test_metrics = self.evaluate_model(splits['X_test'], splits['y_test'], "Test")
        
        self.metrics = {
            'train': train_metrics,
            'validation': val_metrics,
            'test': test_metrics
        }
        
        # Create visualizations
        print("\n" + "="*60)
        print("CREATING VISUALIZATIONS")
        print("="*60 + "\n")
        
        cm = np.array(test_metrics['confusion_matrix'])
        self.plot_confusion_matrix(cm)
        self.plot_feature_importance()
        self.plot_roc_curve(splits['X_test'], splits['y_test'])
        
        # Save model
        self.save_trained_model()
        
        # Print summary
        print("\n" + "="*60)
        print("TRAINING COMPLETE - SUMMARY")
        print("="*60)
        print(f"\nTest Set Performance (with Class Balancing):")
        print(f"  Basic Metrics:")
        print(f"    - Accuracy: {test_metrics['accuracy']:.4f}")
        print(f"    - Precision: {test_metrics['precision']:.4f}")
        print(f"    - Recall: {test_metrics['recall']:.4f}")
        print(f"    - F1 Score: {test_metrics['f1']:.4f}")
        
        print(f"  Balanced Metrics (Important for Imbalanced Data):")
        print(f"    - Balanced Accuracy: {test_metrics['balanced_accuracy']:.4f}")
        print(f"    - Matthews Correlation Coefficient: {test_metrics['matthews_corrcoef']:.4f}")
        
        print(f"  Probability-based Metrics:")
        print(f"    - AUC-ROC: {test_metrics['auc_roc']:.4f}")
        print(f"    - AUC-PR: {test_metrics['auc_pr']:.4f}")
        print(f"    - Log Loss: {test_metrics['log_loss']:.4f}")
        
        print(f"  Threshold Used: {test_metrics['threshold_used']:.3f}")
        
        print(f"\nClass Distribution:")
        print(f"  Actual: {test_metrics['class_distribution']}")
        print(f"  Predicted: {test_metrics['predicted_distribution']}")
        
        print(f"\nModel saved to: models/xgb_adr_model.pkl")
        print(f"Visualizations saved to: reports/")
        print(f"Optimal threshold: {self.optimal_threshold:.3f}")
        print("="*60 + "\n")
        
        return self.model, self.metrics


def main():
    """
    Main function to run training with class balancing
    """
    # Initialize trainer with class balancing enabled
    trainer = ADRModelTrainer(
        use_smote=True,      # Enable SMOTE oversampling
        use_class_weights=True  # Enable class weights in XGBoost
    )
    
    print("Training Configuration:")
    print(f"  - SMOTE Oversampling: {trainer.use_smote}")
    print(f"  - Class Weights: {trainer.use_class_weights}")
    print("  - Threshold Optimization: Enabled")
    print("  - Balanced Metrics: Enabled")
    
    model, metrics = trainer.full_training_pipeline()
    return model, metrics


if __name__ == "__main__":
    main()

