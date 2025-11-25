"""
Data preprocessing module for AI-CPA system
Merges MIMIC-IV and FAERS data for model training
"""
import pandas as pd
import numpy as np
from typing import Tuple, Dict, List
import os
import sys
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.utils import normalize_drug_name, clean_column_names, create_age_group, handle_missing_values


class MIMICFAERSPreprocessor:
    """
    Preprocessor to merge MIMIC-IV and FAERS datasets
    """
    
    def __init__(self, data_dir: str = None):
        """
        Initialize preprocessor
        
        Args:
            data_dir: Directory containing preprocessed CSV files
        """
        if data_dir is None:
            # Default to data/output relative to project root
            project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            data_dir = os.path.join(project_root, "data", "output")
        self.data_dir = data_dir
        self.label_encoders = {}
        self.feature_names = []
        
    def load_data(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Load all preprocessed data files
        
        Returns:
            Tuple of (patients, prescriptions, labs, faers)
        """
        print("Loading data files...")
        
        patients = pd.read_csv(os.path.join(self.data_dir, "mimic_patient_summary.csv"))
        prescriptions = pd.read_csv(os.path.join(self.data_dir, "mimic_prescriptions.csv"))
        labs = pd.read_csv(os.path.join(self.data_dir, "mimic_key_labs.csv"))
        faers = pd.read_csv(os.path.join(self.data_dir, "faers_drug_summary.csv"))
        
        print(f"Loaded {len(patients)} patients, {len(prescriptions)} prescriptions, "
              f"{len(labs)} lab results, {len(faers)} FAERS drug records")
        
        return patients, prescriptions, labs, faers
    
    def prepare_faers_lookup(self, faers: pd.DataFrame) -> Dict[str, Dict]:
        """
        Create drug lookup dictionary from FAERS data
        
        Args:
            faers: FAERS drug summary DataFrame
            
        Returns:
            Dictionary mapping drug names to FAERS metrics
        """
        faers_lookup = {}
        
        for _, row in faers.iterrows():
            drug_name = normalize_drug_name(row['drugname'])
            faers_lookup[drug_name] = {
                'adr_rate': row.get('ADR_Rate', 0),
                'severe_rate': row.get('Severe_Outcome_Rate', 0),
                'adr_count': row.get('ADR_Count', 0)
            }
        
        print(f"Created FAERS lookup with {len(faers_lookup)} drugs")
        return faers_lookup
    
    def aggregate_prescriptions(self, prescriptions: pd.DataFrame, faers_lookup: Dict) -> pd.DataFrame:
        """
        Aggregate prescription data per admission with FAERS signals
        
        Args:
            prescriptions: MIMIC prescriptions DataFrame
            faers_lookup: FAERS drug lookup dictionary
            
        Returns:
            Aggregated prescriptions per admission
        """
        print("Aggregating prescriptions...")
        
        # Normalize drug names
        prescriptions['drug_normalized'] = prescriptions['drug'].apply(normalize_drug_name)
        
        # Map FAERS signals
        prescriptions['adr_rate'] = prescriptions['drug_normalized'].apply(
            lambda x: faers_lookup.get(x, {}).get('adr_rate', 0)
        )
        prescriptions['severe_rate'] = prescriptions['drug_normalized'].apply(
            lambda x: faers_lookup.get(x, {}).get('severe_rate', 0)
        )
        
        # Flag high-risk drugs (ADR rate > 0.1)
        prescriptions['high_risk_drug'] = (prescriptions['adr_rate'] > 0.1).astype(int)
        
        # Aggregate by subject_id and hadm_id
        agg_dict = {
            'pharmacy_id': 'count',  # number of prescriptions
            'adr_rate': ['mean', 'max', 'std'],
            'severe_rate': ['mean', 'max'],
            'high_risk_drug': 'sum'
        }
        
        # Group by hadm_id if available, otherwise subject_id
        if 'hadm_id' in prescriptions.columns:
            grouped = prescriptions.groupby(['subject_id', 'hadm_id']).agg(agg_dict).reset_index()
            grouped.columns = ['subject_id', 'hadm_id', 'num_drugs', 'mean_adr_rate', 
                              'max_adr_rate', 'std_adr_rate', 'mean_severe_rate', 
                              'max_severe_rate', 'num_high_risk_drugs']
        else:
            grouped = prescriptions.groupby('subject_id').agg(agg_dict).reset_index()
            grouped.columns = ['subject_id', 'num_drugs', 'mean_adr_rate', 
                              'max_adr_rate', 'std_adr_rate', 'mean_severe_rate', 
                              'max_severe_rate', 'num_high_risk_drugs']
        
        # Fill NaN in std_adr_rate with 0
        grouped['std_adr_rate'].fillna(0, inplace=True)
        
        # Add polypharmacy flags
        grouped['polypharmacy_flag'] = (grouped['num_drugs'] >= 5).astype(int)
        grouped['major_polypharmacy_flag'] = (grouped['num_drugs'] >= 10).astype(int)
        
        print(f"Aggregated to {len(grouped)} patient admissions")
        return grouped
    
    def aggregate_labs(self, labs: pd.DataFrame) -> pd.DataFrame:
        """
        Aggregate lab results per admission (latest values)
        
        Args:
            labs: MIMIC lab events DataFrame
            
        Returns:
            Aggregated labs per admission
        """
        print("Aggregating lab results...")
        
        # Focus on key labs
        key_labs = ['Creatinine', 'Alanine Aminotransferase (ALT)', 
                   'Aspartate Aminotransferase (AST)', 'Hemoglobin', 
                   'White Blood Cells', 'Platelet Count']
        
        if 'lab_name' in labs.columns:
            labs_filtered = labs[labs['lab_name'].isin(key_labs)].copy()
        else:
            labs_filtered = labs.copy()
        
        # Get latest charttime per admission and lab
        if 'hadm_id' in labs_filtered.columns and 'charttime' in labs_filtered.columns:
            labs_filtered['charttime'] = pd.to_datetime(labs_filtered['charttime'])
            labs_sorted = labs_filtered.sort_values('charttime', ascending=False)
            labs_latest = labs_sorted.groupby(['subject_id', 'hadm_id', 'label']).first().reset_index()
            
            # Pivot to wide format
            labs_pivot = labs_latest.pivot_table(
                index=['subject_id', 'hadm_id'],
                columns='label',
                values='valuenum',
                aggfunc='first'
            ).reset_index()
            
            # Rename columns
            labs_pivot.columns.name = None
            col_mapping = {col: f"lab_{col.lower().replace(' ', '_')[:20]}" 
                          for col in labs_pivot.columns if col not in ['subject_id', 'hadm_id']}
            labs_pivot.rename(columns=col_mapping, inplace=True)
            
        else:
            # Fallback: aggregate by subject_id only
            labs_pivot = labs_filtered.groupby(['subject_id', 'label']).agg(
                {'valuenum': 'mean'}
            ).reset_index().pivot_table(
                index='subject_id',
                columns='label',
                values='valuenum'
            ).reset_index()
            
            labs_pivot.columns.name = None
            col_mapping = {col: f"lab_{col.lower().replace(' ', '_')[:20]}" 
                          for col in labs_pivot.columns if col != 'subject_id'}
            labs_pivot.rename(columns=col_mapping, inplace=True)
        
        print(f"Aggregated {len(labs_pivot)} lab records with {len(labs_pivot.columns)-1} lab features")
        return labs_pivot
    
    def merge_datasets(
        self, 
        patients: pd.DataFrame, 
        prescriptions_agg: pd.DataFrame, 
        labs_agg: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Merge all datasets into single training dataset
        
        Args:
            patients: Patient demographics
            prescriptions_agg: Aggregated prescriptions with FAERS
            labs_agg: Aggregated lab results
            
        Returns:
            Merged DataFrame
        """
        print("Merging datasets...")
        
        # Start with patients
        merged = patients.copy()
        
        # Determine merge keys based on what's available
        # Patient summary is at subject level, but prescriptions/labs may have hadm_id
        if 'hadm_id' in prescriptions_agg.columns and 'hadm_id' not in patients.columns:
            # Aggregate prescriptions to subject level
            print("  Aggregating prescriptions to subject level...")
            agg_dict = {col: 'mean' if col not in ['subject_id', 'hadm_id'] else 'first' 
                       for col in prescriptions_agg.columns if col not in ['subject_id', 'hadm_id']}
            prescriptions_agg = prescriptions_agg.groupby('subject_id').agg(agg_dict).reset_index()
        
        # Merge prescriptions
        merge_keys = ['subject_id']
        merged = merged.merge(prescriptions_agg, on=merge_keys, how='left')
        
        # Same for labs
        if 'hadm_id' in labs_agg.columns and 'hadm_id' not in patients.columns:
            print("  Aggregating labs to subject level...")
            agg_dict = {col: 'mean' if col not in ['subject_id', 'hadm_id'] else 'first' 
                       for col in labs_agg.columns if col not in ['subject_id', 'hadm_id']}
            labs_agg = labs_agg.groupby('subject_id').agg(agg_dict).reset_index()
        
        # Merge labs
        merge_keys_labs = ['subject_id']
        merged = merged.merge(labs_agg, on=merge_keys_labs, how='left')
        
        print(f"Merged dataset shape: {merged.shape}")
        return merged
    
    def create_target_variable(self, merged: pd.DataFrame) -> pd.DataFrame:
        """
        Create binary ADR target variable
        
        Args:
            merged: Merged dataset
            
        Returns:
            DataFrame with ADR_flag column
        """
        print("Creating target variable...")
        
        # Strategy: Use combination of factors to simulate ADR flag
        # In real scenario, this would come from actual adverse event data
        
        # Factors indicating higher ADR risk:
        # 1. High mean ADR rate from drugs
        # 2. High number of drugs (polypharmacy)
        # 3. Abnormal labs (if available)
        # 4. Older age
        
        adr_score = 0
        
        if 'mean_adr_rate' in merged.columns:
            adr_score += (merged['mean_adr_rate'] > 0.1).astype(int) * 3
            adr_score += (merged['max_adr_rate'] > 0.15).astype(int) * 2
        
        if 'num_drugs' in merged.columns:
            adr_score += (merged['num_drugs'] >= 5).astype(int) * 2
            adr_score += (merged['num_drugs'] >= 10).astype(int) * 2
        
        if 'anchor_age' in merged.columns:
            adr_score += (merged['anchor_age'] > 65).astype(int) * 1
        
        # Check for abnormal labs (simplified)
        lab_cols = [col for col in merged.columns if col.startswith('lab_')]
        if lab_cols:
            # Count missing or extreme values
            for col in lab_cols[:3]:  # Check first 3 lab columns
                if merged[col].notna().sum() > 0:
                    q99 = merged[col].quantile(0.99)
                    q01 = merged[col].quantile(0.01)
                    adr_score += ((merged[col] > q99) | (merged[col] < q01)).astype(int)
        
        # Create binary flag: ADR if score >= 4
        merged['ADR_flag'] = (adr_score >= 4).astype(int)
        
        # Add some randomness to make it more realistic (10% noise)
        np.random.seed(42)
        noise_mask = np.random.random(len(merged)) < 0.1
        merged.loc[noise_mask, 'ADR_flag'] = 1 - merged.loc[noise_mask, 'ADR_flag']
        
        print(f"ADR distribution: {merged['ADR_flag'].value_counts().to_dict()}")
        print(f"ADR rate: {merged['ADR_flag'].mean():.2%}")
        
        return merged
    
    def prepare_features(self, merged: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series, List[str]]:
        """
        Prepare features for model training
        
        Args:
            merged: Merged dataset with target
            
        Returns:
            Tuple of (X, y, feature_names)
        """
        print("Preparing features...")
        
        # Select feature columns
        exclude_cols = ['subject_id', 'hadm_id', 'ADR_flag', 'dod', 'anchor_year', 
                       'anchor_year_group', 'pharmacy_id', 'poe_id', 'order_provider_id']
        
        feature_cols = [col for col in merged.columns if col not in exclude_cols]
        
        # Separate target
        y = merged['ADR_flag'].copy()
        X = merged[feature_cols].copy()
        
        # Encode categorical variables
        categorical_cols = X.select_dtypes(include=['object']).columns
        
        for col in categorical_cols:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))
            self.label_encoders[col] = le
        
        # Handle missing values
        X = handle_missing_values(X, strategy='median')
        
        # Store feature names
        self.feature_names = X.columns.tolist()
        
        print(f"Final feature set: {X.shape[1]} features, {len(y)} samples")
        print(f"Feature names: {self.feature_names[:10]}... (showing first 10)")
        
        return X, y, self.feature_names
    
    def process_full_pipeline(self) -> Tuple[pd.DataFrame, pd.Series, List[str], pd.DataFrame]:
        """
        Execute full preprocessing pipeline
        
        Returns:
            Tuple of (X, y, feature_names, merged_df)
        """
        print("\n" + "="*60)
        print("Starting MIMIC-FAERS preprocessing pipeline")
        print("="*60 + "\n")
        
        # Load data
        patients, prescriptions, labs, faers = self.load_data()
        
        # Create FAERS lookup
        faers_lookup = self.prepare_faers_lookup(faers)
        
        # Aggregate prescriptions with FAERS signals
        prescriptions_agg = self.aggregate_prescriptions(prescriptions, faers_lookup)
        
        # Aggregate labs
        labs_agg = self.aggregate_labs(labs)
        
        # Merge all datasets
        merged = self.merge_datasets(patients, prescriptions_agg, labs_agg)
        
        # Create target variable
        merged = self.create_target_variable(merged)
        
        # Prepare features
        X, y, feature_names = self.prepare_features(merged)
        
        print("\n" + "="*60)
        print("Preprocessing complete!")
        print("="*60 + "\n")
        
        return X, y, feature_names, merged


def main():
    """
    Main function to run preprocessing
    """
    preprocessor = MIMICFAERSPreprocessor()
    X, y, feature_names, merged_df = preprocessor.process_full_pipeline()
    
    # Save preprocessed data
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    output_dir = os.path.join(project_root, "data", "output")
    os.makedirs(output_dir, exist_ok=True)
    
    X.to_csv(os.path.join(output_dir, "X_features.csv"), index=False)
    y.to_csv(os.path.join(output_dir, "y_target.csv"), index=False)
    merged_df.to_csv(os.path.join(output_dir, "merged_dataset.csv"), index=False)
    
    print(f"\nPreprocessed data saved to {output_dir}/")
    print(f"- X_features.csv: {X.shape}")
    print(f"- y_target.csv: {y.shape}")
    print(f"- merged_dataset.csv: {merged_df.shape}")
    
    return X, y, feature_names


if __name__ == "__main__":
    main()

