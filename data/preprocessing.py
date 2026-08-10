import pandas as pd
import numpy as np
import polars as pl
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm
try:
    from .utils import MemoryOptimizer
except ImportError:
    from utils import MemoryOptimizer


def load_and_clean_data(path, columns_to_drop):
    try:
        df = pl.read_csv(path)
        existing_cols_to_drop = [col for col in columns_to_drop if col in df.columns]
        if existing_cols_to_drop:
            df = df.drop(existing_cols_to_drop)
        result = df.to_pandas()
        del df  # Free memory
        MemoryOptimizer.cleanup_memory()
        return result
    except Exception as e:
        print(f"Error loading {path}: {e}")
        return None


def preprocess_ultra_fast(train_df, test_df, valid_df, target_col='class'):
    X_train = train_df.drop(columns=[target_col])
    y_train = train_df[target_col]
    
    X_test = test_df.drop(columns=[target_col], errors='ignore')
    y_test = test_df[target_col] if target_col in test_df.columns else None
    
    X_valid = valid_df.drop(columns=[target_col], errors='ignore')
    y_valid = valid_df[target_col] if target_col in valid_df.columns else None
    
    print(f"Raw data shapes - Train: {X_train.shape}, Test: {X_test.shape}, Valid: {X_valid.shape}")
    
    print("Encoding target labels...")
    label_encoder = LabelEncoder()
    y_train_encoded = label_encoder.fit_transform(y_train)
    
    if y_test is not None:
        y_test_encoded = label_encoder.transform(y_test)
    else:
        y_test_encoded = None
        
    if y_valid is not None:
        y_valid_encoded = label_encoder.transform(y_valid)
    else:
        y_valid_encoded = None
    
    # Identify categorical and numerical columns
    categorical_cols = X_train.select_dtypes(include=['object', 'category']).columns.tolist()
    numerical_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()
    
    print(f"Categorical columns ({len(categorical_cols)}): {categorical_cols}")
    print(f"Numerical columns ({len(numerical_cols)}): Found {len(numerical_cols)} numerical columns")
    
    print("Fast categorical encoding...")
    cat_encoders = {}
    
    if categorical_cols:
        with tqdm(total=len(categorical_cols), desc="Encoding categorical") as pbar:
            for col in categorical_cols:
                unique_vals = set()
                unique_vals.update(X_train[col].astype(str).unique())
                if col in X_test.columns:
                    unique_vals.update(X_test[col].astype(str).unique())
                if col in X_valid.columns:
                    unique_vals.update(X_valid[col].astype(str).unique())
                
                unique_list = sorted(list(unique_vals))
                mapping = {val: idx for idx, val in enumerate(unique_list)}
                X_train[col] = X_train[col].astype(str).map(mapping)
                if col in X_test.columns:
                    X_test[col] = X_test[col].astype(str).map(mapping)
                if col in X_valid.columns:
                    X_valid[col] = X_valid[col].astype(str).map(mapping)
                
                reverse_mapping = {idx: val for val, idx in mapping.items()}
                cat_encoders[col] = {'mapping': mapping, 'reverse_mapping': reverse_mapping}
                
                pbar.update(1)
    
    scaler = None

    preprocessors = {
        'feature_engineer': None,
        'label_encoder': label_encoder,
        'categorical_encoders': cat_encoders,
        'scaler': scaler,
        'categorical_cols': categorical_cols,
        'numerical_cols': numerical_cols
    }
    
    print(f"Final shapes - Train: {X_train.shape}, Test: {X_test.shape}, Valid: {X_valid.shape}")
    
    return (X_train, y_train_encoded, X_test, y_test_encoded, X_valid, y_valid_encoded, 
            categorical_cols, numerical_cols, preprocessors)
