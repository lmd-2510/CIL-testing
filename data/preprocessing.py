import pandas as pd
import numpy as np
import polars as pl
import os
import time
from datetime import datetime
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm
try:
    from .utils import MemoryOptimizer
except ImportError:
    from utils import MemoryOptimizer


def log_stage(message: str) -> None:
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {message}", flush=True)


def format_bytes(num_bytes: int) -> str:
    size = float(num_bytes)
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if size < 1024:
            return f"{size:.2f} {unit}"
        size /= 1024
    return f"{size:.2f} PB"


def load_and_clean_data(path, columns_to_drop):
    start_time = time.perf_counter()
    try:
        file_size = os.path.getsize(path) if os.path.exists(path) else 0
        log_stage(f"[LOAD] Start reading CSV: {path} | size={format_bytes(file_size)}")
        df = pl.read_csv(path)
        log_stage(f"[LOAD] Finished Polars read: {path} | shape={df.shape}")

        existing_cols_to_drop = [col for col in columns_to_drop if col in df.columns]
        if existing_cols_to_drop:
            log_stage(f"[LOAD] Dropping columns: {existing_cols_to_drop}")
            df = df.drop(existing_cols_to_drop)

        log_stage(f"[LOAD] Converting Polars -> Pandas: {path}")
        result = df.to_pandas()
        del df  # Free memory
        MemoryOptimizer.cleanup_memory()
        elapsed = time.perf_counter() - start_time
        log_stage(f"[LOAD] Done: {path} | pandas_shape={result.shape} | elapsed={elapsed:.1f}s")
        return result
    except Exception as e:
        elapsed = time.perf_counter() - start_time
        log_stage(f"[LOAD][ERROR] {path} | elapsed={elapsed:.1f}s | error={e}")
        return None


def _scale_numerical_columns(train_df, test_df, valid_df, numerical_cols, chunk_size=256):
    if not numerical_cols:
        return train_df, test_df, valid_df, None

    log_stage(f"[PREPROCESS] Scaling numerical columns in chunks | cols={len(numerical_cols)}")
    stats = {}
    for start in range(0, len(numerical_cols), chunk_size):
        cols = numerical_cols[start : start + chunk_size]
        train_values = train_df[cols].astype(np.float32)
        means = train_values.mean(axis=0)
        stds = train_values.std(axis=0).replace(0, 1.0).fillna(1.0)

        train_df.loc[:, cols] = ((train_values - means) / stds).astype(np.float32)
        if len(test_df) > 0:
            test_df.loc[:, cols] = ((test_df[cols].astype(np.float32) - means) / stds).astype(np.float32)
        if len(valid_df) > 0:
            valid_df.loc[:, cols] = ((valid_df[cols].astype(np.float32) - means) / stds).astype(np.float32)

        stats.update({
            col: {"mean": float(means[col]), "std": float(stds[col])}
            for col in cols
        })

    return train_df, test_df, valid_df, stats


def preprocess_ultra_fast(
    train_df,
    test_df,
    valid_df,
    target_col='class',
    scale_numerical: bool = True,
):
    start_time = time.perf_counter()
    log_stage("[PREPROCESS] Start global preprocessing")
    X_train = train_df.drop(columns=[target_col])
    y_train = train_df[target_col]
    
    X_test = test_df.drop(columns=[target_col], errors='ignore')
    y_test = test_df[target_col] if target_col in test_df.columns else None
    
    X_valid = valid_df.drop(columns=[target_col], errors='ignore')
    y_valid = valid_df[target_col] if target_col in valid_df.columns else None
    
    log_stage(f"[PREPROCESS] Raw feature shapes | train={X_train.shape} test={X_test.shape} valid={X_valid.shape}")
    
    log_stage("[PREPROCESS] Encoding target labels")
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
    
    log_stage(f"[PREPROCESS] Categorical columns ({len(categorical_cols)}): {categorical_cols}")
    log_stage(f"[PREPROCESS] Numerical columns: {len(numerical_cols)}")
    
    log_stage("[PREPROCESS] Fast categorical encoding")
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
    if scale_numerical:
        X_train, X_test, X_valid, scaler = _scale_numerical_columns(
            X_train,
            X_test,
            X_valid,
            numerical_cols,
        )
    elif numerical_cols:
        log_stage("[PREPROCESS] Numerical scaling disabled")

    preprocessors = {
        'feature_engineer': None,
        'label_encoder': label_encoder,
        'categorical_encoders': cat_encoders,
        'scaler': scaler,
        'categorical_cols': categorical_cols,
        'numerical_cols': numerical_cols
    }
    
    elapsed = time.perf_counter() - start_time
    log_stage(f"[PREPROCESS] Done | train={X_train.shape} test={X_test.shape} valid={X_valid.shape} | elapsed={elapsed:.1f}s")
    
    return (X_train, y_train_encoded, X_test, y_test_encoded, X_valid, y_valid_encoded, 
            categorical_cols, numerical_cols, preprocessors)
