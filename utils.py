import os
import gc
import json
import random
import warnings
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, classification_report
from typing import Dict, List, Optional, Tuple

warnings.filterwarnings('ignore')

# ==========================================
# 1. QUẢN LÝ BỘ NHỚ & SEED 
# ==========================================

class MemoryOptimizer:
    """Quản lý và dọn dẹp bộ nhớ GPU/RAM giữa các epoch và task."""
    @staticmethod
    def cleanup_memory():
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    @staticmethod
    def get_memory_usage():
        if torch.cuda.is_available():
            return {
                'gpu_allocated': torch.cuda.memory_allocated() / 1024**3,
                'gpu_reserved': torch.cuda.memory_reserved() / 1024**3,
                'gpu_max_allocated': torch.cuda.max_memory_allocated() / 1024**3
            }
        return {'gpu_allocated': 0, 'gpu_reserved': 0, 'gpu_max_allocated': 0}


def set_seed(seed: int = 42) -> None:
    """Cố định seed ngẫu nhiên cho reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    print(f"[Utils] Fixed random seed: {seed}")


def get_device() -> torch.device:
    """Trả về thiết bị CUDA GPU hoặc CPU."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[Utils] Using device: {device}")
    return device

# ==========================================
# 2. ĐỌC CẤU HÌNH & HELPER LOGGING
# ==========================================

def load_config_defaults(config_path: Optional[str]) -> Dict[str, object]:
    """Đọc file cấu hình YAML."""
    if not config_path or not os.path.exists(config_path):
        return {}
    try:
        import yaml
        with open(config_path, "r", encoding="utf-8") as f:
            payload = yaml.safe_load(f) or {}
        return payload
    except Exception as e:
        print(f"[Utils] Warning loading config: {e}")
        return {}


def json_safe(obj):
    """Chuyển đổi kiểu dữ liệu NumPy/Pandas về chuẩn JSON Python."""
    if isinstance(obj, dict):
        return {str(k): json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [json_safe(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return None if np.isnan(obj) else float(obj)
    if isinstance(obj, float) and np.isnan(obj):
        return None
    return obj

# ==========================================
# 3. METRICS ĐÁNH GIÁ 
# ==========================================

def classification_metrics(y_true, y_pred) -> Dict[str, float]:
    """Tính toán bộ chỉ số Accuracy, Macro/Weighted F1, Precision, Recall."""
    if len(y_true) == 0:
        return {
            "accuracy": np.nan,
            "macro_precision": np.nan,
            "macro_recall": np.nan,
            "macro_f1": np.nan,
            "weighted_f1": np.nan,
        }

    report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    accuracy = accuracy_score(y_true, y_pred)
    return {
        "accuracy": accuracy,
        "macro_precision": report["macro avg"]["precision"],
        "macro_recall": report["macro avg"]["recall"],
        "macro_f1": report["macro avg"]["f1-score"],
        "weighted_f1": report["weighted avg"]["f1-score"],
    }

# ==========================================
# 4. KHAI BÁO UNCERTAINTY 
# ==========================================

class UncertaintyEstimator:
    """Tính toán Entropy & Uncertainty cho dự đoán."""
    @staticmethod
    def entropy_uncertainty(probabilities: np.ndarray) -> np.ndarray:
        epsilon = 1e-8
        probabilities = np.clip(probabilities, epsilon, 1 - epsilon)
        return -np.sum(probabilities * np.log(probabilities), axis=1)

# ==========================================
# 5. LƯU & LOAD CHECKPOINTS CIL
# ==========================================

def save_checkpoint(model, optimizer, task_id, epoch, filepath):
    """Lưu trọng số mô hình sau từng task."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    state = {
        'task_id': task_id,
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict() if optimizer else None,
    }
    torch.save(state, filepath)
    print(f"[Utils] Checkpoint saved: {filepath}")