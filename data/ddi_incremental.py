import os
import torch
import pandas as pd
import numpy as np
import time
from datetime import datetime
from typing import Tuple, List, Dict, Optional
from torch.utils.data import Dataset, DataLoader

# Import hàm tiền xử lý từ preprocessing.py
try:
    from .preprocessing import load_and_clean_data, preprocess_ultra_fast
except ImportError:
    from preprocessing import load_and_clean_data, preprocess_ultra_fast


def log_stage(message: str) -> None:
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {message}", flush=True)


class DDIDataset(Dataset):
    """PyTorch Dataset chuẩn hóa dữ liệu DDI dạng Bảng (Tabular) cho kịch bản CIL.

    Tự động phân tách và ép kiểu Tensor cho biến Categorical và Continuous.
    """

    def __init__(
        self,
        X_df: pd.DataFrame,
        y_array: np.ndarray,
        categorical_cols: List[str],
        numerical_cols: List[str],
        indices: Optional[np.ndarray] = None,
    ):
        # 1. Lọc theo chỉ số dòng (indices) nếu được truyền vào (kể cả khi indices rỗng len=0)
        if indices is not None:
            X_sub = X_df.iloc[indices]
            y_sub = y_array[indices]
        else:
            X_sub = X_df
            y_sub = y_array

        # 2. Chuyển biến Categorical thành LongTensor (torch.long)
        if len(X_sub) > 0 and len(categorical_cols) > 0:
            cat_data = X_sub[categorical_cols].values.astype(np.int64)
            self.cat_x = torch.tensor(cat_data, dtype=torch.long)
        else:
            self.cat_x = torch.empty((len(X_sub), len(categorical_cols)), dtype=torch.long)

        # 3. Chuyển biến Continuous thành FloatTensor (torch.float32)
        if len(X_sub) > 0 and len(numerical_cols) > 0:
            cont_data = X_sub[numerical_cols].values.astype(np.float32)
            self.cont_x = torch.tensor(cont_data, dtype=torch.float32)
        else:
            self.cont_x = torch.empty((len(X_sub), len(numerical_cols)), dtype=torch.float32)

        # 4. Chuyển nhãn target thành LongTensor (torch.long)
        if len(X_sub) > 0:
            self.y = torch.tensor(y_sub, dtype=torch.long)
        else:
            self.y = torch.empty((0,), dtype=torch.long)

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Trả về (cat_features, cont_features, target) phù hợp cho TabTransformer
        forward(cat_x, cont_x).
        """
        return self.cat_x[idx], self.cont_x[idx], self.y[idx]


class DDIIncrementalDataManager:
    """Quản lý dữ liệu động cho bài toán Class Incremental Learning (CIL) trên DDI Dataset."""

    def __init__(
        self,
        data_dir: str = "data/material",
        train_filename: str = "converted_ddi_types.csv",
        test_filename: Optional[str] = None,
        valid_filename: Optional[str] = None,
        columns_to_drop: Optional[List[str]] = None,
        target_col: str = "class",
        seed: int = 42,
    ):
        self.data_dir = data_dir
        self.train_filename = train_filename
        self.test_filename = test_filename
        self.valid_filename = valid_filename
        self.columns_to_drop = columns_to_drop or []
        self.target_col = target_col
        self.seed = seed

        # Metadata toàn cục (dùng cho TabTransformer và các models khác)
        self.categories: List[int] = []  # Số lượng giá trị duy nhất của từng cột categorical
        self.num_continuous: int = 0     # Số lượng cột numerical
        self.num_classes: int = 0        # Tổng số lớp DDI trong toàn bộ dataset
        self.categorical_cols: List[str] = []
        self.numerical_cols: List[str] = []

        # Dữ liệu dạng DataFrame / Numpy sau khi tiền xử lý toàn cục
        self.X_train: pd.DataFrame = None
        self.y_train: np.ndarray = None
        self.X_test: pd.DataFrame = None
        self.y_test: np.ndarray = None
        self.X_valid: pd.DataFrame = None
        self.y_valid: np.ndarray = None
        self.preprocessors: Dict = {}

    def prepare_global_data(self):
        """BUỚC 1: Nạp dữ liệu thô và tiến hành mã hóa toàn cục (Global Preprocessing).

        Trích xuất các thông số hạ tầng (categories, num_continuous, total
        classes).
        """
        print("=== BƯỚC 1: NẠP VÀ TIỀN XỬ LÝ DỮ LIỆU TOÀN CỤC ===")

        # 1. Đường dẫn file
        step_start = time.perf_counter()
        log_stage("[DATA] Step 1 start: load and preprocess global data")
        train_path = os.path.join(self.data_dir, self.train_filename)
        log_stage(f"[DATA] Loading train file: {train_path}")

        # 2. Đọc file thô bằng Polars (thông qua preprocessing.py)
        print(f"Đang đọc dữ liệu train từ: {train_path}")
        train_df = load_and_clean_data(train_path, self.columns_to_drop)

        if train_df is None:
            raise FileNotFoundError(f"Không thể đọc file dữ liệu tại {train_path}")

        # Tách tập test/valid từ train nếu không cung cấp file riêng
        if self.test_filename and os.path.exists(os.path.join(self.data_dir, self.test_filename)):
            log_stage(f"[DATA] Loading test file: {os.path.join(self.data_dir, self.test_filename)}")
            test_df = load_and_clean_data(
                os.path.join(self.data_dir, self.test_filename), self.columns_to_drop
            )
        else:
            log_stage("[DATA] No test file found/provided; using empty test dataframe")
            test_df = pd.DataFrame(columns=train_df.columns)

        if self.valid_filename and os.path.exists(os.path.join(self.data_dir, self.valid_filename)):
            log_stage(f"[DATA] Loading valid file: {os.path.join(self.data_dir, self.valid_filename)}")
            valid_df = load_and_clean_data(
                os.path.join(self.data_dir, self.valid_filename), self.columns_to_drop
            )
        else:
            log_stage("[DATA] No valid file found/provided; using empty valid dataframe")
            valid_df = pd.DataFrame(columns=train_df.columns)

        # 3. Tiến hành mã hóa toàn cục bằng preprocess_ultra_fast
        log_stage("[DATA] Starting global preprocessing")
        (
            self.X_train,
            self.y_train,
            self.X_test,
            self.y_test,
            self.X_valid,
            self.y_valid,
            self.categorical_cols,
            self.numerical_cols,
            self.preprocessors,
        ) = preprocess_ultra_fast(
            train_df, test_df, valid_df, target_col=self.target_col
        )
        log_stage("[DATA] Global preprocessing finished")

        # 4. Trích xuất metadata toàn cục
        self.num_continuous = len(self.numerical_cols)
        self.num_classes = len(np.unique(self.y_train))

        # Tính toán cardinality (số giá trị duy nhất + 1) cho từng thuộc tính categorical
        # TabTransformer yêu cầu tuple kích thước của từng biến categorical
        cat_encoders = self.preprocessors.get("categorical_encoders", {})
        self.categories = []
        for col in self.categorical_cols:
            if col in cat_encoders:
                # Kích thước bảng tra cứu embedding = số lượng unique values
                cat_size = len(cat_encoders[col]["mapping"])
                self.categories.append(cat_size)
            else:
                self.categories.append(1)

        print(f"\n[HOÀN THÀNH BƯỚC 1]")
        print(f" - Tổng số biến Categorical: {len(self.categorical_cols)}")
        print(f" - Bảng kích thước Categories (cho TabTransformer): {self.categories}")
        print(f" - Số biến Continuous: {self.num_continuous}")
        print(f" - Tổng số Lớp DDI (num_classes): {self.num_classes}")
        print(f" - Kích thước tập Train: {self.X_train.shape}")
        elapsed = time.perf_counter() - step_start
        log_stage(f"[DATA] Step 1 done | elapsed={elapsed:.1f}s")

    def split_tasks(
            self,
            num_tasks: int = 10,
            classes_per_task: Optional[int] = None,
            shuffle_classes: bool = False,
            class_order: str = "balanced",
        ) -> Dict[int, List[int]]:
        """BƯỚC 2: Phân chia tổng số lớp DDI thành các Task tăng dần (CIL Tasks).

        Args:
            num_tasks: Số lượng Task muốn chia (mặc định 5).
            classes_per_task: Số lớp mỗi Task (nếu truyền vào thì ưu tiên hơn
            num_tasks).
            shuffle_classes: Có xáo trộn thứ tự lớp trước khi chia hay không.

        Returns:
            Dict[int, List[int]]: Dictionary ánh xạ task_id -> danh sách các
            class nhãn thuộc task đó.
        """
        step_start = time.perf_counter()
        log_stage(f"[TASK] Step 2 start: split {self.num_classes} classes into CIL tasks")
        print(f"\n=== STEP 2: SPLIT {self.num_classes} CLASSES INTO CIL TASKS ===")

        if self.num_classes == 0 or self.y_train is None:
            raise ValueError("Chưa chạy prepare_global_data() hoặc dữ liệu rỗng. Hãy chạy Bước 1 trước.")

        all_classes = np.unique(self.y_train)
        total_cls = len(all_classes)

        # 1. Xáo trộn thứ tự các class nếu cần
        if shuffle_classes:
            class_order = "random"
        class_order = class_order.lower().strip()
        if class_order not in {"balanced", "random", "ordered"}:
            raise ValueError("class_order must be one of: balanced, random, ordered.")

        # 2. Xử lý số lớp mỗi Task
        if classes_per_task is not None and classes_per_task > 0:
            self.num_tasks = int(np.ceil(total_cls / classes_per_task))
        else:
            self.num_tasks = num_tasks

        # 3. Chia danh sách các class thành từng Task
        self.task_classes: Dict[int, List[int]] = {}
        
        # Cắt nhỏ mảng class thành num_tasks phần
        if class_order == "balanced":
            class_counts = {
                int(cls): int(np.sum(self.y_train == cls))
                for cls in all_classes
            }
            sorted_classes = sorted(
                class_counts.keys(),
                key=lambda cls: (-class_counts[cls], cls),
            )
            task_loads = [0 for _ in range(self.num_tasks)]
            task_class_lists = [[] for _ in range(self.num_tasks)]

            for cls in sorted_classes:
                task_id = int(np.argmin(task_loads))
                task_class_lists[task_id].append(cls)
                task_loads[task_id] += class_counts[cls]

            self.task_classes = {
                task_id: sorted(cls_list)
                for task_id, cls_list in enumerate(task_class_lists)
                if len(cls_list) > 0
            }
            self.num_tasks = len(self.task_classes)
        else:
            ordered_classes = all_classes.copy()
            if class_order == "random":
                rng = np.random.default_rng(self.seed)
                rng.shuffle(ordered_classes)

            splits = np.array_split(ordered_classes, self.num_tasks)
            self.task_classes = {
                task_id: cls_list.tolist()
                for task_id, cls_list in enumerate(splits)
                if len(cls_list) > 0
            }
            self.num_tasks = len(self.task_classes)

        # 4. Tìm vị trí (row indices) của từng Task trong tập Train, Test, Valid
        self.train_task_indices: Dict[int, np.ndarray] = {}
        self.test_task_indices: Dict[int, np.ndarray] = {}
        self.valid_task_indices: Dict[int, np.ndarray] = {}

        for task_id, cls_list in self.task_classes.items():
            # Lấy chỉ số các dòng trong y_train thuộc danh sách class của task_id
            self.train_task_indices[task_id] = np.where(np.isin(self.y_train, cls_list))[0]

            if self.y_test is not None and len(self.y_test) > 0:
                self.test_task_indices[task_id] = np.where(np.isin(self.y_test, cls_list))[0]
            else:
                self.test_task_indices[task_id] = np.array([], dtype=int)

            if self.y_valid is not None and len(self.y_valid) > 0:
                self.valid_task_indices[task_id] = np.where(np.isin(self.y_valid, cls_list))[0]
            else:
                self.valid_task_indices[task_id] = np.array([], dtype=int)

        # In thông tin tổng quan sau khi chia Task
        elapsed = time.perf_counter() - step_start
        print(f"-> Created {self.num_tasks} tasks | class_order={class_order} | elapsed={elapsed:.1f}s:")
        for task_id, cls_list in self.task_classes.items():
            num_train = len(self.train_task_indices[task_id])
            num_test = len(self.test_task_indices[task_id])
            print(
                f"  * Task {task_id}: {len(cls_list)} classes "
                f"(Classes: {cls_list[:3]}...{cls_list[-1:] if len(cls_list)>3 else ''}) "
                f"| Train samples: {num_train} | Test samples: {num_test}"
            )

        return self.task_classes

    def get_train_loader(
        self, task_id: int, batch_size: int = 256, shuffle: bool = True, num_workers: int = 0
    ) -> DataLoader:
        """Tạo DataLoader huấn luyện cho Task hiện tại (task_id)."""
        if task_id not in self.train_task_indices:
            raise ValueError(f"Task ID {task_id} không tồn tại. Hãy gọi split_tasks() trước.")

        indices = self.train_task_indices[task_id]
        dataset = DDIDataset(
            X_df=self.X_train,
            y_array=self.y_train,
            categorical_cols=self.categorical_cols,
            numerical_cols=self.numerical_cols,
            indices=indices,
        )

        return DataLoader(
            dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers
        )

    def get_test_loader(
        self, task_id: int, batch_size: int = 256, num_workers: int = 0
    ) -> DataLoader:
        """Tạo DataLoader đánh giá trên duy nhất Task {task_id}."""
        if task_id not in self.test_task_indices:
            raise ValueError(f"Task ID {task_id} không tồn tại.")

        indices = self.test_task_indices[task_id]
        dataset = DDIDataset(
            X_df=self.X_test if len(self.X_test) > 0 else self.X_train,
            y_array=self.y_test if len(self.y_test) > 0 else self.y_train,
            categorical_cols=self.categorical_cols,
            numerical_cols=self.numerical_cols,
            indices=indices,
        )

        return DataLoader(
            dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
        )

    def get_cumulative_test_loader(
        self, task_id: int, batch_size: int = 256, num_workers: int = 0
    ) -> DataLoader:
        """Tạo DataLoader đánh giá tích lũy trên TẤT CẢ các Task đã học từ Task 0
        đến Task {task_id}.

        Rất quan trọng cho việc tính ma trận R_{i,j} và Forgetting Index trong
        CIL.
        """
        cumulative_indices = []
        for t in range(task_id + 1):
            if t in self.test_task_indices:
                cumulative_indices.append(self.test_task_indices[t])

        if len(cumulative_indices) == 0:
            raise ValueError(f"Không có dữ liệu test cho các task từ 0 đến {task_id}")

        all_indices = np.concatenate(cumulative_indices)

        dataset = DDIDataset(
            X_df=self.X_test if len(self.X_test) > 0 else self.X_train,
            y_array=self.y_test if len(self.y_test) > 0 else self.y_train,
            categorical_cols=self.categorical_cols,
            numerical_cols=self.numerical_cols,
            indices=all_indices,
        )

        return DataLoader(
            dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
        )

    def get_memory_samples(
        self, task_id: int, samples_per_class: int = 10
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Lấy ngẫu nhiên các mẫu bộ nhớ đệm (Memory Buffer) từ các Task cũ (< task_id)."""
        if task_id == 0:
            # Task đầu tiên chưa có bộ nhớ cũ -> Trả về Tensor rỗng
            num_cat = len(self.categorical_cols)
            num_cont = len(self.numerical_cols)
            return (
                torch.empty((0, num_cat), dtype=torch.long),
                torch.empty((0, num_cont), dtype=torch.float32),
                torch.empty((0,), dtype=torch.long)
            )

        memory_indices = []
        for t in range(task_id):
            t_indices = self.train_task_indices[t]
            t_y = self.y_train[t_indices]
            unique_classes = np.unique(t_y)

            for cls in unique_classes:
                cls_indices = t_indices[t_y == cls]
                selected = np.random.choice(
                    cls_indices,
                    size=min(samples_per_class, len(cls_indices)),
                    replace=False,
                )
                memory_indices.extend(selected)

        memory_indices = np.array(memory_indices)
        dataset = DDIDataset(
            X_df=self.X_train,
            y_array=self.y_train,
            categorical_cols=self.categorical_cols,
            numerical_cols=self.numerical_cols,
            indices=memory_indices,
        )

        return dataset.cat_x, dataset.cont_x, dataset.y
