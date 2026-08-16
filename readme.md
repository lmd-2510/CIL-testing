# DỰ ÁN: CIL-DDI BENCHMARK (Class Incremental Learning for Drug-Drug Interaction)

## 1. MỤC TIÊU DỰ ÁN
Chuyển đổi mô hình T-DDI v1.0.0 từ bài toán phân loại tĩnh sang kịch bản **Class Incremental Learning (CIL)**. Benchmark đánh giá 3 kiến trúc mô hình (`tddi`, `mlp_resnet`, `tabnet`) với 5 phương pháp Continual Learning (`finetune`, `ewc`, `gem`, `agem`, `er`).

---

## 2. CẤU TRÚC REPO HIỆN TẠI

```text
CIL-testing/
├── checkpoints/
├── results/
├── configs/
│   └── tddi.yaml
├── data/
│   ├── ddi_incremental.py
│   ├── preprocessing.py
│   └── material/
│       └── mock_cil_train.csv
├── methods/
│   ├── finetune.py
│   ├── ewc.py
│   ├── gem.py
│   ├── agem.py
│   └── er.py
├── models/
│   ├── tddi.py
│   ├── mlp_resnet.py
│   └── tabnet.py
├── PyBioMed/
├── utils.py
├── trainer.py
├── evaluator.py
└── main.py
```

