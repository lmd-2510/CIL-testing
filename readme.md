# DỰ ÁN: CIL-DDI BENCHMARK (Class Incremental Learning for Drug-Drug Interaction)

## 1. MỤC TIÊU DỰ ÁN
Chuyển đổi mô hình T-DDI v1.0.0 từ bài toán phân loại tĩnh sang kịch bản **Class Incremental Learning (CIL)**. Benchmark đánh giá 3 kiến trúc mô hình (`tddi`, `mlp_resnet`, `tabnet`) với 5 phương pháp Continual Learning (`finetune`, `ewc`, `gem`, `agem`, `er`).

---

## 2. CHECKLIST CÔNG VIỆC
- [x] 1. Đọc và hiểu kiến trúc T-DDI v1.0.0, dữ liệu và pipeline.
- [x] 2. Chuyển DDI dataset thành CIL scenario (`data/ddi_incremental.py`).
- [x] 3. Tái thiết lập framework CIL dùng chung.
- [x] 4. Chọn thêm 2 model architecture: `mlp_resnet.py`, `tabnet.py`.
- [x] 5. Chuẩn hóa interface để 3 models dùng chung data/trainer/method/evaluator.
- [x] 6. Implement Naive Fine-tuning (`methods/finetune.py`).
- [x] 7. Implement EWC (`methods/ewc.py`).
- [x] 8. Implement GEM (`methods/gem.py`).
- [x] 9. Implement A-GEM (`methods/agem.py`).
- [x] 10. Implement ER / Experience Replay (`methods/er.py`).
- [x] 11. Smoke test end-to-end trên CPU với mock data.
- [ ] 12. Chuẩn bị dataset thật dạng sample-level.
- [ ] 13. Train từng tổ hợp `Model × CL method`.
- [ ] 14. Evaluate sau từng task.
- [ ] 15. Báo Accuracy/Macro-F1, ma trận $R_{i,j}$ và Forgetting Index.
- [ ] 16. Tạo bảng comparison 3 Architecture × 5 CL baselines.

---

## 2.1 TIẾN ĐỘ HIỆN TẠI
- Framework CIL lõi đã chạy được: `main.py`, `trainer.py`, `evaluator.py`, `data/ddi_incremental.py`.
- Đã có 5 methods: `finetune`, `ewc`, `gem`, `agem`, `er`.
- Đã có đủ 3 models: `tddi`, `mlp_resnet`, `tabnet`.
- Smoke test CPU đã pass với mock data nhỏ 16 dòng, 4 class, 2 tasks.
- Kết quả smoke test đã được xóa khỏi `results/`; mock data vẫn giữ lại để test nhanh lại khi cần.
- Dataset thật để train/evaluate benchmark đầy đủ vẫn chưa được chốt.

## 2.2 VIỆC CẦN LÀM TIẾP THEO
1. Chuẩn bị dataset thật dạng sample-level, có cột `class`, feature categorical/numerical.
2. Tạo train/test/valid thật để evaluator có metric có ý nghĩa.
3. Chạy thử từng model với `finetune` trước.
4. Sau đó chạy đủ 15 tổ hợp `3 models × 5 methods`.
5. Tổng hợp Accuracy, Macro-F1, ma trận `R_{i,j}`, Forgetting Index.

---

## 3. CẤU TRÚC REPO HIỆN TẠI

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

## 4. SMOKE TEST NHẸ TRÊN CPU
Mock data dùng cho smoke test:

```bash
python main.py --mode train --model mlp_resnet --method finetune --train-file mock_cil_train.csv --num-tasks 2 --batch-size 4 --run-name smoke_mlp_train
```

Các tổ hợp đã kiểm tra pass:
- `mlp_resnet + finetune`
- `tabnet + finetune`
- `mlp_resnet + ewc`
- `mlp_resnet + gem`
- `mlp_resnet + agem`

Lưu ý: smoke test chỉ kiểm tra luồng code, chưa phải benchmark thật.
