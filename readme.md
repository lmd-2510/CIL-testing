# DỰ ÁN: CIL-DDI BENCHMARK (Class Incremental Learning for Drug-Drug Interaction)

## 1. MỤC TIÊU DỰ ÁN
Chuyển đổi mô hình T-DDI v1.0.0 (Tabular Drug-Drug Interaction) từ bài toán phân loại tĩnh thành kịch bản **Class Incremental Learning (CIL)**. Thí nghiệm đánh giá 3 kiến trúc mô hình (T-DDI + 2 models mới) kết hợp với 4 phương pháp Continual Learning (Fine-tuning, EWC, GEM, A-GEM).

---

## 2. CHECKLIST CÔNG VIỆC THỰC HIỆN
- [x] 1. Đọc và hiểu kiến trúc T-DDI v1.0.0, dữ liệu và pipeline.
- [x] 2. Chuyển DDI dataset thành CIL scenario (`data/ddi_incremental.py`).
- [x] 3. Tái thiết lập cấu trúc repo để sẵn sàng adapt T-DDI thành CIL-compatible model.
- [ ] 4. Survey và chọn thêm 2 model architecture (`mlp_resnet.py`, `model3.py`).
- [ ] 5. Chuẩn hóa interface thư mục/mô-đun để 3 models dùng chung data/framework.
- [x] 6. Implement Naive Fine-tuning baseline (`methods/finetune.py`).
- [x] 7. Implement EWC (`methods/ewc.py`).
- [x] 8. Implement GEM (`methods/gem.py`).
- [x] 9. Implement A-GEM (`methods/agem.py`).
- [ ] 10. Train từng tổ hợp `Model × CL method`.
- [ ] 11. Evaluate sau từng task.
- [ ] 12. Báo Accuracy/Macro-F1, Ma trận $R_{i,j}$ và Forgetting Index.
- [ ] 13. Tạo bảng comparison 3 Architecture × 4 CL baselines.

---

## 2.1 TIẾN ĐỘ HIỆN TẠI
- Khung framework CIL lõi đã có: `main.py`, `trainer.py`, `evaluator.py`, `methods/finetune.py`, `methods/ewc.py`, `methods/gem.py`, `methods/agem.py`, `models/tddi.py`.
- `trainer.py` đã dùng `evaluator.py` để đánh giá `seen tasks`, `cumulative tasks`, ma trận `R_{i,j}` và forgetting.
- `trainer.py` không còn fallback `NaiveFinetuneMethod`; mọi method phải có `build_method()`.
- `models/tddi.py` đã có `build_model_for_cil(...)` để nối trực tiếp với trainer.
- Phần dataset thật để train/evaluate đầy đủ vẫn có thể để sau; hiện ưu tiên tiếp tục hoàn thiện benchmark bằng cách bổ sung thêm model.

## 2.2 VIỆC CẦN LÀM TIẾP THEO
1. Hoàn thiện `models/model3.py` với `build_model_for_cil(...)`.
2. Chạy smoke test end-to-end bằng dữ liệu nhỏ hoặc mock data:
   `main.py -> trainer.py -> method -> model -> evaluator.py`
3. Nếu smoke test ổn, cập nhật lại `trainer.py` hoặc config nếu cần để support nhiều lựa chọn model/method hơn.
4. Khi framework ổn định, mới chuyển sang dataset thật và train benchmark đầy đủ.

---

## 3. CẤU TRÚC REPO HIỆN TẠI (`CIL-testing`)

```text
CIL-testing/
├── checkpoints/                    # Lưu trọng số mô hình (.pth)
├── results/                        # Lưu ma trận kết quả & log JSON/CSV
├── configs/
│   └── tddi.yaml                   # File cấu hình siêu tham số
├── data/
│   ├── ddi_incremental.py          # [ĐÃ XONG] Quản lý dữ liệu động & DataLoader cho CIL Tasks
│   ├── preprocessing.py            # Preprocessing từ T-DDI v1.0.0 cũ
│   └── material/                   # Dữ liệu thô gốc (.csv, .txt)
├── methods/
│   ├── finetune.py                 # [ĐÃ XONG] Naive sequential fine-tuning
│   ├── ewc.py                      # [ĐÃ XONG] Elastic Weight Consolidation
│   ├── gem.py                      # [ĐÃ XONG] Gradient Episodic Memory
│   └── agem.py                     # [ĐÃ XONG] Averaged Gradient Episodic Memory
├── models/
│   ├── tddi.py                     # [ĐÃ ADAPT] T-DDI dùng chung cho CIL
│   ├── mlp_resnet.py               # [ĐÃ XONG] MLP/ResNet tabular cho CIL
│   └── model3.py                   # [CẦN VIẾT]
├── PyBioMed/
├── utils.py
├── trainer.py                      # [ĐÃ XONG] Vòng lặp huấn luyện từng task
├── evaluator.py                    # [ĐÃ XONG] Đánh giá R_ij, Acc, F1, Forgetting
└── main.py                         # [ĐÃ XONG] Argument Parser & Main Loop cho CIL
```
