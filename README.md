# NN-XC-DFT---MSAI

# ml_xc_colab

Chép toàn bộ thư mục này vào `MyDrive/ml_xc_colab/` rồi chạy các notebook theo thứ tự 00 → 08.

| Notebook | Việc làm | Đầu ra chính |
|---|---|---|
| 00_setup | Cài thư viện, kiểm tra Secrets, benchmark PySCF, kiểm tra JAX/GPU | `processed/env_report.json` |
| 01_download_mp | Tải cấu trúc + metadata từ Materials Project | `raw/mp_manifest.csv`, `raw/cif/` |
| 02_preprocess_mp | Kiểm tra, loại lỗi/trùng, chọn ≤1000 mẫu, chia train/val/test | `processed/selected.csv` |
| 03_pbe_pilot | Chạy thử, chốt basis/k-point/lưới/spin/ngưỡng hội tụ | `processed/pbe_settings.json` |
| 04_run_pbe_density | PBE PySCF PBC cho cả tập (có resume) | `densities/` |
| 05_reference_labels | Nhãn tham chiếu (r2SCAN trên mật độ PBE) + sai số | `reference_labels/` |
| 06_build_dataset | Ghép thành HDF5 + manifest + metadata | `ml_xc_dataset.h5` |
| 07_train_ml_xc | Huấn luyện (train fit, val chọn mô hình) | `models/best_model.npz` |
| 08_evaluate | Đánh giá test, so sánh PBE/LDA/hằng số | `processed/test_metrics.json` |

Trước khi chạy 01: tạo Colab Secret tên `MP_API_KEY` (bật Notebook access). Không dán key vào code.
`data/models/` là thư mục thêm so với khung gốc, để lưu mô hình.
