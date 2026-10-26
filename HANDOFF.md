# Bàn giao

## 2026-10-25: trích data từ DWH Databricks (Claude)

**Mục đích.** Người dùng muốn paper lấy data từ DWH. Data từ DWH không khác data CSV; cái được là nguồn gốc ghi được trong paper.

**Đã làm:**
- `scripts/extract_dwh.py`:
  - tổng hợp theo ngày × vùng trên Databricks từ `fact_order_item` + `dim_date` + `dim_customer` → `dim_geography`, và lấy `dim_promotion`;
  - ghi ra `data/dwh/` (`daily_series.csv`, `promotions.csv`, `manifest.json`), không ghi đè `data/processed/`;
  - tự đối chiếu với bản CSV và với `rpt_forecast_actual`.
- `requirements-dwh.txt` có `databricks-sdk==0.117.0`, đã cài vào `.venv` bằng `uv pip`. `requirements.txt` không đổi, nên Colab không bị ảnh hưởng.

**Bằng chứng** (bản `ccde5a5e21c54e9e9054066f`, catalog `retail_lab`, user `uyenuyen9780@gmail.com`):
- 4 chuỗi khớp bản CSV đủ 3.833 ngày: lệch tối đa 1,3e-8 ở TOTAL, ≤ 1e-8 ở vùng.
- Vùng của DWH lấy qua `customers.zip`, vùng của CSV lấy qua `orders.zip`; kết quả cho thấy hai cách khớp nhau.
- Promotions: 50 dòng, `assert_frame_equal` đạt sau khi ép `discount_value` về double và `stackable_flag` về int.
- TOTAL so với `rpt_forecast_actual`: lệch 0,005, do bảng đó làm tròn COGS 2 chữ số.
- `rsf.data.load(Path('data/dwh'))` đạt; `pytest` 7/7.
- **Không cần chạy lại CV.** Data trùng bản CSV đã dùng, nên không phải đổi protocol.

**Quyền trên Databricks** (cấp tay 2026-10-25, chỉ cho `uyenuyen9780@gmail.com`):
- `USE SCHEMA` + `SELECT` trên schema `r_ccde5a5e…_marts`;
- `SELECT` trên `r_ccde5a5e…_reporting.rpt_forecast_actual`.

**Bẫy:**
- **Không** cấp cho `account users` hay nhóm workspace. Làm vậy khóa chat AI của Retail Analytics, vì guard chỉ đọc thấy SP AI có quyền thừa; đã xảy ra ngày 2026-10-25.
- Khi pipeline publish bản mới, schema `r_<release>` đổi, phải cấp quyền lại và truyền `--release` mới.

**Lệnh:** `python scripts/extract_dwh.py --profile retail-dev --release ccde5a5e21c54e9e9054066f`. Script trả exit 1 nếu lệch.

**Bước tiếp:**
- Câu nguồn data cho paper (§3) có thể ghi: data trích từ DWH, bản `ccde5a5e…`, đã kiểm trùng khớp với nguồn Datathon.
- Muốn train trực tiếp từ bản DWH thì đặt `RSF_PROCESSED_DIR=data/dwh`.

## 2026-10-24: làm lại repo theo hướng paper NCKH (Claude)

**Bối cảnh.** Bản GPT (export Databricks từ Colab, 40 model, 5 horizon) chạy xong nhưng không rút được kết luận. Nguyên nhân là thiết kế:
- model không có cờ khuyến mãi, cờ chế độ hay ngày trong tháng;
- SARIMA dùng mùa vụ tuần s=7;
- cây và LSTM dự báo đệ quy 548 bước;
- chỉ chạy được 1 fold, mỗi task một hình riêng.

Bản GPT giữ ở tag `legacy-gpt-pilot` và nhánh `legacy/gpt-pilot`, chỉ ở local, chưa push.

**Người dùng đã chốt:**
- tổng + 3 vùng (gọi là "multi-region");
- làm lại trong repo này;
- script chạy được cả laptop lẫn Colab, việc nặng do người dùng chạy;
- paper viết bằng tiếng Anh.

**Đã làm:**
- Khung mới `src/rsf` (data, features, splits, metrics, models, grids, ensemble, results, run), `scripts/make_tables.py`, `scripts/make_figures.py`, `tests/`, `notebooks/colab_run.ipynb`.
- `protocol.md` đóng băng thiết kế, kèm RQ1–RQ5 và giả thuyết.
- `paper/refs.bib`: 39 mục, DOI kiểm qua Crossref, arXiv kiểm qua export.arxiv.org. `paper/literature_matrix.csv` ghi những gì đã đọc. `paper/main.tex` là khung IEEE theo template 4.1–4.5.
- `data/processed/` đã dựng từ CSV gốc. 3 vùng khớp TOTAL: revenue lệch 4e-9, COGS lệch 0,005.
- `.venv` Python 3.13.5; phiên bản thư viện ở `requirements.txt`.

**Bằng chứng:**
- `pytest`: 7/7 pass.
- Smoke 16 model trên TOTAL/revenue fold F4, tham số mặc định:
  - bản calendar-aware có MAE 548k (XGBoost) – 678k (SARIMAX);
  - bản pure có MAE 0,88M (LightGBM lag đệ quy) – 1,8M (SARIMA s=7);
  - SNaive-1Y 668k.
  Đây là số kiểm code, **không phải kết quả paper**: chưa tune và chỉ có 1 fold.
- Smoke end-to-end (7 model, F3–F4, 4 chuỗi) → đủ bảng T1–T8 và hình F01–F09 ở thư mục scratch, không nằm trong `results/`.

**Tune model thống kê + cây (đã chạy 2026-10-24, TOTAL/revenue, 4 fold):**
- Kết quả ở `results/best_params.json` và `results/tuning/`.
- MAE trung bình tốt nhất: XGBoost 949k, LightGBM 964k, GB 973k, Prophet 1,01M, SARIMAX 1,02M, RF 1,07M, SARIMA (Fourier) 1,70M.
- Fold F2 (chứa đứt gãy) có MAE khoảng gấp đôi các fold khác.
- Đã sửa lỗi điểm xuất phát AR(2) và chạy lại; ghi ở protocol §9.

**Tín hiệu sớm cần biết trước khi viết paper** (smoke F3–F4, chỉ TOTAL + EAST, chưa phải kết quả paper):
- Bỏ cờ chế độ `post2019` làm MAE tăng 75–160%.
- Bỏ ngày trong tháng làm SARIMAX tăng ~26%.
- Bỏ cờ khuyến mãi chỉ đổi ±10%.
- Nếu CV đầy đủ cũng ra như vậy:
  - phần lớn lợi thế đến từ việc xử lý tường minh sự dời mức sau 2018, không phải từ biến lịch;
  - đây vẫn là một phát hiện có giá trị: LightGBM không có cờ chế độ còn kém hơn SNaive-1Y;
  - tiêu đề tạm trong `paper/main.tex` phải đổi theo luật ở protocol §5, so trên F3–F4.
- Lưu ý về smoke này:
  - Bản gốc chạy tham số mặc định: LightGBM mặc định, SARIMAX (1,0,1).
  - Biến thể chạy tham số đã tune: LightGBM đã tune, SARIMAX (2,0,2).
  - Với SARIMAX, tune cho hai bậc gần như bằng nhau, nên số vẫn dùng được.
  - Với LightGBM, mức "−5 đến −13%" khi bỏ promo/dom chủ yếu là do khác tham số. Đừng coi đó là kết quả.

**Chưa làm / bước tiếp:**
1. `tune` cho RNN/LSTM (nặng).
2. `cv`: 4 chuỗi × 2 target × 4 fold. Người dùng chạy trên laptop, trong PowerShell riêng ngoài Claude Code.
   - Ước tính trên laptop 12 lõi, `--jobs 3`:
     - tune RNN/LSTM khoảng 30–60 phút;
     - CV các model không phải mạng nơ-ron khoảng 1 giờ;
     - CV RNN/LSTM (5 seed) và bản AR (3 seed) khoảng 3–4 giờ.
   - RNN chậm nhất: khoảng 100 giây mỗi lần fit.
   - Colab bản miễn phí chỉ có 2 lõi CPU, nên sẽ chậm hơn laptop.
3. Đọc bảng CV, chốt phương pháp đề xuất, rồi mới chạy `test` một lần.
4. Đọc bản đầy đủ các bài trong `literature_matrix.csv` (cột status) và viết research gap.
5. Chưa làm model global cho RQ4 (một LightGBM chung 3 vùng). Hiện RQ4 chỉ so bottom-up với trực tiếp.
6. Viết vào `paper/main.tex` các phần không cần số: Methodology, Experimental setup, nháp Introduction/Related Works. Nguồn là `protocol.md` và `refs.bib`.

**Thứ tự chạy (đổi 2026-10-26: người dùng chốt chạy TẤT CẢ trên Colab, laptop không chạy gì vì thiếu RAM):**
- Ngày 2026-10-25 laptop đã chạy xong CV 22 model thống kê/cây/ablation (704 task, 0 lỗi). Bản đó không dùng; chạy lại trên Colab để mọi model cùng một môi trường. Người dùng giữ bản sao ngoài repo.
- Tham số model thống kê/cây vẫn lấy từ `results/best_params.json` đã tune trên laptop ngày 2026-10-24 (notebook chép sang Drive), không tune lại.
- `notebooks/colab_run.ipynb`, trình tự:
  - mục 4: `cv --models baselines,arima,sarima,sarimax,prophet,rf,gb,xgboost,lightgbm,ablation --jobs 2`, rồi `cv --models sarima_w7,prophet_pure,rf_lag,gb_lag,xgboost_lag,lightgbm_lag --jobs 2`;
  - mục 5: `tune --models rnn,lstm`;
  - mục 6: `cv --models rnn,lstm` (5 seed), rồi `cv --models rnn_ar,lstm_ar --seeds 3`;
  - mục 8: `make_tables.py`, `make_figures.py`.
- Kết quả ở Drive `MyDrive/rsf/results`. Người dùng tải về, cắt `results/` cũ ra ngoài repo rồi đặt bản Drive vào `results/`. Không cần `merge_colab.py` (giữ lại, dùng khi chia laptop/Colab).
- **Bẫy:** `cv` bỏ qua task đã có file trong `preds/` mà không kiểm tham số. Đừng chạy `--models main` trước khi tune RNN/LSTM. Đổi tham số của model nào thì phải xóa `results/preds/cv/<model>/` và các dòng của model đó trong `runs_cv.csv` rồi mới chạy lại.
- Ghi trong paper môi trường chạy (Colab) theo `env_*.json`; riêng tune model thống kê/cây chạy trên laptop (`env_tune.json`).
- Nếu paper điểm cộng sát hạn, có thể bỏ `rnn_ar` và `lstm_ar`. Người dùng chưa cho biết hạn nộp.

**Code chưa từng chạy:**
- `notebooks/colab_run.ipynb`: viết lại ngày 2026-10-26 để chạy toàn bộ CV và dựng bảng/hình. Code mới kiểm trên pandas 3.1, còn Colab dùng pandas 2.x;
- nhánh `--test` của `make_tables.py` / `make_figures.py` (T7, F10, F11).

Không chạy thử các phần này trên vùng test thật. Khi đến lúc chạy test mà gặp lỗi, chỉ cần sửa script rồi dựng lại bảng từ dự báo đã lưu, không phải chạy lại model.

**Commit/push:** 2026-10-26 `origin/main` đã có tới 26af59c. Notebook chạy-tất-cả-trên-Colab commit sau đó, người dùng cần push trước khi mở Colab.
