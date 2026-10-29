# Bàn giao

## 2026-10-28: CV đầy đủ đã chạy xong trên Colab (Claude)

**Đã làm:**
- Notebook Colab chạy hết mục 4–8 (2026-10-27 → 2026-10-28):
  - 1.216 task, tất cả `ok`;
  - 22 model × 32, cộng `rnn`/`lstm` 160 mỗi model (5 seed) và `rnn_ar`/`lstm_ar` 96 mỗi model (3 seed).
- Tune trên Colab cho RNN hidden=32 và LSTM hidden=64, cả hai lookback=56, đã ghi vào `best_params.json`. Tham số model thống kê/cây giữ bản tune trên laptop ngày 2026-10-24; file `tuning/` khớp bản cũ.
- `results/` trong repo là bản Drive. `results/` cũ (gồm bản CV laptop ngày 2026-10-25, không dùng) đã chuyển ra `D:\retail-sales-forecasting-results-laptop-old`.
- `env_tune.json` bị Colab ghi đè, đã gộp lại: 2 mục laptop (tune thống kê/cây) và 3 mục Colab (tune RNN/LSTM).
- `env_cv.json`: Colab 2 lõi, Python 3.13.16/3.13.18, cùng bộ thư viện.

**Nhìn sơ** (TOTAL/revenue, MAE trung bình 4 fold):
- RNN 960k, LSTM 961k, XGBoost 961k, LightGBM 964k, GB 970k; ensemble 996–999k.
- Bỏ cờ chế độ: LightGBM 1,56M, SARIMAX 1,30M.
- Xếp hạng trên 16 bài toán: RNN 3,06, LSTM 3,19, XGBoost 3,44, LightGBM 3,81. Nemenyi CD = 4,17, nên không tách được nhóm đầu.
- H1 ("RNN không thắng cây") không được ủng hộ ở CV. Phải bàn trong paper.

**Đọc kỹ CV.** Mọi số dưới đây là revenue; "trung bình 4 chuỗi" là MAE trung bình của 4 chuỗi.
- **Luật §7 chọn RNN.** MAE trung bình 4 fold: RNN 508k, XGBoost 512k, LSTM 513k, LightGBM 514k. Nếu chỉ tính F2–F4, RNN vẫn đứng đầu (503k), ens_avg_topk 504k.
  - Lưu ý: số của RNN trong T1 là **trung bình dự báo của 5 seed**. Tính riêng từng seed, RNN cho MAE 522k ± 17k, LSTM 519k ± 8k, đều kém XGBoost (1 lần fit). Paper phải nói rõ điều này.
  - Ở F4 (gần test nhất): XGBoost 316k, LightGBM 317k, RNN 338k, LSTM 360k.
- **Luật tiêu đề §5 → phải đổi tiêu đề.** Trên F3–F4, khi bỏ một nhóm biến:
  - bỏ `regime`: MAE tăng 64–84% với SARIMAX (riêng WEST chỉ 5–6%) và 81–170% với LightGBM;
  - bỏ `dom`: tăng 12–27% với SARIMAX, khoảng 0 với LightGBM;
  - bỏ `promo`: SARIMAX tăng 1–3%, LightGBM từ −9% đến +5%.
- **H2 đạt.** Bản dùng biến lịch thắng bản pure ở ≥ 3/4 fold trong 72/72 cặp. Mức giảm MAE: cây 7–34%; SARIMAX/RNN/LSTM 47–81%.
- **H3 đạt yếu.** Trên F2–F4, ens_avg_topk 504k so với sel_online 506k. Sd giữa các fold nhỏ hơn ở 3/4 chuỗi, riêng TOTAL thì lớn hơn. Nemenyi CD = 6,66, không tách được.
- **RQ4.** Bottom-up tốt hơn dự báo TOTAL trực tiếp với mọi model chính, 1,2–4%.
- **RQ5.** COGS dự báo trực tiếp tốt hơn đi qua tỷ lệ, 1–4% với các model tốt.

**Hướng paper (người dùng chốt 2026-10-28):**
- Trọng tâm là RQ1: model nào dự báo revenue chính xác nhất.
- RQ2–RQ5 thu gọn thành phần phụ. Ablation biến lịch/cờ chế độ vẫn giữ, để giải thích vì sao nhóm đầu thắng.
- Tiêu đề tạm (trung tính, không khẳng định calendar > model, nên không vướng luật §5): *Benchmarking Statistical, Machine Learning and Deep Learning Models for Daily Retail Revenue Forecasting Across Multiple Regions*. Có thể chỉnh lại sau khi có test.

**Phương pháp đề xuất: RNN** (trung bình 5 seed). Người dùng chốt ngày 2026-10-28 theo luật §7, **trước** khi chạy test, và cho phép chạy test cùng ngày.
- Notebook có thêm mục 10 (test), với 3 lệnh `rsf.run test`, cell kiểm đủ file và lệnh `make_tables/make_figures --test`.
- Nhánh `--test` của hai script đã thử trên dự báo giả (lặp lại dự báo F4, ở thư mục scratch): ra đủ `scores_test.csv`, T7, F08, F10, F11, không lỗi.

## 2026-10-29: test đã chạy xong (một lần, trên Colab)

**Đã làm:**
- 304/304 task test `ok`, không trùng: 22 model × 8, `rnn`/`lstm` 40 mỗi model, bản `_ar` 24 mỗi model.
- `preds/cv` trùng từng byte với bản trước, `runs_cv.csv` không đổi. `env_cv.json` có thêm vài mục do chạy lại cell CV (mọi task đều bỏ qua).
- `results/` cũ (chỉ có CV) chuyển ra `D:\retail-sales-forecasting-results-cv-20261028`.
- `env_tune.json` gộp lại lần nữa: 2 mục laptop và 6 mục Colab.
- Cell kiểm của mục 10 từng lỗi import. Nguyên nhân: kernel coi `/content/rsf` là namespace package. Đã sửa: lấy danh sách model qua `subprocess`.

**Kết quả test** (revenue, MAE trung bình 4 chuỗi):

| Model | MAE |
|---|---|
| XGBoost | 325k |
| GB | 326k |
| LightGBM | 327k |
| ens_stack_nnls | 328k |
| RNN | 337k (từng seed: 344k ± 4k) |
| LSTM | 348k |
| SNaive-1Y | 423k |

- XGBoost thấp hơn SNaive-1Y 23%. MAPE của nhóm đầu khoảng 29%.
- **RNN (phương pháp đề xuất) thua model tốt nhất ở test.** Theo T7 hiện tại, DM cho p = 0,002–0,04 ở cả 4 chuỗi.
  - **Cẩn thận:** `metrics.diebold_mariano` được gọi với h=1, nên phương sai không tính tự tương quan của sai số theo ngày, và p-value bị nhỏ quá mức.
  - Tính thử bằng Newey–West (Bartlett, độ trễ 8 và 30), so XGBoost/GB với RNN:
    - TOTAL p ≈ 0,013, EAST p ≈ 0,02–0,03: vẫn có ý nghĩa;
    - WEST p ≈ 0,04–0,06: sát ngưỡng;
    - CENTRAL so với XGBoost p ≈ 0,35: không có ý nghĩa.
  - XGBoost, GB và LightGBM vẫn không tách được nhau. TOTAL XGBoost so với LightGBM thành p ≈ 0,13–0,15.
  - Sửa DM là việc làm sau khi đã xem điểm, nên phải ghi vào protocol §9 và cần người dùng đồng ý.
- Đúng như tín hiệu ở F4. Paper báo trung thực: RNN thắng sát nút trên CV nhưng không giữ được ở test.
- Bỏ cờ chế độ: LightGBM 659k, gấp đôi. Riêng SARIMAX bỏ cờ thì lại nhỉnh hơn một chút ở test (356k so với 361k).
- COGS ở test: `lightgbm_nopromo` tốt nhất (294k), XGBoost 305k.

**DM đã sửa (2026-10-29, người dùng đồng ý; protocol §9):**
- `metrics.diebold_mariano` dùng phương sai Newey–West, nhân Bartlett, độ trễ ⌊n^(1/3)⌋ = 8.
- `T7_dm_test.csv` dựng lại trên laptop từ dự báo đã lưu. Các bảng khác chỉ lệch nhiễu số thực khoảng 1e-10 do khác phiên bản pandas, nên giữ bản Colab.
- Kết quả: RNN thua có ý nghĩa ở 3/4 chuỗi (Total 0,013, East 0,026, Central 0,029); West 0,064.

**Paper (2026-10-29):**
- Đã viết Methodology và Results/Discussion trong `paper/main.tex`.
- Bảng LaTeX sinh bằng `scripts/make_paper_tables.py` → `paper/tables/*.tex`, được `\input` vào paper. Số trong câu văn lấy từ `results/tables`.
- Máy chưa có LaTeX, nên chưa biên dịch thử; người dùng định dùng Overleaf.
- **Bản nháp đủ mọi mục (2026-10-29):** Abstract (245 từ), Introduction, Related Work, Methodology, Results/Discussion, Conclusion.
- Xem PDF qua Overleaf: chạy `scripts/make_overleaf_zip.py` → `paper/overleaf.zip` (gitignore). Lần đầu upload cả zip; về sau chỉ upload đè `main.tex`.
- Đã điền (2026-10-29):
  - tác giả theo kiểu IEEE (tên trước, họ sau): Thi My Uyen Tran, Duy Thong Ngo;
  - mô tả dữ liệu theo datathon.site và data/raw;
  - link repo GitHub (người dùng đồng ý ghi).
- Người dùng chọn không ghi đơn vị và email tác giả (2026-10-29). Paper không còn `\todo`.
- Tiền tệ: đề thi không ghi (người dùng xác nhận 2026-10-29); paper ghi "units of the source data".
- **Đã kiểm tài liệu (2026-10-29):**
  - Đọc bản đầy đủ: hasan2022, hobor2026, retail2023 (Haque et al.). Chỉ đọc tóm tắt: ensafi2022, albassam2026, theodoridis2025, qureshi2024. MDPI, ScienceDirect và IEEE chặn tải PDF tự động; người dùng có thể tải bằng trình duyệt nếu muốn kiểm kỹ hơn.
  - Đã sửa trích dẫn sai: retail2023 không có model deep learning, nên đã chuyển sang mục Machine learning.
  - Bài chỉ đọc tóm tắt chỉ được dẫn cho kết luận chính, không dẫn cho chi tiết thiết kế. Chi tiết trong `paper/literature_matrix.csv`.
  - `refs.bib` đã có đủ tác giả, số tập và số trang (kiểm qua Crossref). Mục arXiv có thêm `howpublished`, vì IEEEtran.bst bỏ qua `eprint`. Khi mở PDF trên Overleaf, kiểm xem mã arXiv có in ra không.

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
- Token PAT dùng trên Colab (`colab-sales-pilot`) chỉ có scope `sql`, nên script chỉ được gọi API SQL. Ngày 2026-10-27, `current_user.me()` (SCIM) làm Colab lỗi `Unauthenticated`; đã đổi sang `SELECT current_user()`.

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
4. ~~Đọc bản đầy đủ các bài trong `literature_matrix.csv` (cột status) và viết research gap.~~ Xong 2026-10-29 (xem mục Paper).
5. Chưa làm model global cho RQ4 (một LightGBM chung 3 vùng). Hiện RQ4 chỉ so bottom-up với trực tiếp.
6. Viết vào `paper/main.tex` các phần không cần số: Methodology, Experimental setup, nháp Introduction/Related Works. Nguồn là `protocol.md` và `refs.bib`.

**Thứ tự chạy (đổi 2026-10-26: người dùng chốt chạy TẤT CẢ trên Colab, laptop không chạy gì vì thiếu RAM):**
- Ngày 2026-10-25 laptop đã chạy xong CV 22 model thống kê/cây/ablation (704 task, 0 lỗi). Bản đó không dùng; chạy lại trên Colab để mọi model cùng một môi trường. Người dùng giữ bản sao ngoài repo.
- Data trên Colab trích thẳng từ DWH bằng `scripts/extract_dwh.py` (mục 2 notebook, bản `ccde5a5e…`, token Databricks nhập qua getpass) vào Drive `MyDrive/rsf/data/`. Không upload CSV. Đã thử 2026-10-26 với thư mục data trống như Colab: exit 0, file ra trùng từng byte với `data/dwh/`.
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
