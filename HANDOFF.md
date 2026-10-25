# Bàn giao

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

**Thứ tự chạy đã thống nhất (2026-10-25):**
- Chạy trước phần nhanh, khoảng 1 giờ: `cv --models baselines,main,ablation --jobs 3`. Ở lần này RNN/LSTM dùng tham số mặc định.
- Sau đó chạy `tune --models rnn,lstm`, rồi `cv --models rnn,lstm`, rồi `cv --models pure --seeds 3`, rồi chạy `make_tables.py` và `make_figures.py`.
- Nếu paper điểm cộng sát hạn, có thể bỏ phần `pure` của mạng nơ-ron (`rnn_ar`, `lstm_ar`). Người dùng chưa cho biết hạn nộp.

**Code chưa từng chạy:**
- `notebooks/colab_run.ipynb`;
- nhánh `--test` của `make_tables.py` / `make_figures.py` (T7, F10, F11).

Không chạy thử các phần này trên vùng test thật. Khi đến lúc chạy test mà gặp lỗi, chỉ cần sửa script rồi dựng lại bảng từ dự báo đã lưu, không phải chạy lại model.

**Commit/push:** 2026-10-25 đã commit trên `main` ở local, mỗi file một commit (sau c370252). Chưa push, kể cả tag và nhánh legacy.
