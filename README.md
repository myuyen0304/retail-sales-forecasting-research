# retail-sales-forecasting-research

Benchmark dự báo **doanh thu (Revenue) và giá vốn (COGS) theo ngày** cho paper NCKH (viết bằng tiếng Anh).
Paper so sánh 9 model (ARIMA, SARIMA, SARIMAX, Prophet, Random Forest, Gradient Boosting, XGBoost, LightGBM, RNN, LSTM), 2 baseline và 3 cách ensemble.
Dữ liệu gồm 4 chuỗi: TOTAL và 3 vùng (multi-region).

- **Thiết kế thực nghiệm (đóng băng):** [protocol.md](protocol.md). Đọc trước khi sửa code.
- **Trạng thái và bước tiếp:** [HANDOFF.md](HANDOFF.md).
- **Bản cũ do GPT dựng** (export Databricks, 40 model, chạy Colab): tag `legacy-gpt-pilot`, nhánh `legacy/gpt-pilot`.

## Cài đặt (laptop Windows hoặc Linux, CPU là đủ)

```bash
uv venv .venv --python 3.13
uv pip install --python .venv/Scripts/python.exe -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu
uv pip install --python .venv/Scripts/python.exe -e .
```

## Dữ liệu (không có trong git)

Chép 6 file CSV Datathon vào `data/raw/`: `sales`, `orders`, `order_items`, `products`, `geography`, `promotions`. Sau đó chạy:

```bash
.venv/Scripts/python.exe -m rsf.data      # → data/processed/daily_series.csv + promotions.csv, tự kiểm 3 vùng cộng về TOTAL
```

Lên Colab chỉ cần upload 2 file trong `data/processed/`. Xem [notebooks/colab_run.ipynb](notebooks/colab_run.ipynb).

## Chạy thực nghiệm

Thứ tự bắt buộc: `tune` → `cv` → (khóa giao thức) → `test`. Mỗi task ghi một file, nên chạy lại sẽ bỏ qua task đã xong.

```bash
PY=.venv/Scripts/python.exe
$PY -m pytest -q                                          # kiểm giao thức (rò rỉ, split, metric)
$PY -m rsf.run tune --models main --jobs 3                # lưới tham số trên TOTAL/revenue → results/best_params.json
$PY -m rsf.run cv   --models all  --jobs 3                # 4 chuỗi × 2 target × 4 fold
$PY scripts/make_tables.py && $PY scripts/make_figures.py # bảng + hình CV
$PY -m rsf.run test --models all  --jobs 3                # CHỈ chạy sau khi đã chọn phương pháp đề xuất trên CV
$PY scripts/make_tables.py --test && $PY scripts/make_figures.py --test
```

Trên console Windows, nếu lỗi in tiếng Việt thì đặt `PYTHONUTF8=1`. Đổi thư mục kết quả bằng biến môi trường `RSF_RESULTS_DIR`.

## Cấu trúc

| Đường dẫn | Vai trò |
|---|---|
| `src/rsf/data.py` | Dựng chuỗi TOTAL + 3 vùng từ CSV gốc, kiểm khớp |
| `src/rsf/features.py` | Biến lịch, khuyến mãi, chế độ (chống rò rỉ theo origin) |
| `src/rsf/splits.py` | 4 fold CV + test |
| `src/rsf/models/` | baseline, thống kê (ARIMA/SARIMA/SARIMAX/Prophet), cây, RNN/LSTM |
| `src/rsf/grids.py` | Lưới tham số |
| `src/rsf/ensemble.py`, `results.py` | Ensemble, bottom-up, COGS qua tỷ lệ, đọc dự báo |
| `src/rsf/run.py` | CLI chạy tune / cv / test |
| `scripts/make_tables.py`, `make_figures.py` | Nguồn duy nhất của mọi bảng/hình trong paper |
| `results/` | `tables/`, `figures/`, `best_params.json`, `tuning/`, `runs_*.csv` (commit); `preds/` (không commit) |
| `paper/` | `refs.bib` (DOI đã kiểm), `literature_matrix.csv`, bản thảo |
| `tests/` | Test giao thức |
