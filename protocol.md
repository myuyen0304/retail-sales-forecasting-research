# Giao thức thực nghiệm (đóng băng trước khi chạy CV)

**Ngày đóng băng:** 2026-10-24, trước lần chạy `cv` đầy đủ đầu tiên.

**Những gì đã được nhìn thấy trước khi đóng băng.** Ghi ra để paper không gọi sai là "pre-registered" hay "untouched hold-out":
1. *Smoke run khi viết code.*
   - 16 model trên TOTAL/revenue fold F4, tham số mặc định.
   - 7 model trên F3–F4 cho cả 4 chuỗi.
   - Giả thuyết H1, H2 viết **sau** khi đã thấy các số này.
2. *Bộ biến và vùng test đã được dùng trong dự án trước.*
   - Bộ biến (cờ khuyến mãi, `post2019`, ngày trong tháng) và việc chọn LightGBM lấy từ khóa luận Retail Analytics.
   - Khóa luận đó backtest trên 2020-07 → 2022-12, tức là gần như trùng toàn bộ vùng test 2021-07 → 2022-12 ở đây.
   - Vì vậy test **không** phải dữ liệu chưa từng dùng để phát triển model. Paper phải ghi điều này trong phần Limitations.
3. *Mốc đứt gãy:*
   - Ở mọi origin có biến `post2019`, việc sụt mức đã quan sát được trong dữ liệu train: khoảng 6 tháng ở F3, 18 tháng ở F4, 30 tháng ở TEST.
   - Chỉ riêng ngày bắt đầu chính xác (2019-01-01) là lấy từ EDA trên toàn chuỗi.
   - Ablation `*_noregime` (§5) đo giá trị của việc xử lý tường minh sự dời mức này.

**Bước tune của các model thống kê + cây** đã chạy ngày 2026-10-24 (`results/tuning/`). Bước này chỉ dùng TOTAL/revenue trên 4 fold CV, không dùng test.
Mọi thay đổi sau ngày này phải ghi ở §9 kèm lý do. Kết quả có từ thay đổi sau khi đã xem điểm phải gắn nhãn *exploratory* trong paper.

Mã nguồn tương ứng: `src/rsf/` (split ở `splits.py`, biến ở `features.py`, model ở `models/`, lưới ở `grids.py`).

## 1. Dữ liệu

- **Nguồn:** bộ dữ liệu Datathon 2026 Round 1. Cột `Date, Revenue, COGS` của `sales.csv` tái tạo chính xác từ `order_items × orders × products`:
  - Revenue = Σ quantity × unit_price, không trừ chiết khấu, gồm mọi trạng thái đơn;
  - COGS = Σ quantity × cogs.
- **4 chuỗi theo ngày:** TOTAL và 3 vùng EAST / CENTRAL / WEST (qua `orders.zip → geography.region`).
- **Điều kiện kiểm** (`rsf.data.check`, tự động mỗi lần load):
  - 3 vùng cộng lại bằng TOTAL;
  - không thiếu ngày;
  - mọi giá trị > 0.
- **Khoảng thời gian:** 2012-07-04 → 2022-12-31, 3.833 ngày mỗi chuỗi.
- **2 target:** revenue (chính) và cogs.
- **Gọi tên trong paper:** "multi-region". Dữ liệu không có cửa hàng (store), nên không dùng chữ "store".

## 2. Câu hỏi nghiên cứu và giả thuyết

| RQ | Giả thuyết đăng ký trước | Cách kiểm |
|---|---|---|
| RQ1: họ model nào tốt nhất cho dự báo ngày dài hạn (≈1 năm ở CV, 549 ngày ở test)? | H1: model cây dùng biến lịch (LightGBM/XGBoost) tốt nhất; mạng hồi quy không hơn cây | Bảng MAE/RMSE/MAPE; Friedman + Nemenyi trên (chuỗi × fold); DM ở test |
| RQ2: biến lịch quan trọng thế nào so với cấu trúc model? | H2: trong cùng họ, bản calendar-aware có MAE thấp hơn bản pure ở ≥ 3/4 fold | Cặp ablation `models.ABLATION` |
| RQ3: ensemble có hơn model đơn tốt nhất không? | H3: ít nhất một cách kết hợp có MAE CV ≤ model đơn tốt nhất và sd giữa các fold nhỏ hơn | `ensemble.py` |
| RQ4: tổng trực tiếp hay cộng từ vùng (bottom-up)? | H4: không đặt chiều; báo kết quả | Σ 3 vùng so với TOTAL, cùng model |
| RQ5: COGS trực tiếp hay qua tỷ lệ? | Không đặt chiều | COGS trực tiếp so với revenue dự báo × tỷ lệ tháng × Urban Blowout học trên train |

**Phân tích sai số bắt buộc**, lấy từ cùng một lần dự báo, không chạy thêm:
- theo lead time: 1–30, 31–90, 91–365, 366–549 ngày;
- theo tháng;
- riêng tháng 8 năm lẻ (Urban Blowout).

## 3. Phân chia thời gian

| Phần | Khoảng | Vai trò |
|---|---|---|
| CV F1 | origin 2017-06-30, chấm 365 ngày | Tháng 8/2017 lẻ, trước đứt gãy |
| CV F2 | origin 2018-06-30 | Chứa đứt gãy cuối 2018 |
| CV F3 | origin 2019-06-30 | Tháng 8/2019 lẻ |
| CV F4 | origin 2020-06-30 | Cũng là "validation" của sơ đồ train/val/test |
| **Test** | origin 2021-06-30, chấm 2021-07-01 → 2022-12-31 (549 ngày) | Chấm **một lần** sau khi khóa tham số và luật ensemble |

- CV dùng cửa sổ mở rộng: train luôn bắt đầu từ 2012-07-04.
- Mỗi model dự báo một lần từ origin cho toàn bộ vùng chấm. Không cập nhật bằng số thực tế giữa chừng.
- Các fold không độc lập: báo từng fold kèm mean ± sd, không dùng t-test thường.
- Paper ghi rõ đây là **retrospective benchmark**: dữ liệu lịch sử đã được EDA trong dự án trước.
- Năm 2023–2024 không có số thực tế, nên không khẳng định độ chính xác cho giai đoạn đó.

## 4. Biến (dùng chung, cố định trước)

**Lý do chọn biến:** đây là biến có thể biết trước ở thời điểm dự báo. Mọi bảng phụ của bộ dữ liệu dừng ở 2022-12-31, nên không dùng traffic/inventory tương lai.

**Bộ biến `linear`** (cho SARIMAX và Prophet):
- Fourier năm K=5;
- `days_to_eom`, `is_last1..3`, `is_first1..3`;
- 6 cờ khuyến mãi;
- `post2019`.

**Bộ biến `tree`** (cho cây và RNN/LSTM) = `linear` + `month`, `dom`, `doy`, `is_odd_year` + số ngày kể từ khi đợt khuyến mãi bắt đầu.

**Chống rò rỉ:**
- Cửa sổ khuyến mãi lặp theo ngày dương lịch, chỉ học từ các đợt có `start_date ≤ origin`.
- Đợt chỉ thấy ở năm lẻ (≥ 2 lần) chỉ được bật ở năm lẻ.
- `post2019` chỉ có khi origin ≥ 2019-06-30, tức đã quan sát được ít nhất 180 ngày sau đứt gãy. Ở F1, F2 không có biến này.
- Mốc 2019-01-01 là kiến thức từ EDA trước đó. Paper ghi rõ điều này.

**Không dùng ở bản chính:** thứ trong tuần (EDA cũ: R² = 0,008), biến năm dạng số.

## 5. Model

**Bản chính (xếp hạng RQ1)**, mọi model fit trên log(y):

| Model | Định nghĩa |
|---|---|
| `arima` | Không biến ngoài; (p,d,q) chọn bằng AIC trong {0,1,2}×{0,1}×{0,1,2} trên train |
| `sarima` | Mùa vụ năm bằng Fourier K=5 + sai số ARMA. Đây là dạng khả thi cho s=365 (Hyndman & Athanasopoulos, dynamic harmonic regression) |
| `sarimax` | Bộ biến linear + ARMA |
| `prophet` | Mùa vụ năm của Prophet + biến linear (trừ Fourier) làm regressor |
| `rf`, `gb` (HistGradientBoosting), `xgboost`, `lightgbm` | Hồi quy trực tiếp trên bộ biến tree, không lag |
| `rnn`, `lstm` | Chuỗi biến tree của L ngày kết thúc tại t → log y_t, không tự hồi quy. Số epoch chọn bằng early stopping trên 10% cuối của train, rồi fit lại toàn train. 5 seed, báo mean ± sd; ensemble dùng trung bình 5 seed |

**Baseline:**
- `snaive1y`: cùng ngày năm trước;
- `snaive2y`: cùng ngày 2 năm trước, giữ tính chẵn/lẻ của năm.

**Bản pure (ablation RQ2):**
- `sarima_w7`: (1,1,1)×(1,0,1,7);
- `prophet_pure`: mặc định;
- `*_lag`: lag {1,2,3,7,14,28,30,60,90,180,365} + trung bình trượt {7,28,90}, dự báo đệ quy;
- `rnn_ar`, `lstm_ar`: đơn biến L=56, đệ quy.

Bản pure dùng tham số mặc định, không tinh chỉnh; `rnn_ar` / `lstm_ar` chạy 3 seed. Paper phải ghi rõ điều này.

**Ablation theo nhóm biến (RQ2)**, thêm trước CV:
- 6 biến thể: `sarimax_no{regime,promo,dom}` và `lightgbm_no{regime,promo,dom}`.
- Mỗi biến thể bỏ một nhóm biến:
  - `regime` = `post2019`;
  - `promo` = 6 cờ khuyến mãi + `_since` + `is_odd_year`;
  - `dom` = `days_to_eom`, `dom`, `is_last*`, `is_first*`.
- Biến thể dùng tham số đã tune của model gốc.
- Kết quả ở bảng `T2b_feature_groups.csv`.
- So sánh các nhóm **chỉ trên F3–F4**, là các fold mà cả ba nhóm biến đều tồn tại. Ở F1–F2 chưa có `post2019`.
- **Tiêu đề paper** ("calendar information matters more than model class") chỉ giữ nếu, trên F3–F4, phần tăng sai số khi bỏ `promo` hoặc `dom` vượt phần khi bỏ `regime`.
  Nếu ngược lại, viết lại tiêu đề theo hướng: xử lý tường minh sự dời mức quan trọng hơn chọn họ model.

## 6. Tinh chỉnh

- Lưới ở `grids.py`, mỗi model tối đa 12 cấu hình.
- Chấm trên chuỗi TOTAL / revenue, 4 fold, 1 seed. Chọn cấu hình có MAE trung bình 4 fold thấp nhất (`python -m rsf.run tune`).
- Cấu hình thắng dùng cho mọi chuỗi và target.
- Không dùng test để chọn bất cứ thứ gì.

## 7. Ensemble

- Ứng viên: 10 model bản chính.
- Ba cách: `avg_topk` (k=3), `inv_mae` (top 3), `stack_nnls`.
- Ở fold Fk, trọng số học từ dự báo của các fold trước Fk; F1 dùng trung bình đều.
- Ở test, trọng số học từ cả 4 fold.
- **Đối chứng công bằng cho H3:** `sel_online`, một model đơn chọn bằng cùng luật online (MAE thấp nhất trên các fold trước).
  - H3 được đánh giá trên **F2–F4**, so ensemble với `sel_online` (cột `mae_F2F4`, bảng `T3b_ranks_F2F4`).
  - Lý do bỏ F1: ở F1, ensemble chưa có fold trước để học, nên là trung bình đều cả 10 model.
  - Không so với "model đơn tốt nhất theo CV", vì model đó được chọn khi đã biết kết quả.
- "Majority voting" là phương pháp cho phân loại. Với hồi quy, bản tương đương là averaging (Voting Regressor); paper trình bày đúng như vậy.
- **Phương pháp đề xuất:** chọn trên CV, theo MAE trung bình (revenue, 4 chuỗi). Chọn xong rồi mới mở test.

## 8. Đánh giá

- RMSE, MAE, MAPE (%), tính trên đơn vị gốc, sau khi đổi ngược từ log.
- DM (bản HLN, |e|) cho các cặp đầu bảng ở test.
- Friedman + Nemenyi (α = 0,05) → critical-difference diagram trên 4 chuỗi × 4 fold (revenue).
- Mọi bảng và hình của paper sinh từ `scripts/make_tables.py` và `scripts/make_figures.py`.

## 9. Nhật ký thay đổi sau khi đóng băng

| Ngày | Thay đổi | Lý do | Đã xem điểm chưa |
|---|---|---|---|
| 2026-10-24 | Sửa lỗi code: điểm xuất phát AR cho bậc p=2 của `sarima`/`sarimax` (trước đây lặp ρ cho mọi bậc nên ra điểm không dừng → NaN / lỗi Schur); chạy lại các cấu hình bị lỗi. Lưới và luật chọn giữ nguyên | Lỗi cài đặt, không phải đổi thiết kế | Đã thấy MAE của cấu hình (1,0,·) trong tuning; chưa thấy CV/test |
| 2026-10-24 | Thêm 6 biến thể ablation theo nhóm biến (§5) | Ablation calendar-vs-pure gộp chung cờ chế độ (kiến thức hindsight) với biến lịch thật, nên không tách được hai phần đóng góp | Đã thấy tuning: SARIMA (không cờ chế độ) ≈ 1,70M so với SARIMAX ≈ 1,02M; chưa thấy CV/test |
| 2026-10-24 | Bảng ablation theo nhóm biến chỉ tính trên F3–F4; luật tiêu đề cũng theo F3–F4 | Ở F1–F2, bản `*_noregime` trùng bản gốc, nên tính trung bình 4 fold sẽ làm loãng hiệu ứng của cờ chế độ | Chưa thấy CV |
| 2026-10-24 | Thêm `sel_online`; đánh giá H3 trên F2–F4 (§7) | So ensemble với model đơn được chọn khi đã biết kết quả là thiên lệch | Chưa thấy CV ensemble |
| 2026-10-24 | Ghi chú: fold F2 (chứa đứt gãy) có MAE gấp khoảng 2 lần các fold khác, nên chi phối việc chọn tham số theo MAE trung bình. Không đổi luật; ghi vào phần thảo luận của paper | Minh bạch | Đã thấy tuning |
