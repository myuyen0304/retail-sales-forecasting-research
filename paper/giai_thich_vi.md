# Giải thích paper bằng tiếng Việt

File này đi kèm `paper/main.tex`. Mỗi mục có 3 phần:
- **Câu chính trong paper**: trích nguyên văn tiếng Anh.
- **Nghĩa là gì**: giải thích bằng lời dễ hiểu.
- **Khi trình bày nói**: 1–2 câu để nói trước hội đồng.

Số liệu trong file này lấy từ paper bản ngày 29/10/2026. Nếu paper đổi số thì phải sửa file này theo.

---

## 0. Toàn bài trong 4 ý

1. **Vấn đề:** các bài báo trước mỗi bài nói một mô hình khác là tốt nhất, vì họ thi trên dữ liệu khác nhau và theo luật khác nhau.
2. **Cách làm:** cho 10 mô hình thi công bằng trên cùng dữ liệu, cùng thông tin đầu vào và cùng luật chấm.
3. **Ai thắng:** XGBoost, Gradient Boosting và LightGBM (nhóm mô hình cây) chính xác nhất và rẻ nhất.
4. **Bài học:** cho mô hình thông tin đúng (lịch, khuyến mãi, mốc 2019) quan trọng hơn chọn mô hình phức tạp. Chênh lệch nhỏ lúc thử chưa chắc là thắng thật.

---

## Tên bài

> Benchmarking Statistical, Machine Learning and Deep Learning Models for Daily Retail Revenue Forecasting Across Multiple Regions

**Nghĩa là gì:** So sánh (benchmark) 3 nhóm mô hình (thống kê, máy học, học sâu) trong việc dự báo doanh thu bán lẻ theo ngày, cho nhiều vùng.

---

## Abstract (Tóm tắt)

| Câu trong paper | Nghĩa là gì |
|---|---|
| Published comparisons of retail forecasting models often differ in data, horizons and inputs, which makes their conclusions hard to transfer. | Các bài so sánh trước đây dùng dữ liệu, độ dài dự báo và đầu vào khác nhau, nên kết luận của họ khó áp dụng sang chỗ khác. |
| This study benchmarks ten widely used models, namely ARIMA, …, RNN and LSTM, against seasonal-naive baselines and three forecast-combination schemes. | Bài này so 10 mô hình với mốc so sánh "copy năm trước" và 3 cách ghép mô hình. |
| The data are daily revenue and cost of goods sold of an online retailer for a national total and three regions (3,833 days, 2012–2022). | Dữ liệu là doanh thu và giá vốn theo ngày của một cửa hàng online, gồm tổng cả nước và 3 vùng, 3.833 ngày. |
| All models receive the same known-in-advance calendar, promotion and regime covariates and forecast directly up to 18 months ahead. | Mô hình nào cũng được cho cùng thông tin biết trước (lịch, khuyến mãi, mốc 2019) và dự báo một lần tới 18 tháng. |
| Models are compared with four-fold rolling-origin cross-validation, and the selected method is evaluated once on a 549-day hold-out test set. | So ở vòng loại 4 lượt, rồi mô hình được chọn thi chung kết 1 lần trên 549 ngày giấu kín. |
| On the test set, gradient-boosted trees are both the most accurate and the cheapest models. | Ở chung kết, nhóm mô hình cây vừa chính xác nhất vừa rẻ nhất. |
| XGBoost reduces revenue MAE by 23% relative to a seasonal-naive forecast (MAPE 29%). | XGBoost sai ít hơn cách "copy năm trước" 23%. Trung bình mỗi ngày lệch khoảng 29%. |
| The RNN, which ranked first in cross-validation by a margin of about 1%, is significantly less accurate than the best model in three of four series. | RNN đứng nhất vòng loại, chỉ hơn khoảng 1%, nhưng ở chung kết thì thua rõ ở 3 trên 4 chuỗi. |
| Calendar information reduces error by 21–84%. | Cho mô hình biết lịch giúp giảm sai số 21–84%. |
| How a model receives information about the series matters more than the choice of model family. | Cách đưa thông tin vào mô hình quan trọng hơn chọn loại mô hình. |

**Khi trình bày nói:** "Nhóm em so 10 mô hình dự báo doanh thu theo ngày trong cùng một luật thi. Kết quả là XGBoost tốt nhất, giảm 23% sai số so với cách đoán theo năm trước, và thông tin lịch quan trọng hơn việc chọn mô hình phức tạp."

---

## I. Introduction (Giới thiệu)

**Đoạn 1: Bối cảnh**

> Daily forecasts drive replenishment, workforce scheduling and cash-flow planning.

Dự báo doanh thu theo ngày giúp cửa hàng nhập hàng, xếp ca nhân viên và lo dòng tiền. Có 3 nhóm mô hình để chọn: thống kê (ARIMA, Prophet), máy học (Random Forest, XGBoost…), học sâu (RNN, LSTM).

> Yet applied comparisons on retail data name different winners.

Các bài báo trước mỗi bài một kết quả: bài thì ARIMA thắng, bài thì LSTM, bài thì XGBoost.

**Đoạn 2: Vì sao kết quả mâu thuẫn (5 lý do)**

| Lý do | Nghĩa là gì |
|---|---|
| different inputs | Mô hình này được cho biết lịch, mô hình kia thì không, nên thi không công bằng. |
| horizons are often short | Nhiều bài chỉ dự báo ngắn (28 ngày), trong khi doanh nghiệp cần dự báo xa. |
| multi-step strategies differ | Có mô hình dự báo từng ngày rồi lấy kết quả đó đoán tiếp (lỗi dồn lại), có mô hình dự báo cả đoạn một lần. |
| single train/test split | Chỉ thử 1 lần nên kết quả có thể do may. |
| rarely tested for significance | Không kiểm định thống kê nên chênh lệch nhỏ cũng bị coi là thắng. |

**Đoạn 3: Dữ liệu của bài này khó ở đâu**

- Doanh thu có mùa vụ theo năm và nhịp trong tháng.
- Khuyến mãi lặp lại vào ngày cố định, có đợt chỉ diễn ra năm lẻ.
- Doanh thu tụt khoảng 40% từ năm 2019.
- Có 4 chuỗi (tổng + 3 vùng), nên phải chọn dự báo tổng trực tiếp hay cộng từ 3 vùng.

**Đoạn 4: Bài này trả lời 4 câu hỏi**

| Câu hỏi | Tiếng Việt |
|---|---|
| (i) which model family is most accurate | Nhóm mô hình nào chính xác nhất? |
| (ii) how much of the accuracy comes from calendar information | Độ chính xác đến từ thông tin lịch nhiều cỡ nào? |
| (iii) whether forecast combination improves on the best single model | Ghép nhiều mô hình có tốt hơn mô hình tốt nhất không? |
| (iv) whether the total is better forecast directly or bottom-up | Dự báo tổng trực tiếp hay cộng từ 3 vùng thì tốt hơn? |

**Đoạn 5: Đóng góp (contributions)**

1. Một cuộc so sánh có kiểm soát; luật thi được viết và khóa trước khi chạy.
2. Mô hình cây tốt nhất và rẻ nhất, giảm 23% sai số.
3. RNN/LSTM ngang ngửa ở vòng loại nhưng không thắng ở chung kết.
4. Chỉ ra điều gì quyết định độ chính xác: thông tin lịch và mốc 2019.

**Khi trình bày nói:** "Các nghiên cứu trước kết luận khác nhau vì luật thi khác nhau. Nhóm em làm một cuộc so sánh công bằng để trả lời 4 câu hỏi."

---

## II. Related Work (Các nghiên cứu liên quan)

| Mục | Nghĩa là gì |
|---|---|
| A. Statistical models | ARIMA đoán dựa trên giá trị quá khứ của chính chuỗi. Với dữ liệu ngày, mùa vụ 365 ngày quá dài nên phải dùng sóng sin/cos (Fourier) thay thế. Prophet (của Facebook) tách xu hướng, mùa vụ, ngày lễ. |
| B. Machine learning | Random Forest lấy trung bình nhiều cây quyết định. Gradient Boosting xây cây sau sửa lỗi cây trước; XGBoost và LightGBM là bản nhanh, mạnh của nó. Các đội đứng đầu cuộc thi M5 (dữ liệu Walmart) chủ yếu dùng mô hình cây. Cây không tự hiểu thời gian nên cần biến lịch. |
| C. Deep learning | RNN đọc chuỗi từng bước; LSTM thêm "cổng" để nhớ lâu hơn. Độ chính xác phụ thuộc nhiều vào cách thiết kế. Một bài nổi tiếng cho thấy mô hình tuyến tính đơn giản thắng Transformer khi dự báo dài hạn. |
| D. Forecast combination | Ghép dự báo của nhiều mô hình thường giúp chính xác hơn; lấy trung bình đơn giản thường khó bị đánh bại. |
| E. Evaluation | Nên đánh giá kiểu "cuốn chiếu" theo thời gian (rolling-origin). Nên báo nhiều thước đo. Dùng kiểm định DM và Friedman–Nemenyi để biết chênh lệch là thật hay do may. |
| F. Research gap | Ít bài nào làm đủ cùng lúc: cùng đầu vào, dự báo xa, đánh giá nhiều lượt, kiểm định thống kê, và ablation. Bài này làm đủ. |

**Khi trình bày nói:** "Nghiên cứu trước đã dùng cả 3 nhóm mô hình, nhưng ít bài nào so chúng trong cùng điều kiện và có kiểm định thống kê. Đó là khoảng trống bài này lấp vào."

---

## III. Methodology (Phương pháp)

### A. Data (Dữ liệu)

- Dữ liệu đơn hàng của một cửa hàng thời trang online ở Việt Nam, từ cuộc thi DATATHON 2026 (VinTelligence, VinUni).
- 646.945 đơn, 2.412 sản phẩm, 4 nhóm hàng.
- Ban tổ chức không ghi đơn vị tiền, nên paper giữ nguyên đơn vị trong dữ liệu.
- Doanh thu ngày = số lượng × đơn giá. Giá vốn (COGS) ngày = số lượng × giá nhập.
- 3 vùng: Đông 46,5%, Trung 30,1%, Tây 23,4% doanh thu. Cộng 3 vùng ra đúng con số tổng.
- 3.833 ngày, từ 4/7/2012 đến 31/12/2022, không thiếu ngày nào.
- 6 đợt khuyến mãi lặp lại hằng năm, 2 đợt chỉ có năm lẻ.
- Cuối 2018 doanh thu tụt 39% và ở mức thấp luôn từ đó.

### B. Forecasting task (Bài toán dự báo)

> Each model is fitted on y_1,…,y_T and then produces ŷ_{T+1},…,ŷ_{T+H} in a single pass.

Mô hình học dữ liệu đến ngày T rồi dự báo một lèo H ngày sau đó, không được xem số thật giữa chừng. H = 365 ngày ở vòng loại, 549 ngày ở chung kết. Mô hình học trên log doanh thu (để số đỡ chênh lệch lớn) rồi đổi ngược lại.

### C. Covariates (Biến đầu vào)

Toàn là thông tin **biết trước được** ở ngày dự báo:

| Nhóm | Ví dụ |
|---|---|
| Mùa vụ năm | sóng sin/cos theo ngày trong năm, tháng |
| Nhịp trong tháng (dom) | còn bao nhiêu ngày tới cuối tháng, có phải mấy ngày đầu/cuối tháng không |
| Khuyến mãi (promo) | hôm nay có khuyến mãi nào không, khuyến mãi đã chạy được mấy ngày, có phải năm lẻ không |
| Chế độ (regime) | có phải sau năm 2019 không |

Có 2 luật chống "nhìn trộm tương lai": lịch khuyến mãi tương lai chỉ được suy từ các đợt đã xảy ra; cờ 2019 chỉ được dùng khi đã thấy đủ 180 ngày sau 2019.

### D. Models (Các mô hình)

- **Mốc so sánh:** SNaive-1Y (copy cùng ngày năm trước), SNaive-2Y (copy 2 năm trước, giữ được nhịp năm lẻ).
- **Thống kê:** ARIMA, SARIMA, SARIMAX, Prophet.
- **Máy học:** Random Forest, Gradient Boosting, XGBoost, LightGBM. Dự báo trực tiếp từ biến lịch, không dùng doanh thu hôm trước nên lỗi không dồn.
- **Học sâu:** RNN, LSTM. Đọc 56 ngày biến lịch gần nhất. Chạy 5 lần với 5 hạt giống ngẫu nhiên (seed) rồi lấy trung bình.
- **Ablation (thí nghiệm bỏ bớt):**
  - Bản "pure": bỏ hết biến lịch, dự báo từng ngày nối tiếp, để xem biến lịch đáng giá bao nhiêu.
  - Bỏ từng nhóm biến (regime, promo, dom) để xem nhóm nào quan trọng.

### E. Hyperparameter tuning (Chỉnh tham số)

Mỗi mô hình thử tối đa 12 cấu hình, chọn cấu hình sai ít nhất trên vòng loại. **Không bao giờ dùng tập test để chọn.**

### F. Evaluation protocol (Luật chấm)

- **Vòng loại (CV):** 4 lượt, mốc 30/6 các năm 2017, 2018, 2019, 2020. Mỗi lượt chấm 365 ngày sau mốc.
- **Chung kết (test):** mốc 30/6/2021, chấm 549 ngày (1/7/2021–31/12/2022), **chỉ chạy 1 lần**.
- **Thước đo:**
  - MAE: trung bình mỗi ngày lệch bao nhiêu tiền.
  - RMSE: giống MAE nhưng phạt nặng những ngày lệch nhiều.
  - MAPE: lệch trung bình bao nhiêu phần trăm.
- **Kiểm định:**
  - Friedman–Nemenyi (vòng loại): xếp hạng nhiều mô hình, xem nhóm nào thật sự khác nhau.
  - Diebold–Mariano (chung kết): so từng cặp mô hình, xem chênh lệch có phải do may không.

### G. Forecast combination (Ghép mô hình)

3 cách ghép: trung bình 3 mô hình tốt nhất; trung bình có trọng số theo độ chính xác; stacking (máy tự học trọng số). Trọng số chỉ học từ các lượt trước đó, không nhìn trước.

### H. Multi-region và COGS

- So dự báo tổng trực tiếp với cộng 3 vùng (bottom-up).
- So dự báo giá vốn trực tiếp với lấy doanh thu dự báo nhân tỉ lệ giá vốn.

### I. Chọn phương pháp đề xuất

Luật viết trước: mô hình nào sai ít nhất ở vòng loại thì được chọn. Luật này chọn ra **RNN**, và đã ghi lại trước khi mở tập test.

**Khi trình bày nói:** "Mọi mô hình được cho cùng thông tin biết trước, dự báo một lần tới 18 tháng. Nhóm em chọn mô hình ở vòng loại 4 lượt, rồi chấm chung kết 1 lần trên 549 ngày giấu kín, có kiểm định thống kê."

---

## IV. Experiments and Results (Kết quả)

### Vòng loại (CV), MAE trung bình 4 chuỗi, đơn vị nghìn

| Mô hình | MAE |
|---|---|
| RNN | **508** (thấp nhất) |
| XGBoost | 512 |
| LSTM | 513 |
| LightGBM | 514 |
| Gradient Boosting | 517 |
| Prophet / SARIMAX | 536–537 |
| Random Forest | 567 |
| Copy năm trước (SNaive-1Y) | 698 |
| SARIMA / ARIMA | 846 / 1.454 (còn tệ hơn copy năm trước) |

- Kiểm định cho thấy 8 mô hình có biến lịch **không tách được nhau**, tức là coi như hòa.
- RNN chỉ thắng nhờ lấy trung bình 5 seed. Từng seed riêng lẻ thì sai 522, tệ hơn XGBoost.
- Ở lượt gần chung kết nhất (F4), XGBoost (316) và LightGBM (317) đã thắng RNN (338). Đây là dấu hiệu báo trước.

### Chung kết (test), MAE trung bình 4 chuỗi, đơn vị nghìn

| Mô hình | MAE |
|---|---|
| XGBoost | **325** (MAPE 29,2%) |
| Gradient Boosting | 326 |
| LightGBM | 327 |
| Stacking | 328 |
| RNN | 337 (kém XGBoost 3,7%) |
| LSTM | 348 |
| Copy năm trước | 423 |

- XGBoost sai ít hơn copy năm trước **23%**.
- Kiểm định DM: XGBoost, GB, LightGBM, Stacking không thua mô hình tốt nhất (coi như hòa hạng nhất).
- RNN thua có ý nghĩa ở 3 trên 4 chuỗi; vùng Tây p = 0,064 nên chưa đủ để kết luận thua.
- Mô hình dự báo trực tiếp không bị sai nhiều hơn khi dự báo xa.
- Tháng 8/2021 có khuyến mãi năm lẻ: cách copy năm trước sai tới 893 nghìn, còn các mô hình có biến lịch chỉ sai 263–348.

### Giá trị của thông tin lịch (ablation)

- Bỏ biến lịch đi thì sai tăng mạnh: có lịch giúp giảm sai số 21–84%.
- RNN/LSTM hưởng lợi nhiều nhất (65–84%), mô hình cây 24–36%.
- **Cờ 2019 quan trọng nhất:** bỏ nó, LightGBM sai gấp đôi.
- **Khuyến mãi:** bỏ đi thì sai số tổng chỉ đổi ≤ 2%, vì khuyến mãi rơi vào ngày cố định nên biến mùa vụ đã "biết" rồi. Nhưng riêng tháng 8/2021, bỏ khuyến mãi làm LightGBM sai từ 295 lên 445.

### Ghép mô hình

- Ghép giúp không chọn nhầm mô hình dở, nhưng **không vượt** được XGBoost.

### Nhiều vùng và giá vốn

- Cộng 3 vùng lại tốt hơn dự báo tổng trực tiếp 1,3–4%.
- Dự báo giá vốn trực tiếp tốt hơn dùng tỉ lệ 0,5–4%.

### Discussion (Thảo luận): 3 phát hiện

1. **Mô hình cây** với biến lịch là lựa chọn chính xác và rẻ nhất. Khớp với cuộc thi M5 và bài của Hobor. Các bài mà học sâu thắng dùng dữ liệu và luật khác nên không so trực tiếp được.
2. **Cách đưa thông tin vào mô hình** quan trọng hơn chọn mô hình nào.
3. **Xếp hạng giữa các mô hình sát nhau không ổn định.** Hơn 1% ở vòng loại thì không giữ được ở chung kết, nên phải kiểm định thống kê.

### Limitations (Hạn chế)

- Chỉ 1 cửa hàng, chỉ 1 lần test.
- Biến đầu vào và mốc 2019 lấy từ một phân tích cũ có chạm vào giai đoạn test, nên test không hoàn toàn "chưa thấy".
- Chỉ chỉnh tham số trên chuỗi tổng doanh thu rồi dùng cho tất cả.
- Bản "pure" không được chỉnh tham số.
- Cách tính kiểm định DM được sửa sau khi đã thấy kết quả test (để tính đúng việc lỗi các ngày liền nhau liên quan nhau). Việc sửa làm 1 so sánh của RNN (vùng Tây) từ "thua có ý nghĩa" thành "chưa đủ kết luận". Paper ghi rõ điều này.

**Khi trình bày nói:** "RNN thắng vòng loại sát nút nhưng thua chung kết. XGBoost tốt nhất, giảm 23% sai số. Thông tin lịch, nhất là mốc 2019, là thứ quyết định độ chính xác."

---

## V. Conclusion (Kết luận)

**3 khuyến nghị cho doanh nghiệp:**

| Câu trong paper | Nghĩa là gì |
|---|---|
| use a gradient-boosted tree model with calendar, promotion and regime covariates as a strong and inexpensive default | Dùng XGBoost/LightGBM kèm biến lịch làm lựa chọn mặc định: tốt và rẻ. |
| invest in encoding what is known about the calendar and about structural changes before investing in more complex models | Lo đưa thông tin lịch và các thay đổi lớn (như 2019) vào trước, rồi mới nghĩ tới mô hình phức tạp. |
| treat small differences in cross-validation accuracy as ties unless they are confirmed by significance tests | Chênh lệch nhỏ lúc thử thì coi như hòa, trừ khi kiểm định thống kê xác nhận. |

**Hướng tiếp theo:**
- Làm lại với dữ liệu cửa hàng khác và nhiều lần test.
- Tự phát hiện các mốc thay đổi thay vì cài sẵn năm 2019.
- Thử mô hình học chung cả 3 vùng một lúc.

**Khi trình bày nói:** "Với dữ liệu bán lẻ này, mô hình cây kèm thông tin lịch là đủ tốt và rẻ. Hướng tiếp theo là thử trên dữ liệu của các cửa hàng khác."

---

## Câu hội đồng hay hỏi

| Câu hỏi | Gợi ý trả lời |
|---|---|
| CV chọn RNN mà test lại thua, vậy chọn sai à? | Luật chọn đã cố định trước nên phải theo. Báo cáo trung thực việc RNN thua chính là một phát hiện: chênh lệch nhỏ ở vòng loại không đáng tin. |
| Sao không chạy lại test khi thấy RNN thua? | Chạy lại là "nhìn trộm đề", kết quả mất khách quan. Test chỉ được chạy 1 lần. |
| Sao không chỉnh tham số nhiều hơn? | Mỗi mô hình có lưới tối đa 12 cấu hình, cố định trước, để công bằng giữa các mô hình. |
| Kiểm định DM là gì? | Kiểm tra xem 2 mô hình khác nhau thật hay chỉ do may. p < 0,05 thì coi là khác nhau thật. |
| Đơn vị tiền là gì? | Ban tổ chức Datathon không công bố, nên paper giữ nguyên đơn vị dữ liệu gốc. |
| Áp dụng cho công ty khác được không? | Chưa chắc, vì mới thử 1 cửa hàng. Đây là hạn chế đã ghi trong paper và là hướng làm tiếp. |
| Tại sao dùng MAPE mà không chỉ MAE? | MAE cho biết lệch bao nhiêu tiền, MAPE cho biết lệch bao nhiêu phần trăm, dễ so giữa các vùng có quy mô khác nhau. |
