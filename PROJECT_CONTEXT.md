# Bối cảnh dự án nghiên cứu dự báo sales

Ngày bàn giao: 23/10/2026. Tài liệu này giúp bắt đầu phiên mới mà không cần lịch sử chat hoặc checkout hệ thống nguồn.

## 1. Dự án này là gì?

Repository: https://github.com/myuyen0304/retail-sales-forecasting-research
Checkout hiện tại: D:/retail-sales-forecasting-research.

Đây là nghiên cứu độc lập theo hướng survey + empirical benchmark cho dự báo doanh thu gộp (G) và giá vốn (COGS) theo ngày. Người dùng muốn tổng quan nhiều họ mô hình, thực nghiệm có hệ thống, so sánh RSE/MAE/MAPE, hiệu suất tính toán và vẽ biểu đồ train/test/cross-validation.

Trạng thái hiện tại: đã chuẩn bị code/notebook pilot CPU trên Colab; chưa có cloud training được xác nhận. [PLAN.md](PLAN.md) là thiết kế đầy đủ; [HANDOFF.md](HANDOFF.md) ghi tiến độ và điểm cần xử lý tiếp.

## 2. Các yêu cầu người dùng đã nói rõ

| Nội dung | Quyết định |
|---|---|
| Loại công trình | Survey kèm benchmark thực nghiệm |
| Target | G và COGS, tổng toàn hệ thống theo ngày |
| Horizon | 7, 30, 90, 365, 548 ngày |
| Độ rộng | Survey rộng, chạy 30–40 ứng viên; PLAN chọn thiết kế 40 thực nghiệm và khoảng 80 phương pháp/biến thể tổng quan |
| RSE | Relative Squared Error, không lấy căn |
| Compute | Không huấn luyện local; dùng CPU/GPU cloud |
| Tách biệt | Repository riêng, không ghép code model vào hệ thống chính |
| Việc được giao hiện tại | Bắt đầu triển khai và train trên Colab miễn phí; không train local |
| Ngôn ngữ | Giải thích và tài liệu tiếng Việt; định danh code tiếng Anh |

Colab CPU/GPU là phương án đề xuất mặc định, VM cloud là dự phòng. Chưa chốt ngân sách, cấu hình thiết bị, hạn nộp, nơi gửi paper hoặc cloud storage thực tế. Đây không phải sự cho phép tự thuê tài nguyên.

Ngày 22/10/2026 người dùng đã nhấn mạnh chỉ tạo plan thành file Markdown. Ngày 23/10/2026 người dùng yêu cầu repo riêng, chỉ định repo GitHub đã tạo, rồi yêu cầu bàn giao tài liệu cho Codex/Claude. Sau bàn giao, người dùng đã giao bắt đầu huấn luyện cloud và chọn Colab miễn phí. Yêu cầu mới mở giai đoạn triển khai; chưa cho phép thuê compute trả phí.

## 3. Hệ thống nguồn chỉ cung cấp dữ liệu

Hệ thống nguồn là Retail Analytics, một DWH phân tích doanh thu có dbt, reporting, ứng dụng BI, chat và pipeline công bố dữ liệu theo release trên Databricks.

Các thành phần đó là bối cảnh để hiểu dữ liệu. Repository nghiên cứu không cần app, chat, Airflow, module dự báo production hay môi trường Python của hệ thống nguồn để chạy.

Ranh giới:
- Đọc actual của một release đã công bố, đóng băng thành snapshot riêng.
- Sau export, mọi thí nghiệm đọc snapshot; không gọi DWH trong từng fold.
- Không ghi bảng nghiên cứu, prediction hoặc cấu hình vào production.
- Không import code từ checkout hệ thống chính, không submodule/symlink tới hệ thống đó.
- Không copy dữ liệu giao dịch khách hàng, secrets hoặc toàn bộ 14 CSV nếu chỉ cần hai chuỗi tổng hợp.
- Không tự deploy model thắng vào sản phẩm.
- Có thể yêu cầu chủ nguồn xác nhận hợp đồng dữ liệu khi cần; nghiên cứu phải ghi lại câu trả lời trong repo riêng.

## 4. Định nghĩa target

| Target | Định nghĩa từ hệ thống nguồn | Lưu ý |
|---|---|---|
| G / Revenue | Tổng quantity × unit_price theo ngày đơn hàng | Gồm đơn hủy; không trừ discount_amount |
| COGS | Tổng quantity × cogs theo ngày đơn hàng | Model aggregate hiện tại làm tròn tới hai chữ số thập phân |
| R | Doanh thu đơn giao thành công dùng ở một số trang BI nguồn | Không phải target G; không thay thế lẫn nhau |

G/COGS không phải dòng tiền thu thực tế. Không dùng payment_date, shipment_date hay doanh thu sau giảm giá để thay định nghĩa.

Grain nguồn dự kiến: một ngày với hai cột G và COGS. Grain nghiên cứu dạng dài: một target trong một ngày, khóa (unique_id, ds), unique_id thuộc G/COGS.

Lịch sử theo tài liệu nguồn:
- 04/07/2012–31/12/2022: 3.833 ngày, dự kiến không thiếu ngày.
- 01/01/2023–01/07/2024: 548 ngày cần dự báo của bài toán gốc.
- Chưa có actual 2023–2024 được xác nhận cho nghiên cứu này.

Các con số trên là kỳ vọng từ tài liệu nguồn, cần kiểm khi export cloud. Chưa chạy truy vấn live để xác nhận snapshot nghiên cứu.

## 5. Bẫy quan trọng: fact có cả actual và dòng giữ chỗ

SQL nguồn được đọc trực tiếp ngày 23/10/2026 cho thấy:
- fact_daily_sales tổng hợp actual từ fact_order_item, gắn is_actual = true.
- Bảng này UNION ALL dữ liệu sample_submission và gắn is_actual = false.
- sample_submission chỉ là mẫu định dạng, không phải ground truth.
- Không dùng prediction production hoặc placeholder để train, validation hoặc tính test accuracy.

Nguồn ưu tiên là reporting.rpt_forecast_actual trong schema vật lý đúng release. Model này đã lọc is_actual:
- dim_date.full_date → sale_date.
- fact_daily_sales.revenue → g.
- fact_daily_sales.cogs → cogs.
- Join qua date_sk, điều kiện fact_daily_sales.is_actual.

Tên có chữ forecast nhưng rpt_forecast_actual chứa actual. Không nhầm với rpt_forecast_daily hoặc rpt_forecast_explain. Model reporting actual được bật có điều kiện theo forecast_path khi build nguồn, nên không giả định mọi release đều có bảng.

Nếu bảng reporting thiếu, có thể đề xuất query tương đương từ marts của cùng release khi có quyền; không bỏ qua sự thiếu quyền hoặc đổi sang nguồn khác âm thầm.

## 6. Release và snapshot

Cấu trúc resolver nguồn đã đọc ngày 23/10/2026:
- Registry logic: retail_control.state, bản ghi name = main, cột payload dạng JSON.
- State chứa active và releases.
- Record hợp lệ có release_id, status = ready, schema dạng r_<release_id>_reporting.
- Đây là contract trong code nguồn, chưa xác minh trạng thái live.
- Catalog, hostname, endpoint, profile và active release của nghiên cứu chưa được chốt. Không lấy thông tin cũ trong chat hoặc memory làm cấu hình mặc định.

Khi được giao triển khai export:
1. Xác minh kết nối chỉ đọc đúng workspace do người dùng chọn.
2. Resolve một release hợp lệ và ghi cố định schema vật lý.
3. Export actual đúng release; không đổi con trỏ active giữa các câu query.
4. Ghi SQL, source relation, release_id, thời điểm export, số dòng, ngày đầu/cuối, tổng theo target và checksum.
5. Lưu snapshot tại cloud storage riêng; kiểm lại checksum khi dùng.
6. Dừng export nếu contract không đúng. Lỗi quyền đọc không phải lý do để cấp thêm quyền ghi.

Snapshot chuẩn có ds, unique_id, y, release_id, snapshot_id. Kiểm null/Inf, trùng ngày, thiếu ngày, zero/negative, đối soát tổng ngày/tháng. Không tự xóa outlier hoặc fill ngày thiếu bằng 0.

## 7. Những hiểu biết EDA được kế thừa

Bảng này là tóm tắt tài liệu đã có ở hệ thống nguồn. Các notebook chứng minh chưa được chạy lại trong repo nghiên cứu; chưa được coi là kết quả thực nghiệm mới của paper.

| Hiểu biết từ nguồn | Hệ quả cần cân nhắc |
|---|---|
| Chuỗi đổi chế độ cuối 2018 | So sánh train window; không ngoại suy một CAGR chung toàn lịch sử |
| Mùa vụ tháng mạnh | Có baseline mùa vụ và thử biến lịch/Fourier |
| Hiệu ứng ngày trong tháng | Ablation feature ngày trong tháng |
| EDA cũ thấy weekday yếu | Không tự hứa weekday vô ích trên mọi split; ghi nguồn hiểu biết và đánh giá nếu đưa vào ablation |
| Tháng 8 năm lẻ có khác biệt gắn với Urban Blowout | Giữ seasonal naive hai năm; báo riêng sai số kỳ này |
| Có ngày COGS > G | Không ép giá vốn nhỏ hơn doanh thu |
| Các bảng phụ dừng ở 2022 | Actual tương lai của traffic/inventory/reviews không phải feature sẵn có |

Cờ mùa Urban Blowout là kiến thức đã được suy ra trước nghiên cứu, không phải lịch tương lai đã được xác nhận độc lập. Đưa vào nhánh F2 có nhãn rõ ràng; không trình bày như phát hiện trên test.

Mối liên hệ số học của G với các bảng con không có nghĩa được dùng giao dịch tương lai làm feature. Chỉ dùng thông tin có tại forecast origin.

COGS qua tỷ lệ là một ứng viên để so với dự báo trực tiếp, không phải luật bắt buộc hoặc mô hình đã thắng.

Baseline cũ của hệ thống nguồn có cảnh báo về ngoại suy xu hướng qua đứt gãy 2018. Không lấy notebook baseline cũ làm nền code nghiên cứu.

## 8. Thiết kế nghiên cứu hiện được đề xuất

Các lựa chọn kỹ thuật bên dưới do trợ lý thiết kế trong PLAN, chưa phải protocol đã thực thi hoặc đăng ký với một hội nghị.

- Development: 04/07/2012–01/07/2021.
- Final test: 02/07/2021–31/12/2022, 548 ngày.
- CV: 5 rolling origins, theo PLAN §6; không random K-fold.
- Screening: 40 × 2 target × 5 horizon × 5 fold = 2.000 đơn vị đánh giá.
- Finalist: 2 thống kê, 3 ML, 2 DL, 1 foundation; giữ baseline seasonal để đối chứng.
- RSE = sum((y − yhat)^2) / sum((y − mean(y))^2), mean tính trên tập được chấm.
- Bảng chính dùng MAE; điểm chọn finalist dùng trung bình MASE theo PLAN.
- Foundation models chạy zero-shot ở benchmark chính.
- Tuning, early stopping và scaling phải nằm trong development/train tương ứng.
- Sau khi khóa protocol mới chấm final test; không chọn seed bằng test.

Dữ liệu đã được khám phá trong hệ thống nguồn. Vì vậy paper là retrospective benchmark; không khẳng định final test chưa từng được con người nhìn thấy. Fold dài chồng lấn nên không mặc định coi chúng độc lập.

Nguyên tắc metric: MAPE không xác định với actual = 0; RSE/MASE mẫu số 0 cũng không xác định. Ghi rõ coverage/NA, không sửa bằng epsilon tùy ý. Đảo log và scaler về đơn vị gốc trước khi chấm. Phân biệt sai số ngày với sai số tổng tháng.

## 9. Tính độc lập với quyết định production cũ

Hệ thống nguồn từng chọn model theo mục tiêu/giao thức riêng. Kết quả đó không phải kết quả của paper này và không được dùng làm model thắng mặc định.

Không mang sang:
- Luật publish/rollback, điều kiện deployment hay is_primary production.
- Bộ ba fold cũ hoặc điểm kết hợp MAE G/COGS để thay protocol nghiên cứu.
- Test pass, live-LLM smoke hoặc trạng thái app làm bằng chứng model research.
- Deadline sản phẩm cũ làm deadline paper.
- Quy tắc bắt buộc sửa notebook/dbt nguồn mỗi lần thay đổi nghiên cứu.

Tất cả kết quả research phải được tạo từ snapshot, split, config và prediction của repo này.

## 10. Bản đồ bằng chứng

Nguồn được đọc trực tiếp ngày 23/10/2026 từ working tree hệ thống nguồn; HEAD lúc đọc là ed83193fcabd9149bb04f1d6cd3cc02e07cac8c9. Working tree có thay đổi khác nên HEAD chỉ là mốc định hướng, không khẳng định toàn bộ file đọc đều đã commit tại SHA đó.

| Nguồn lịch sử | Dùng để hiểu |
|---|---|
| CLAUDE.md §1, §5, §6 | Định nghĩa dữ liệu, khoảng lịch sử, EDA và lỗi baseline |
| retail_dbt/models/marts/fact_daily_sales.sql | Actual cộng placeholder, công thức aggregate |
| retail_dbt/models/reporting/rpt_forecast_actual.sql | Bảng actual và điều kiện lọc |
| apps/retail_app/dwh/releases.py | Registry, release pinning và schema |
| PLAN.md trong repo này | Thiết kế nghiên cứu độc lập |

Các đường dẫn nguồn trong bảng là provenance văn bản, không phải đường dẫn bắt buộc tồn tại trên máy người tiếp nhận. Không cần clone hệ thống chính để bắt đầu đọc hiểu hoặc viết kế hoạch.

Chưa có: kết nối Databricks research hợp lệ, snapshot research, môi trường cloud đã kiểm chứng, run benchmark hoặc score/winner mới. Code pilot đã chuẩn bị; xem CLOUD_RUN.md.

## 11. Cách dùng tài liệu khi mở phiên mới

Đọc theo thứ tự: AGENTS.md → PROJECT_CONTEXT.md → PLAN.md → HANDOFF.md. Claude đọc thêm CLAUDE.md để nhận cùng hướng dẫn. Chỉ dùng file trong repo này cho ngữ cảnh mặc định.

Nếu phát hiện mâu thuẫn, ưu tiên yêu cầu mới nhất của người dùng; cập nhật quyết định vào tài liệu liên quan. Nếu chỉ thiếu một thông tin vận hành, xử lý các phần độc lập trước rồi hỏi đúng thông tin thiếu, không bắt người dùng kể lại cả dự án.
