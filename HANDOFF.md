# Bàn giao hiện tại — pilot Colab CPU

Cập nhật 23/10/2026 sau yêu cầu bắt đầu huấn luyện và lựa chọn Colab miễn phí.

- Đã chuẩn bị cloud_pilot.py, export_snapshot.py, configs/pilot_cpu.json, requirements-cloud.txt, tests/test_protocol.py và notebooks/01_colab_cpu_pilot.ipynb.
- Pilot: 6 baseline + Ridge + LightGBM + log-SARIMAX; G/COGS; F5, H=30/548; 36 đơn vị đánh giá development. Chưa đủ 40 model và chưa có phần GPU.
- Notebook tự chứa code/config, có thể Upload notebook vào Colab mà không push GitHub.
- Chưa export snapshot, chưa có training local/cloud và chưa có score thực nghiệm mới.
- Đã kiểm databricks auth profiles ngoài sandbox: retail-dev tồn tại nhưng Valid=NO. Chưa dùng profile này để truy vấn actual hoặc chạy compute.
- Phiên trợ lý không có công cụ điều khiển Colab. Người dùng đã chọn mở notebook và bấm chạy. Cần catalog/release đã chọn và thông tin kết nối nguồn trong Colab; không gửi secret vào chat.
- Đã commit riêng từng file và push lên GitHub main ngày 23/10/2026; đợt đầu kết thúc ở a25fc96. Kiểm tra Git log/remote để biết trạng thái mới nhất.
- Hướng dẫn chạy và giới hạn: CLOUD_RUN.md.
- Các mục bên dưới là bản bàn giao lịch sử trước yêu cầu triển khai; trạng thái “chỉ kế hoạch/chưa code” đã được thay bởi mục hiện tại này.

---

# Bàn giao cho phiên Codex/Claude tiếp theo

Cập nhật: 23/10/2026. Phạm vi bàn giao: tài liệu nghiên cứu, chưa triển khai mô hình.

## 1. Đọc nhanh

- Repo GitHub: https://github.com/myuyen0304/retail-sales-forecasting-research.
- Thư mục local: D:/retail-sales-forecasting-research.
- Nghiên cứu độc lập, không nằm trong D:/retail-analytics.
- Chỉ train trên cloud; local không dùng cho huấn luyện.
- Người dùng đang yêu cầu kế hoạch và chuyển ngữ cảnh sang repo mới.
- Không có số MAE/MAPE/RSE mới hoặc model thắng.
- Mở phiên mới: đọc AGENTS.md, PROJECT_CONTEXT.md, PLAN.md và file này.

## 2. Những việc đã hoàn thành

| Việc | Bằng chứng/trạng thái |
|---|---|
| Chốt phạm vi paper với người dùng | G/COGS theo ngày; năm horizon; survey + benchmark; RSE không căn; 30–40 model thực nghiệm |
| Soạn kế hoạch | PLAN.md, 14 mục; thiết kế 40 ứng viên chính, khoảng 80 phương pháp/biến thể survey |
| Chuyển compute sang cloud | PLAN §3 và §9; không train local |
| Tách repo | Clone repo do người dùng tạo, origin đúng URL phía trên |
| Bỏ thư mục tạo nhầm tên | D:/retail-sales-forecast-survey đã xóa theo yêu cầu; kế hoạch được giữ lại |
| Bỏ bản kế hoạch trong hệ thống chính | File docs/ke_hoach_paper_survey_du_bao_sales.md đã chuyển ra khỏi repo nguồn |
| Bàn giao bối cảnh | PROJECT_CONTEXT.md |
| Hướng dẫn hai agent | AGENTS.md cho quy tắc chung; CLAUDE.md trỏ tới cùng bộ tài liệu |
| Điểm vào cho người đọc | README.md |

Base commit của repo nghiên cứu tại lúc soạn bàn giao: b9043ea39ecc76f12ea94a66c3d8b490fced8d8b. Các tài liệu bàn giao là thay đổi local chưa commit/push tại thời điểm soạn; không suy ra GitHub đã có các file này. Phiên sau cần kiểm git status/log/remote lại.

## 3. Chưa làm

- Chưa viết code model, adapter, feature pipeline, metric runner hoặc notebook training.
- Chưa cài môi trường nghiên cứu và khóa dependency.
- Chưa truy vấn Databricks live trong công việc nghiên cứu.
- Chưa chọn active release, export hoặc đối soát snapshot research.
- Chưa tạo Drive/object storage cho artifact.
- Chưa thuê compute, mua quota hoặc chạy Colab.
- Chưa benchmark/tuning/ablation/final test.
- Chưa viết phần Results có số thực nghiệm.
- Chưa commit/push bộ bàn giao.

Không dùng việc đọc SQL nguồn để thay cho bằng chứng dữ liệu cloud; không dùng kết quả production cũ thay cho score nghiên cứu.

## 4. Quyết định và thiết kế

### Người dùng đã chốt

- Nghiên cứu riêng, repository riêng.
- Mục tiêu G và COGS theo ngày.
- Horizon 7, 30, 90, 365, 548 ngày.
- Survey rộng, thực nghiệm 30–40 mô hình.
- RSE = Relative Squared Error.
- Không train local.
- Hiện tại chỉ tài liệu/kế hoạch.

### Thiết kế do trợ lý đề xuất trong PLAN

- 40 ứng viên screening, khoảng 80 ứng viên survey.
- Development đến 01/07/2021; test 02/07/2021–31/12/2022.
- Năm rolling-origin folds, 2.000 đơn vị đánh giá screening.
- Điểm chính từng bài toán là MAE, chọn finalist theo macro MASE.
- Tám finalist, tuning giới hạn, năm seed cho finalist ngẫu nhiên.
- Colab CPU/GPU mặc định; VM cloud dự phòng.
- Tám tuần là lộ trình ước lượng.

Chưa đóng băng protocol thực nghiệm. Nếu điều chỉnh thiết kế, ghi lý do trước khi xem điểm; không đổi target hoặc mục tiêu người dùng âm thầm.

## 5. Thông tin còn thiếu trước triển khai cloud

| Thông tin | Cách giải quyết khi được giao triển khai |
|---|---|
| Ngân sách và thời hạn | Hỏi người dùng mức trần chi phí/hạn nộp; không mặc định miễn phí đủ toàn bộ |
| Workspace/warehouse/profile đọc dữ liệu | Xác minh với lựa chọn hiện tại của người dùng, không đoán từ repo nguồn |
| Quyền đọc và release | Kiểm đọc đúng registry/relation; ghi nhận release cố định |
| Nơi lưu snapshot/artifact | Chọn Drive hoặc object storage theo tài khoản đã có và ràng buộc dữ liệu |
| Runtime thực tế | Pilot CPU/GPU cloud, ghi thiết bị và bộ nhớ được cấp |
| Venue/ngôn ngữ paper | Chưa quyết; tài liệu làm việc hiện bằng tiếng Việt |
| Quyền chia sẻ dataset/artifact | Xác định trước công bố; không mặc định quyền public do có repo GitHub |

Không cần những thông tin này để tiếp tục sửa tài liệu; chỉ cần trước phần công việc phụ thuộc vào chúng.

## 6. Các chi tiết kỹ thuật cần cụ thể hóa trước benchmark

PLAN đủ làm định hướng nhưng chưa có config chạy. Trước khi triển khai và khóa protocol cần bổ sung trong repo nghiên cứu:
- Search space cụ thể, default hyperparameters, context/lookback, số bước/epoch và early stopping từng model.
- Quy tắc dùng train nội bộ cho early stopping không ăn vào validation chấm điểm.
- Timeout, giới hạn bộ nhớ và số trial theo dự toán pilot.
- Chính sách xử lý model không hoàn thành đủ nhiệm vụ hoặc thiếu lịch sử cho lag dài.
- Point forecast mean/median và cách đảo log cho từng adapter.
- Giới hạn horizon của checkpoint; chiến lược recursive blocks tương ứng.
- Tính khả thi của 40 model × nhiều horizon/seed trên quota cloud.
- Seed, package lock, hardware metadata và cơ chế lưu artifact khi mất phiên.
- Kiểm lại lịch split và tính đầy đủ của actual sau khi export.

Không tự mở code/training để lấp các chi tiết này khi người dùng vẫn chỉ yêu cầu kế hoạch.

## 7. Bẫy cần tránh khi tiếp quản

1. Đọc fact_daily_sales mà quên is_actual sẽ trộn nhãn giả từ sample_submission.
2. Dùng actual tương lai trong lag/rolling hoặc recursive prediction làm rò rỉ.
3. Dùng R thay G làm thay đổi bài toán.
4. Áp COGS <= G làm sai dữ liệu.
5. Gọi test mới là hoàn toàn chưa từng được nhìn thấy bỏ qua EDA lịch sử.
6. Tính t-test thông thường trên fold dài chồng lấn.
7. Xếp hạng runtime khi mỗi model chạy GPU khác nhau mà không công khai.
8. Gọi baseline/model production cũ là winner của paper.
9. Đưa nguyên AGENTS/skills production sang research khiến agent tự sửa dbt/pipeline.
10. Gọi repository đã có PLAN là đã có code/model hoặc đã chạy nghiên cứu.

## 8. Bước tiếp theo theo yêu cầu người dùng

Nếu yêu cầu là tiếp tục kế hoạch:
- Rà tính khả thi, làm rõ search space và dự toán cloud ở mức tài liệu.
- Cập nhật PLAN và ghi thay đổi; không triển khai model.

Nếu yêu cầu sau này là bắt đầu triển khai:
- Xác định phạm vi và ngân sách đã được giao.
- Dựng công cụ trong repo này, bắt đầu contract/split/metric và pilot cloud.
- Xác minh nguồn dữ liệu chỉ đọc trước export.
- Hoàn thành pilot và khóa config trước screening.
- Chỉ commit/push khi được giao; không tự mở rộng sang hệ thống chính.

## 9. Mẫu lời mở đầu cho phiên mới

> Đọc AGENTS.md, PROJECT_CONTEXT.md, PLAN.md và HANDOFF.md. Đây là repo nghiên cứu độc lập, không train local. Hiện chỉ hoàn thiện kế hoạch, chưa triển khai code mô hình. Hãy kiểm trạng thái Git trước khi sửa và tiếp tục công việc tôi giao dưới đây.

Có thể thay câu cuối bằng một yêu cầu triển khai cụ thể nếu người dùng thực sự muốn chuyển giai đoạn.

## 10. Quy tắc cập nhật bàn giao

Mỗi lần hoàn thành công việc, bổ sung ngày, phạm vi, file/artifact, cách kiểm, việc còn thiếu và trạng thái commit/push. Đánh dấu rõ thông tin lịch sử có thể đã thay đổi. Không ghi token, password, nội dung secret hoặc khẳng định cloud chưa được kiểm.

## Kiểm tra trước khi bàn giao pilot (23/10/2026)

Đã dùng parser kiểm cú pháp 3 module Python và 10 cell code của notebook; nội dung các file nhúng khớp file repo; cấu hình có đúng 36 task. Đây chỉ là kiểm tra tĩnh trên máy soạn code: không import module train, không fit model, không truy vấn dữ liệu. Bộ test protocol chưa được thực thi; notebook sẽ chạy bộ test trên Colab trước export/training. Chưa kiểm chứng cài dependency hoặc tương thích runtime cloud thực tế.
