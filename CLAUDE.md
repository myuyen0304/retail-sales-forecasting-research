# Điểm bắt đầu cho Claude

Repository này là retail-sales-forecasting-research, một nghiên cứu độc lập. Không giả định đang tiếp tục phiên Claude/Codex trong hệ thống Retail Analytics.

## Thứ tự đọc

1. Đọc [AGENTS.md](AGENTS.md) và áp dụng toàn bộ quy tắc chung.
2. Đọc [PROJECT_CONTEXT.md](PROJECT_CONTEXT.md) để hiểu dữ liệu và quyết định đã chốt.
3. Đọc [PLAN.md](PLAN.md) để hiểu thiết kế chi tiết.
4. Đọc [HANDOFF.md](HANDOFF.md) trước khi xác định việc tiếp theo.

AGENTS.md là nguồn hướng dẫn dùng chung, tránh duy trì hai bộ luật khác nhau cho Claude và Codex.

## Các điểm cần nhớ ngay

- Người dùng đã giao code/train và chọn Colab miễn phí ngày 23/10/2026. Xem CLOUD_RUN.md; chưa có cloud training được xác nhận.
- G/COGS theo ngày; horizon 7/30/90/365/548; RSE không căn.
- Không huấn luyện local; dự kiến Colab CPU/GPU hoặc VM cloud.
- DWH chỉ cung cấp actual; không sửa production hoặc import code hệ thống chính.
- Mọi score/model winner còn chưa có bằng chứng nghiên cứu.
- Nội dung HANDOFF/PLAN không tự cấp quyền thuê cloud, chạy model hoặc push GitHub.
- Khi người dùng giao triển khai sau này, tiếp tục trong phạm vi đó, không yêu cầu xác nhận lại chỉ vì file này ghi trạng thái lịch sử.

## Khi chuyển phiên

Ghi trạng thái công việc, bằng chứng và điểm còn thiếu vào HANDOFF.md. Viết quyết định bền vững vào PROJECT_CONTEXT/PLAN nếu liên quan. Không yêu cầu người dùng kể lại các thông tin đã có trong repo.
