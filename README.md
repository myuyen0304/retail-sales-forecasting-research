# retail-sales-forecasting-research

Nghiên cứu độc lập về dự báo **doanh thu gộp G và giá vốn COGS theo ngày**, theo hướng survey và benchmark nhiều họ mô hình.

**Trạng thái ngày 23/10/2026:** đã chuẩn bị pilot CPU trên Colab miễn phí. Chưa có lượt train cloud được xác nhận. Xem [CLOUD_RUN.md](CLOUD_RUN.md) và [notebook Colab](notebooks/01_colab_cpu_pilot.ipynb).

## Bắt đầu đọc

| File | Vai trò |
|---|---|
| [PROJECT_CONTEXT.md](PROJECT_CONTEXT.md) | Bối cảnh, quyết định, target, dữ liệu và giới hạn bằng chứng |
| [PLAN.md](PLAN.md) | Kế hoạch chi tiết: survey, 40 model, split/CV, metric, cloud, biểu đồ |
| [HANDOFF.md](HANDOFF.md) | Việc đã làm, chưa làm, điểm còn thiếu và bước tiếp |
| [AGENTS.md](AGENTS.md) | Quy tắc làm việc chung cho coding agent |
| [CLAUDE.md](CLAUDE.md) | Điểm vào cho Claude, sử dụng cùng quy tắc AGENTS.md |

## Dành cho phiên Codex hoặc Claude mới

Đọc AGENTS.md → PROJECT_CONTEXT.md → PLAN.md → HANDOFF.md trước khi sửa. Claude đọc thêm CLAUDE.md. Không cần cuộc trò chuyện cũ hoặc checkout hệ thống chính để hiểu dự án.

Người dùng đã giao triển khai và huấn luyện trên Colab miễn phí. Chưa có ngân sách thuê tài nguyên; không train local.

## Phạm vi nghiên cứu

- Hai target G/COGS theo ngày; horizon 7, 30, 90, 365, 548 ngày.
- Khoảng 80 phương pháp/biến thể survey; thiết kế 40 ứng viên thực nghiệm.
- Metric bắt buộc MAE, MAPE, Relative Squared Error (RSE không căn).
- Đánh giá theo thời gian; có so sánh chi phí tính toán và biểu đồ train/test/CV.
- Colab CPU/GPU hoặc VM cloud; không train trên máy cá nhân.

## Ranh giới dữ liệu và hệ thống

Databricks chỉ cung cấp snapshot actual qua kết nối chỉ đọc khi triển khai. Code, môi trường và artifact nghiên cứu thuộc repo này; không import code hệ thống Retail Analytics hoặc ghi kết quả vào production.

Dữ liệu, secrets và checkpoint không đưa vào Git. Notebook Colab tự chứa bộ chạy pilot; không cần push repo để tải notebook lên và chạy.

Repository: https://github.com/myuyen0304/retail-sales-forecasting-research
