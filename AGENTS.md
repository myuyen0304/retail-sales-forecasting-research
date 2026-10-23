# Hướng dẫn làm việc trong repository nghiên cứu

## 1. Đọc trước khi làm

1. [PROJECT_CONTEXT.md](PROJECT_CONTEXT.md): mục tiêu, quyết định và hợp đồng dữ liệu.
2. [PLAN.md](PLAN.md): thiết kế nghiên cứu chi tiết.
3. [HANDOFF.md](HANDOFF.md): trạng thái mới nhất và bước tiếp theo.
4. [README.md](README.md): bản đồ tài liệu.

Không giả định có ngữ cảnh cuộc trò chuyện cũ, memory cá nhân hoặc checkout hệ thống nguồn. Các tài liệu trên là điểm bắt đầu chung cho Codex và Claude.

## 2. Phạm vi hiện tại

Cập nhật 23/10/2026: người dùng đã giao bắt đầu triển khai và huấn luyện, chọn Colab miễn phí. Được viết notebook, adapter và test; training chỉ trên cloud. Chưa có ngân sách trả phí.

- Được triển khai pilot cloud theo yêu cầu mới, giữ code và dữ liệu độc lập với hệ thống chính.
- Được viết code/notebook cho pilot. Export chỉ đọc sau khi chọn nguồn; cài thư viện và train trên Colab. Không thuê compute trả phí hoặc chuyển sang train local.
- Khi người dùng giao triển khai rõ ràng sau này, thực hiện phần đã giao; không hỏi lại cùng một quyền đã được cấp.
- PLAN là công việc dự kiến, không phải lệnh tự động thực hiện mọi bước.
- Ngân sách tài nguyên trả phí phải có mức trần trước khi phát sinh chi phí.

## 3. Ranh giới repository

- Repo này độc lập với hệ thống Retail Analytics; không sửa hệ thống chính để phục vụ nghiên cứu.
- Không import module, venv, submodule hoặc symlink từ hệ thống chính.
- Databricks chỉ là nguồn actual qua kết nối chỉ đọc khi được giao export.
- Không write-back vào DWH production hoặc tự đưa model nghiên cứu vào app/pipeline.
- Không train local. CPU/GPU cho training đều trên cloud; local có thể dùng cho chỉnh tài liệu và kiểm tra nhẹ khi cần.
- Không tự chọn workspace/profile/release từ memory. Xác minh thông tin kết nối trước khi dùng.

## 4. Quy tắc dữ liệu và bằng chứng

- G là doanh thu gộp gồm đơn hủy, chưa trừ chiết khấu; không nhầm với R hoặc tiền thực nhận.
- Không train/chấm trên sample_submission, is_actual = false hoặc prediction của model cũ.
- Không shuffle chuỗi thời gian; không để nhãn/feature sau origin lọt vào train.
- Chỉ fit scaler/selection/decomposition trên train đúng fold.
- COGS có thể vượt G; không áp ràng buộc ngược.
- RSE là Relative Squared Error không căn; không đổi metric hoặc split âm thầm.
- Mỗi phát biểu kết quả phải gắn với artifact thực nghiệm. Tách rõ đọc code, kiểm dữ liệu, chạy model và kiểm cloud.
- Chỉ số, thứ hạng và thời gian trong kế hoạch là thiết kế/dự toán; không trình bày thành kết quả đã đo.

## 5. Ngôn ngữ và thao tác

- Viết phản hồi, tài liệu và comment bằng tiếng Việt; định danh code bằng tiếng Anh.
- Trước khi sửa, kiểm git status, đọc file hiện có; không ghi đè công việc của người dùng.
- Thay đổi đúng phạm vi yêu cầu. Chỉ stage file được nêu tên khi được giao commit.
- Không commit dữ liệu, secrets, token, checkpoint hoặc output nặng.
- Không tự push, tạo PR hoặc publish khi người dùng mới chỉ giao soạn tài liệu.
- Không tự tạo sub-agent; chỉ dùng khi người dùng hoặc hướng dẫn áp dụng yêu cầu rõ.
- Không sao chép AGENTS/CLAUDE của hệ thống chính vào đây vì quy trình production khác nghiên cứu.
- Không biến một thiếu sót kỹ thuật có thể tự tra cứu thành câu hỏi bắt người dùng cung cấp lại.
- Nếu quyền sandbox chặn ghi file, nói rõ giới hạn môi trường; không coi đó là thiếu ủy quyền của người dùng.

## 6. Kiểm tra và bàn giao

Với thay đổi tài liệu: kiểm link tương đối, tính nhất quán quyết định, git diff --check và phạm vi diff. Không chạy model hoặc toàn bộ test hệ thống nguồn để kiểm một file Markdown.

Khi kết thúc công việc đã được giao:
- Cập nhật HANDOFF.md với ngày, việc đã làm, bằng chứng, phần chưa làm và bước tiếp.
- Chỉ sửa PLAN khi thiết kế thay đổi; chỉ sửa PROJECT_CONTEXT khi quyết định/hợp đồng thay đổi.
- Nêu file thay đổi, cách kiểm tra, trạng thái commit/push.
- Không lưu kiến thức bắt buộc chỉ trong memory riêng của agent.

Yêu cầu mới nhất của người dùng có ưu tiên hơn các mặc định trong file này.
