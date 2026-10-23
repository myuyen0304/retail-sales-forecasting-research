# Chạy pilot CPU trên cloud

Cập nhật 23/10/2026: người dùng đã giao bắt đầu triển khai và huấn luyện cloud. Đã chuẩn bị code pilot; **chưa có lượt train cloud được xác nhận thành công**.

## Phạm vi pilot

- F5, origin 31/12/2019; H=30 và H=548; hai target G/COGS.
- Sáu baseline, Ridge, LightGBM, log-SARIMAX: 36 đơn vị đánh giá development.
- Final test vẫn đóng; chưa triển khai đủ 40 model, DL/GPU hoặc foundation models.
- Tham số được đăng ký trong configs/pilot_cpu.json trước khi có điểm.
- Training bị chặn nếu runner không nhận ra runtime Colab/Databricks.
- Snapshot actual phải đủ contract và có manifest/checksum.
- Lưu model.joblib, predictions.csv, fitted.csv nếu có, plot.png và status.json theo từng task.
- Resume chỉ dùng task success khi checksum artifact khớp; task hỏng/dở sẽ chạy lại.
- metrics.csv và summary.json không tự biến một pilot thành kết quả cuối của paper.
- Thời gian pilot chỉ có fit/predict; chưa có benchmark latency nhiều lần, đo RAM/VRAM hoặc timeout cưỡng chế theo task.
- Chưa khóa khả năng tương thích runtime thực tế; requirements là cấu hình đề xuất cần xác nhận bằng pilot cloud.

## Colab

Mở notebooks/01_colab_cpu_pilot.ipynb qua Upload notebook trên Colab. Notebook tự chứa bản code pilot và config, không cần push repo hay cài project local. Chọn **hosted CPU runtime**, không dùng local runtime. Chọn **Latest runtime / Python 3.13**; notebook dừng sớm nếu runtime khác. Sau khi cài thư viện, nếu Colab yêu cầu restart session thì restart và chạy lại từ đầu.

1. Chạy cell kiểm cloud và cài dependency.
2. Chạy cell ghi module/config trong ổ tạm VM và chạy test không train.
3. Mount Drive để lưu snapshot/artifact bền vững.
4. Điền catalog, release_id đã chọn và thông tin SQL warehouse; token nhập ẩn bằng getpass, không ghi vào file. Nguồn đọc bằng SELECT; việc đọc warehouse có thể dùng quota/tính phí của Databricks.
5. Chạy export lần đầu; lần sau dùng lại snapshot có manifest.
6. Chạy pilot, xem summary/metrics và ảnh.

Nếu chưa có quyền kết nối DWH, chưa chạy được export/training. Không thay actual bằng file mẫu hoặc dữ liệu giả.

Phiên trợ lý hiện không có công cụ điều khiển Colab trực tiếp. Người dùng cần mở notebook và chạy các cell trên tài khoản của mình. Điều này khác với việc trợ lý đã tự chạy model.

## Databricks

Có thể dùng cùng các module trên runtime Databricks khi profile/workspace và compute được chọn, đăng nhập hợp lệ, ngân sách được xác định. Hàm export_actual nhận callback query trả DataFrame; có thể dùng spark.sql(statement).toPandas() trong notebook cloud với quyền SELECT.

Chưa deploy job hoặc upload code vào workspace. Không tự dùng production compute hoặc writer identity.

## Kiểm tra và giới hạn hiện tại

- Test protocol nằm trong tests/test_protocol.py và chạy trong notebook cloud trước export/train.
- Việc kiểm cú pháp ở máy soạn code không phải chứng cứ training hoặc cloud smoke.
- Không có score nào được công bố trước khi lấy artifact từ cloud và đối chiếu manifest.
- Không chia sẻ model.joblib có nguồn không tin cậy; artifact model chỉ được đọc từ run do mình tạo.

## Kiểm tra trước khi bàn giao pilot (23/10/2026)

Đã dùng parser kiểm cú pháp 3 module Python và 10 cell code của notebook; nội dung các file nhúng khớp file repo; cấu hình có đúng 36 task. Đây chỉ là kiểm tra tĩnh trên máy soạn code: không import module train, không fit model, không truy vấn dữ liệu. Bộ test protocol chưa được thực thi; notebook sẽ chạy bộ test trên Colab trước export/training. Chưa kiểm chứng cài dependency hoặc tương thích runtime cloud thực tế.
