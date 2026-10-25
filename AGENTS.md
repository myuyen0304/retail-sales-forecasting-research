# Hướng dẫn cho coding agent (Codex, Claude)

Đọc theo thứ tự: [README.md](README.md) → [protocol.md](protocol.md) → [HANDOFF.md](HANDOFF.md).

## Phạm vi

- Repo độc lập, phục vụ paper NCKH. Không import code, không sửa hệ thống Retail Analytics.
- Việc nặng (`tune` / `cv` / `test` đầy đủ): đưa lệnh cho người dùng tự chạy, trên laptop hoặc Colab. Agent chỉ đọc `results/`.
  Smoke run một fold để kiểm code thì agent được chạy.
- Không tự push, tạo PR hay publish khi chưa được giao.

## Quy tắc khoa học

- `protocol.md` đã đóng băng. Đổi split, biến, lưới tham số hay luật ensemble sau khi đã xem điểm phải ghi vào protocol §9, và kết quả gắn nhãn *exploratory*.
- **Không chạy stage `test`** trước khi người dùng chốt phương pháp đề xuất trên CV.
- Mọi con số trong paper phải có dòng nguồn trong `results/tables/*.csv`. Mọi hình phải sinh từ `scripts/make_figures.py`.
- Tài liệu tham khảo chỉ thêm khi đã kiểm DOI/arXiv. Ghi kết quả đọc được vào `paper/literature_matrix.csv`; phần chưa đọc bản đầy đủ thì ghi rõ.
- Không shuffle chuỗi thời gian. Biến và scaler chỉ fit trên train của đúng fold. Không dùng y sau origin.

## Ngôn ngữ và git

- Tài liệu repo và comment viết tiếng Việt; định danh code tiếng Anh; paper (`paper/`) viết tiếng Anh.
- Commit message: tiếng Việt không dấu, một dòng. Không thêm dòng `Co-Authored-By`.
- `git add` từng file cụ thể. Không commit `data/` hay `results/preds/`.
- Kết thúc một việc: cập nhật [HANDOFF.md](HANDOFF.md) (ngày, việc đã làm, bằng chứng, bước tiếp).
