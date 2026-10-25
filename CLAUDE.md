# Điểm vào cho Claude

Đọc và làm theo [AGENTS.md](AGENTS.md). Đây là bộ luật dùng chung cho Claude và Codex.

Nhắc nhanh:
- Python của repo: `.venv/Scripts/python.exe`. Không dùng venv của repo khác.
- `protocol.md` đã đóng băng. Stage `test` chỉ chạy khi người dùng cho phép.
- Số trong paper lấy từ `results/tables/`; hình lấy từ `scripts/make_figures.py`.
- Mở phiên mới thì đọc [HANDOFF.md](HANDOFF.md) trước, rồi làm tiếp từ mục "Chưa làm / bước tiếp". Không thiết kế lại từ đầu.
- Repo này được dựng từ một phiên Claude mở ở `D:\retail-analytics`. Memory của phiên đó không theo sang đây, nên bối cảnh cần giữ đều phải nằm trong file của repo.
- Lệnh nặng: đưa cho người dùng chạy trong PowerShell riêng, không chạy trong Claude Code (Claude Code từng giết run dài vì hết RAM). Mở đầu bằng `$env:PYTHONUTF8=1`, vì console Windows dùng cp1252.
