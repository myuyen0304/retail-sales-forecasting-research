"""Retail sales forecasting research."""
import sys

# Console Windows mặc định cp1252 không in được tiếng Việt
for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass
