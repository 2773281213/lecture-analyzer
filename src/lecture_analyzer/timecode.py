"""时间码的格式化与解析（毫秒 ↔ mm:ss / h:mm:ss）。"""

from __future__ import annotations


def format_ms(ms: int) -> str:
    """毫秒转成 mm:ss，超过一小时为 h:mm:ss。"""
    total_seconds = max(ms, 0) // 1000
    hours, rest = divmod(total_seconds, 3600)
    minutes, seconds = divmod(rest, 60)
    if hours:
        return f"{hours}:{minutes:02d}:{seconds:02d}"
    return f"{minutes:02d}:{seconds:02d}"


def parse_timecode(text: str) -> int:
    """把 mm:ss、h:mm:ss 或带小数秒的形式解析成毫秒；格式不对时抛 ValueError。"""
    pieces = text.strip().split(":")
    if not 2 <= len(pieces) <= 3:
        raise ValueError(f"无法解析的时间：{text!r}")
    seconds = float(pieces[-1])
    minutes = int(pieces[-2])
    hours = int(pieces[0]) if len(pieces) == 3 else 0
    if seconds < 0 or minutes < 0 or hours < 0:
        raise ValueError(f"无法解析的时间：{text!r}")
    return int(round((hours * 3600 + minutes * 60 + seconds) * 1000))
