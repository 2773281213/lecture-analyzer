"""提示词模板：每个步骤一个 Markdown 文件，占位符写作 {{名称}}。"""

from __future__ import annotations

import re
from importlib import resources

_PLACEHOLDER = re.compile(r"\{\{(\w+)\}\}")


def load_prompt(name: str, **values: str) -> str:
    """读取模板并替换 {{名称}} 占位符；缺少取值时报错，避免把占位符原样发给模型。"""
    template = resources.files(__package__).joinpath(f"{name}.md").read_text(encoding="utf-8").strip()

    def substitute(match: re.Match[str]) -> str:
        key = match.group(1)
        if key not in values:
            raise KeyError(f"提示词 {name} 缺少占位符取值：{key}")
        return values[key]

    return _PLACEHOLDER.sub(substitute, template)
