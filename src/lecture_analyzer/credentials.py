"""API 密钥：优先从 Windows 凭据管理器读取（keyring），没有时用环境变量 / .env。

- 按变量名保存（如 ANTHROPIC_API_KEY），服务名固定为 lecture-analyzer；密钥只在本机按当前 Windows 账号加密保存。
- 界面保存密钥时写入凭据管理器，并删掉 .env 里同名的旧值，免得两处不一致。
- 读取失败（例如凭据管理器不可用）时照常回退到环境变量，不报错，也不记录密钥内容。
"""

from __future__ import annotations

import os
import re

import keyring
from keyring.errors import KeyringError, PasswordDeleteError

SERVICE = "lecture-analyzer"
_NAME = re.compile(r"^[A-Z][A-Z0-9_]{1,63}$")
SOURCE_LABELS = {"credential": "凭据管理器", "env": ".env / 环境变量", "": "未设置"}


class CredentialError(ValueError):
    """密钥名称不合法或保存失败；消息是给使用者看的说明。"""


def check_name(name: str) -> str:
    if not _NAME.match(name or ""):
        raise CredentialError(f"密钥的变量名不合法：{name}（只能用大写字母、数字和下划线）")
    return name


def stored(name: str) -> str | None:
    """凭据管理器里保存的值；凭据管理器不可用时当作没有。"""
    try:
        return keyring.get_password(SERVICE, check_name(name)) or None
    except KeyringError:
        return None


def get_secret(name: str) -> str | None:
    """凭据管理器 → 环境变量（含 .env）。"""
    if not name:
        return None
    return stored(name) or os.environ.get(name) or None


def secret_source(name: str) -> str:
    """密钥来自哪里：credential / env / 空（未设置）。只用于界面显示，不返回密钥本身。"""
    if stored(name):
        return "credential"
    return "env" if os.environ.get(name) else ""


def set_secret(name: str, value: str) -> None:
    value = value.strip()
    if not value:
        raise CredentialError("密钥不能为空")
    try:
        keyring.set_password(SERVICE, check_name(name), value)
    except KeyringError as exc:
        raise CredentialError(f"保存到凭据管理器失败：{exc}") from None


def delete_secret(name: str) -> None:
    try:
        keyring.delete_password(SERVICE, check_name(name))
    except PasswordDeleteError:
        pass  # 本来就没有
    except KeyringError as exc:
        raise CredentialError(f"从凭据管理器删除失败：{exc}") from None
