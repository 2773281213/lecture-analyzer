"""设置页：模型档案（型号、接口地址、密钥、测试连接）、各步骤用哪个模型、转写方案和本地模型、
资料识别并发数和报告语言、费用统计。第一次使用时（?welcome=1）页面顶部有填写说明。

- 密钥只显示"已设置 / 来源"，从不回显；保存时存进 Windows 凭据管理器（config_edit.save_key）；
- config.toml 的修改保留注释（config_edit.py）；改完重新读取配置，后台任务的子进程每次都重新读取；
- 取值不对时在页面顶部提示，不改动任何文件。
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from starlette.concurrency import run_in_threadpool

from .. import config_edit
from ..asr.models import QWEN3_REPOS, cache_root, model_status
from ..config import AppConfig
from ..credentials import SOURCE_LABELS, delete_secret, secret_source
from ..llm import ModelProfile
from ..llm.probe import ping
from .forms import enqueue, flash, int_field, library_of, read_form, redirect
from .manage_courses import LOCAL_ASR

router = APIRouter()
SETTINGS_FORM_BYTES = 8 * 1024
MAX_KEY_CHARS = 400


def settings_view(cfg: AppConfig) -> dict[str, Any]:
    profiles = [_profile_row(name, profile) for name, profile in cfg.models.items()]
    steps = [(step, label, cfg.steps.get(step, ""), _usable(cfg, step)) for step, label in config_edit.STEP_LABELS.items()]
    return {
        "profiles": profiles, "steps": steps, "asr": cfg.asr, "ASR_BACKENDS": config_edit.ASR_BACKENDS,
        "LOCAL_ASR": LOCAL_ASR, "audio_profiles": [name for name, p in cfg.models.items() if p.audio],
        "funasr_presets": _funasr_presets(), "qwen3_sizes": list(QWEN3_REPOS),
        "local_models": {spec: _download_state(cfg, spec) for spec in LOCAL_ASR}, "model_root": cache_root(cfg),
        "env_keys": config_edit.env_keys(cfg), "home": cfg.root, "data_dir": cfg.data_dir,
        "digest_workers": cfg.materials.digest_workers,
    }


def _funasr_presets() -> list[str]:
    from ..asr.funasr_backend import PRESETS

    return list(PRESETS)


def _download_state(cfg: AppConfig, spec: str) -> dict[str, int]:
    status = model_status(cfg, spec)
    return {"missing": sum(not done for _, done in status), "total": len(status)}


def _profile_row(name: str, profile: ModelProfile) -> dict[str, Any]:
    source = secret_source(profile.api_key_env) if profile.api_key_env else "none"
    return {
        "name": name, "adapter": profile.adapter, "model": profile.model, "base_url": profile.resolved_base_url() or "",
        "model_place": f".env 的 {profile.model_env}" if profile.model_env else "config.toml",
        "url_place": f".env 的 {profile.base_url_env}" if profile.base_url_env else "config.toml",
        "key_env": profile.api_key_env, "key_source": SOURCE_LABELS.get(source, "不需要密钥"),
        "key_set": source in ("credential", "env"), "key_in_credential": source == "credential",
        "ready": profile.is_configured and (not profile.api_key_env or source != ""),
        "vision": profile.vision, "audio": profile.audio,
    }


def _usable(cfg: AppConfig, step: str) -> list[str]:
    """能用于这一步的档案（具备需要的能力）。"""
    names = []
    for name in cfg.models:
        try:
            cfg.profile_for_step(step, name)
        except ValueError:
            continue
        names.append(name)
    return names


@router.get("/settings", response_class=HTMLResponse)
def settings_page(request: Request, welcome: int = 0) -> HTMLResponse:
    with library_of(request) as library:
        costs = library.cost_summary()
    context = {**settings_view(request.app.state.cfg), "costs": costs, "welcome": bool(welcome),
               "total_cost": sum(row["cost_usd"] or 0 for row in costs)}
    return request.app.state.templates.TemplateResponse(request, "settings.html", context)


def _apply(request: Request, change: Callable[[AppConfig], tuple[AppConfig | None, str]]) -> None:
    """执行一项修改：成功时换上新配置并提示；取值不对（ConfigError、CredentialError 等）时提示原因。"""
    try:
        cfg, message = change(request.app.state.cfg)
    except ValueError as exc:
        flash(request, str(exc), "error")
        return
    if cfg is not None:
        request.app.state.cfg = cfg
    flash(request, message)


def _profile_name(request: Request, name: str) -> str:
    if name not in request.app.state.cfg.models:
        raise HTTPException(404, "没有这个模型档案")
    return name


@router.post("/settings/models/{name}")
async def save_profile(request: Request, name: str) -> RedirectResponse:
    form = await read_form(request, SETTINGS_FORM_BYTES)
    name = _profile_name(request, name)

    def change(cfg: AppConfig) -> tuple[AppConfig, str]:
        cfg = config_edit.set_profile_model(cfg, name, form.get("model", ""))
        return config_edit.set_profile_base_url(cfg, name, form.get("base_url", "")), f"已保存「{name}」的型号和接口地址"

    _apply(request, change)
    return redirect(f"/settings#model-{name}")


@router.post("/settings/models/{name}/key")
async def save_key(request: Request, name: str) -> RedirectResponse:
    form = await read_form(request, SETTINGS_FORM_BYTES)
    profile = request.app.state.cfg.profile(_profile_name(request, name))
    value = form.get("key", "")
    if len(value) > MAX_KEY_CHARS:
        raise HTTPException(400, "密钥过长")

    def change(cfg: AppConfig) -> tuple[None, str]:
        config_edit.save_key(cfg, profile.api_key_env, value)
        return None, f"已把 {profile.api_key_env} 保存到 Windows 凭据管理器"

    _apply(request, change)
    return redirect(f"/settings#model-{name}")


@router.post("/settings/models/{name}/key/delete")
async def delete_key(request: Request, name: str) -> RedirectResponse:
    await read_form(request)
    profile = request.app.state.cfg.profile(_profile_name(request, name))

    def change(cfg: AppConfig) -> tuple[None, str]:
        delete_secret(profile.api_key_env)
        return None, f"已从凭据管理器删除 {profile.api_key_env}"

    _apply(request, change)
    return redirect(f"/settings#model-{name}")


@router.post("/settings/models/{name}/test")
async def test_profile(request: Request, name: str) -> RedirectResponse:
    await read_form(request)
    profile = request.app.state.cfg.profile(_profile_name(request, name))
    if not profile.is_configured:
        flash(request, f"「{name}」还没有填写型号", "error")
        return redirect(f"/settings#model-{name}")
    ok, text = await run_in_threadpool(ping, profile)
    flash(request, f"「{name}」（{profile.model}）{'连接成功' if ok else '连接失败'}：{text}", "ok" if ok else "error")
    return redirect(f"/settings#model-{name}")


@router.post("/settings/keys/move")
async def move_keys(request: Request) -> RedirectResponse:
    await read_form(request)

    def change(cfg: AppConfig) -> tuple[None, str]:
        moved = config_edit.move_env_keys(cfg)
        return None, ("已移到凭据管理器：" + "、".join(moved)) if moved else ".env 里没有需要移动的密钥"

    _apply(request, change)
    return redirect("/settings#models")


@router.post("/settings/steps")
async def save_steps(request: Request) -> RedirectResponse:
    form = await read_form(request, SETTINGS_FORM_BYTES)
    steps = {step: form[step] for step in config_edit.STEP_LABELS if form.get(step)}
    _apply(request, lambda cfg: (config_edit.set_steps(cfg, steps), "已保存各步骤使用的模型"))
    return redirect("/settings#steps")


@router.post("/settings/asr")
async def save_asr(request: Request) -> RedirectResponse:
    form = await read_form(request, SETTINGS_FORM_BYTES)
    keys = ("backend", "funasr_preset", "qwen3_size", "device", "model_dir", "llm_profile")
    values = {key: form[key] for key in keys if key in form}
    _apply(request, lambda cfg: (config_edit.set_asr(cfg, values), "已保存转写设置"))
    return redirect("/settings#asr")


@router.post("/settings/general")
async def save_general(request: Request) -> RedirectResponse:
    form = await read_form(request)
    workers = int_field(form, "digest_workers", "并发数") or 1
    _apply(request, lambda cfg: (config_edit.set_general(cfg, workers), "已保存"))
    return redirect("/settings#general")


@router.post("/settings/models-download")
async def download_models(request: Request) -> RedirectResponse:
    form = await read_form(request)
    spec = form.get("spec", "")
    if spec not in LOCAL_ASR:
        raise HTTPException(400, "未知的转写方案")
    with library_of(request) as library:
        job_id = enqueue(request, library, None, "download_models", {"spec": spec})
    return redirect(f"/jobs/{job_id}")
