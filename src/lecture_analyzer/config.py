"""配置加载：config.toml（路径、转写、步骤分配、模型档案）+ .env（密钥）。"""

from __future__ import annotations

import os
import tomllib
from pathlib import Path
from typing import Literal

from dotenv import dotenv_values
from pydantic import BaseModel, ValidationError

from .llm.base import Effort, ModelProfile

CONFIG_ENV = "LEC_CONFIG"
DEFAULT_CONFIG_NAME = "config.toml"

# 各步骤对模型能力的要求
STEP_REQUIREMENTS: dict[str, tuple[str, ...]] = {
    "page_digest": ("vision",),
}


class ConfigError(ValueError):
    """配置缺失或不一致。"""


class PathsConfig(BaseModel):
    data_dir: Path = Path("data")


class AsrConfig(BaseModel):
    backend: Literal["auto", "funasr", "qwen3", "llm"] = "auto"  # auto：按课程的授课语言选本地模型
    funasr_preset: str = "paraformer"
    qwen3_size: str = "1.7b"
    model_hub: Literal["modelscope", "huggingface"] = "modelscope"
    device: str = "auto"  # auto：有可用的 NVIDIA 显卡就用 cuda:0，否则用 CPU；也可以写死 cuda:0 / cpu
    model_dir: str = ""  # 本地模型的下载位置（相对配置文件所在目录）；不填用 ModelScope 默认的缓存目录
    llm_profile: str = "gemini"
    chunk_minutes: float = 10
    llm_alternates: list[str] = []  # 切到最短仍被拦截的片段依次换用的型号（同一档案、同一接口）
    fallback: str = "auto"  # 换型号、对半切之后仍被拦截的片段用什么补转：auto / none / 具体方案


class MemoryConfig(BaseModel):
    """课程记忆与总复习的输入预算（token）；实际预算还受所选模型的上下文上限约束。"""

    integrate_budget: int = 60_000
    recent_lectures: int = 2  # 整合时带入最近几讲的笔记摘要
    review_budget: int = 60_000


class MaterialsConfig(BaseModel):
    """课程资料（课件、教材、试题……）的处理。"""

    digest_workers: int = 4  # 逐页识别时同时发出的请求数
    digest_batch: int = 6  # 每次请求识别几页


class ScoringConfig(BaseModel):
    """考点打分的权重；signals、materials 只写要改的项，其余用默认值（见 pipeline/scoring.py）。"""

    signals: dict[str, float] = {}  # 课堂信号类型 → 权重（乘以强度）
    materials: dict[str, float] = {}  # 资料类型 → 权重（乘以分值系数）
    per_extra_lecture: float = 2  # 每多在一讲出现
    per_10_minutes: float = 1  # 每讲 10 分钟
    max_minutes_bonus: float = 3
    review_lecture: float = 3  # 在复习课里出现
    class_and_material: float = 3  # 课堂和资料里都出现
    max_score_factor: float = 2  # 题目分值系数的上限
    question_type_fit: float = 6  # 题型匹配：权重 × 知识点适合的各题型在考试方案里的分值占比之和（最多算 1）


class ExamPlanConfig(BaseModel):
    """考试方案（先于知识点分析）。分析来源讲的转写时只送和考试有关的部分：开头若干分钟，加上提到考试、题型、
    分值等关键词的句子前后各若干分钟。"""

    auto_confirm: bool = False  # 分析出方案后直接采用，不停下来等用户确认
    lead_minutes: float = 40
    window_minutes: float = 2
    budget: int = 40_000  # 分析请求的输入上限（token），还受模型上下文约束


class ReportConfig(BaseModel):
    """考前报告一律用中文写（M4 修订）；旧配置里的 language、disagreement_gap 两项读取时忽略。"""

    top_n: int = 15  # 每个模型排出前几个考点
    max_models: int = 4  # 一次报告最多用几个模型
    budget: int = 80_000  # 每次请求的输入上限（token），还受模型上下文约束


# 新增的步骤在 config.toml 里没写时的默认档案和推理强度，老配置不用改也能用；
# 课件匹配（match）没写时与页面识别（page_digest）用同一个档案，问答检索（qa_retrieve）没写时与课件匹配相同
DEFAULT_STEPS = {"integrate": "claude", "review": "claude", "material": "claude", "report": "claude", "qa": "claude"}
DEFAULT_EFFORT: dict[str, Effort] = {
    "integrate": "medium", "review": "medium", "match": "low", "material": "medium", "report": "high",
    "qa_retrieve": "low", "qa": "medium", "exam_plan": "medium",
}


class AppConfig(BaseModel):
    root: Path = Path(".")  # 配置文件所在目录，相对路径以它为基准
    source: Path | None = None  # 读取的配置文件（后台任务的子进程用它重新读取配置）
    paths: PathsConfig = PathsConfig()
    asr: AsrConfig = AsrConfig()
    memory: MemoryConfig = MemoryConfig()
    materials: MaterialsConfig = MaterialsConfig()
    scoring: ScoringConfig = ScoringConfig()
    report: ReportConfig = ReportConfig()
    exam_plan: ExamPlanConfig = ExamPlanConfig()
    models: dict[str, ModelProfile] = {}
    steps: dict[str, str] = {}
    effort: dict[str, Effort] = {}

    @property
    def data_dir(self) -> Path:
        path = self.paths.data_dir
        return path if path.is_absolute() else (self.root / path).resolve()

    @property
    def model_dir(self) -> Path | None:
        if not self.asr.model_dir.strip():
            return None
        path = Path(self.asr.model_dir.strip())
        return path if path.is_absolute() else (self.root / path).resolve()

    def profile(self, name: str) -> ModelProfile:
        try:
            return self.models[name]
        except KeyError:
            raise ConfigError(f"config.toml 里没有名为「{name}」的模型档案") from None

    def profile_for_step(self, step: str, override: str | None = None) -> ModelProfile:
        """取某一步要用的模型档案（命令行 --model 优先），并检查能力是否满足。"""
        name = override or self.steps.get(step)
        if not name:
            raise ConfigError(f"[steps] 没有为「{step}」指定模型档案")
        profile = self.profile(name)
        missing = _missing_capabilities(step, profile)
        if missing:
            raise ConfigError(f"步骤「{step}」需要模型能力 {', '.join(missing)}，档案「{name}」不具备")
        return profile

    def effort_for_step(self, step: str) -> Effort | None:
        return self.effort.get(step)

    def check(self) -> list[str]:
        """不发请求，只检查配置本身的问题。"""
        problems: list[str] = []
        for step, name in self.steps.items():
            profile = self.models.get(name)
            if profile is None:
                problems.append(f"步骤「{step}」指向的档案「{name}」不存在")
                continue
            if not profile.is_configured:
                problems.append(f"步骤「{step}」使用的档案「{name}」还没有填写 model")
            missing = _missing_capabilities(step, profile)
            if missing:
                problems.append(f"步骤「{step}」需要 {', '.join(missing)}，档案「{name}」不具备")
        if self.asr.backend == "llm":
            profile = self.models.get(self.asr.llm_profile)
            if profile is None:
                problems.append(f"[asr] llm_profile 指向的档案「{self.asr.llm_profile}」不存在")
            elif not profile.audio:
                problems.append(f"[asr] llm_profile 档案「{profile.name}」不支持音频输入（audio = false）")
        return problems


def _missing_capabilities(step: str, profile: ModelProfile) -> list[str]:
    return [cap for cap in STEP_REQUIREMENTS.get(step, ()) if not getattr(profile, cap)]


def find_config_path(explicit: Path | None = None) -> Path:
    """查找顺序：命令行参数 > 环境变量 LEC_CONFIG > 当前目录的 config.toml > 桌面程序选定的书库文件夹。"""
    from .apphome import saved_home

    env_value = os.environ.get(CONFIG_ENV)
    home = saved_home()
    candidates = (explicit, Path(env_value) if env_value else None, Path(DEFAULT_CONFIG_NAME),
                  home / DEFAULT_CONFIG_NAME if home else None)
    for candidate in candidates:
        if candidate is not None and candidate.is_file():
            return candidate.resolve()
    raise ConfigError("找不到配置文件：请把 config.example.toml 复制为 config.toml，或用 --config 指定；"
                      "安装版请先打开「课程书架」完成首次设置")


def load_env_file(path: Path) -> None:
    """读取 .env：填了值的项覆盖进程里已有的同名环境变量，空项忽略。

    项目配置优先：例如在 Claude Code 里运行时，它自己设置的 ANTHROPIC_BASE_URL（官方地址）
    不应盖过 .env 里的中转地址和中转密钥；而 .env 里留空的项也不应清掉已有的有效值。
    """
    for key, value in dotenv_values(path).items():
        if value:
            os.environ[key] = value


def _apply_model_env(profile: ModelProfile) -> ModelProfile:
    """档案设置了 model_env 且 .env 里填了型号时，用它覆盖 config.toml 里的 model。"""
    value = os.environ.get(profile.model_env, "").strip() if profile.model_env else ""
    return profile.model_copy(update={"model": value}) if value else profile


def load_config(path: Path | None = None) -> AppConfig:
    config_path = find_config_path(path)
    load_env_file(config_path.parent / ".env")
    with config_path.open("rb") as f:
        raw = tomllib.load(f)
    try:
        models = {
            name: _apply_model_env(ModelProfile(name=name, **values)) for name, values in raw.get("models", {}).items()
        }
        steps = {**DEFAULT_STEPS, **raw.get("steps", {})}
        steps.setdefault("match", steps.get("page_digest", "gemini"))
        steps.setdefault("qa_retrieve", steps["match"])
        if "extract" in steps:  # 考试方案（exam_plan）没写时与整理笔记用同一个档案
            steps.setdefault("exam_plan", steps["extract"])
        return AppConfig(
            root=config_path.parent,
            source=config_path,
            paths=raw.get("paths", {}),
            asr=raw.get("asr", {}),
            memory=raw.get("memory", {}),
            materials=raw.get("materials", {}),
            scoring=raw.get("scoring", {}),
            report=raw.get("report", {}),
            exam_plan=raw.get("exam_plan", {}),
            models=models,
            steps=steps,
            effort={**DEFAULT_EFFORT, **raw.get("effort", {})},
        )
    except ValidationError as exc:
        raise ConfigError(f"配置文件 {config_path} 有误：\n{exc}") from exc
