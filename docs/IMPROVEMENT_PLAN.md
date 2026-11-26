# lecture-analyzer 代码评审与改进方案

> 评审基准分支：`master`（版本 `0.3.5`，提交 `37cb789`）  
> 改进分支：第一轮 `improve/review-2026-11`；第二轮 `improve/round2-2026-11-25`

---

## 一、项目现状概述

`lecture-analyzer` 是一款面向个人本地使用的课程录音分析与考点预测工具（Windows 优先，固定 Python 3.12，`uv` 管理依赖）。整体架构分为六层：

1. **配置、凭据与本地存储**：`config.py` / `config_edit.py`（`tomlkit` 保留注释改写 `config.toml` + `.env`）、`credentials.py`（Windows 凭据管理器）、`library.py` / `library_schema.py`（SQLite WAL 模式 + 按版本备份迁移）、`knowledge.py` / `materials.py` / `reports.py` / `conversations.py` / `exam_plans.py`。
2. **统一大模型接入层（`llm/`）**：抽象 `LLMProvider`（自动修复结构化 JSON、瞬态重试、拒答识别），提供 `anthropic`、`gemini`、`openai_compat` 三个适配器及拒答备用包装 `FallbackProvider`。
3. **语音识别层（`asr/`）**：本地 FunASR（Paraformer / SenseVoice）、Qwen3-ASR 与大模型切片转写（含二分重试与本地补转）。
4. **分析流水线（`pipeline/`）**：抽音频 → 转写 → 课件匹配 → 考试方案关卡 → 笔记抽取 → 原话三级核对 → 跨讲课程记忆整合（L2）→ 总复习讲义 → 多模型 Borda 共识考前报告 → 带附件连续问答。
5. **并发与任务系统**：`locks.py`（进程内 `RLock` + 跨进程文件锁）、`jobs.py` / `job_runner.py` / `worker.py`（独立子进程 + 重/轻双队列，支持暂停、恢复、取消、孤儿任务清理）。
6. **交互入口**：`cli.py` / `cli_materials.py` / `cli_plan.py`（Typer + Rich）、`web/`（FastAPI + Jinja2 + 原生 JS 工作台，CSRF 令牌 + TrustedHostMiddleware）、`desktop.py`（pywebview + 单实例互斥体 + 原生文件对话框）及 `packaging/`（Inno Setup 安装包打包）。

### 运行方式与基线测试观察

- 基础安装：`uv sync`，运行 CLI `uv run lec --help`、网页书架 `uv run lec serve` 或桌面窗口 `uv run lec desktop`。
- 完整本地 ASR 依赖：`uv sync --extra asr`（含约 3GB CUDA PyTorch + FunASR + ModelScope）。
- **基线测试问题（评审前）**：在仅执行 `uv sync`（未安装 3GB `--extra asr`）的环境下直接运行 `uv run pytest -m "not office"` 会出现 **2 个测试文件收集报错**（`test_funasr_backend.py`、`test_qwen3_backend.py` 顶层直接 `import numpy`）和 **5 个用例失败**（包括 `GET /settings` 直接返回 HTTP 500）。

---

## 二、问题清单（按严重程度排序）

> 行号均基于评审基准提交 `37cb789` 的代码位置。

### P0 — 严重缺陷（阻断基础功能或导致死锁 / 崩溃 / 安全风险）

| 编号 | 类别 | 位置 | 问题描述与影响 | 建议修法 |
| --- | --- | --- | --- | --- |
| P0-1 | 正确性 / 依赖 | `src/lecture_analyzer/asr/funasr_backend.py:11`、`asr/vad.py:13`、`asr/__init__.py:105`、`asr/qwen3_backend.py:124` | `funasr_backend.py` 与 `vad.py` 在模块顶层无条件 `import numpy as np`，`resolve_device("auto")` 未捕获 `import torch` 的 `ImportError`，且 `qwen3_backend._language_table()` 在无 `transformers` 时仅兜底 `zh/ko/en` 三种语言。仅做基础 `uv sync` 时，访问 `/settings`（需读取 `PRESETS`）、调用 `config_edit.set_asr`、`asr.models.required_repos` 或触发补转均直接抛出 `ModuleNotFoundError` 导致 500 崩溃，法语等语种的 Qwen3 语种映射也会失效。 | 将 `numpy` 放入 `TYPE_CHECKING` 并在运行期函数内延迟导入（缺失时抛出明确 `ASRError` 提示 `uv sync --extra asr`）；`resolve_device("auto")` 捕获 `ImportError` 回退 `"cpu"`；补全无 `transformers` 时的常用语种兜底表。 |
| P0-2 | 正确性 / 并发 | `src/lecture_analyzer/locks.py:42-45, 76-82` | `_acquire` 在 `entry.guard.acquire()` 后调用 `_lock_file`，若 `_lock_file` 抛异常（锁忙超时或 `OSError`），`entry.guard` 不会被释放，导致同进程内后续线程永久死锁；Windows 下 `_lock_file` 对并发打开的锁文件调用 `path.open("a+b")` 未包进 `OSError` 重试循环，偶发瞬时 `PermissionError` 直接失败。 | `_acquire` 用 `try/except` 在 `_lock_file` 失败时释放 `entry.guard`；`_lock_file` 把 `path.open("a+b")` 纳入重试循环。 |
| P0-3 | 正确性 / 资源回收 | `src/lecture_analyzer/job_runner.py:140-156, 179-189, 240-258` | `_run` 中 `_wait`/`_settle` 位于 `try` 之外，一旦抛未捕获异常（如任务运行中途整门课被删除导致 `store.get(job_id)` 抛 `LibraryError`），子进程不会被终止、`_processes` 条目泄漏且工作线程死亡；`shutdown()` 仅等待队列线程，未 `join` 已退出信号下的 `_threads` 即释放 `_owner_lock`。 | 将 `_wait` / `_settle` 纳入 `try/except BaseException/finally` 确保清理子进程与 `_processes`；`_wait` 捕获 `LibraryError` 终止孤儿子进程；`shutdown()` 释放文件锁前 `join` 所有 `_threads`。 |
| P0-4 | 正确性 / 死锁 | `src/lecture_analyzer/pipeline/organize.py:97-98, 155-160` | `move_lecture` 先锁 `source` 再锁 `target`，两门课互移讲次时存在经典 AB/BA 死锁；`delete_materials` 按调用传入顺序逐个加 `material_lock`，未按 ID 排序且未去重。 | `move_lecture` 固定按 `course.id` 升序加两把 `memory_lock`；`delete_materials` 按 `material.id` 去重排序后加锁。 |
| P0-5 | 安全 / 压缩包解压 | `src/lecture_analyzer/archive.py:114-127, 134-156, 198-206` | `extract_zip` 仅累加 ZIP 目录头声明的 `info.file_size` 判断 `MAX_TOTAL_BYTES`，伪造头部的解压炸弹可绕过限制；未捕获损坏流抛出的 `zlib.error` / `EOFError` / `NotImplementedError` / `RuntimeError`；`expand` 处理 `loose_images` 时未跳过符号链接；`_check_extracted` 未检测 Windows 目录联接（junction）。 | 流式拷贝解压时按**实际写入字节数**累计并拦截超限；统一将解压异常转为 `ArchiveError`；跳过符号链接与 Windows junction。 |

---

### P1 — 高优先级问题（逻辑错误、计费不准、数据残留、校验缺失）

| 编号 | 类别 | 位置 | 问题描述与影响 | 建议修法 |
| --- | --- | --- | --- | --- |
| P1-1 | 正确性 / LLM | `src/lecture_analyzer/llm/registry.py:29-31`、`fallback.py:19-46`、`probe.py:27-35`、`cli.py:105` | `create_provider` 只要档案配置了 `fallback_profile`，就会立刻构造备用模型客户端：即使主模型完全正常，只要备用模型未填密钥就会导致整个步骤启动失败；同时 `probe.ping` 和 `lec models --remote` 未屏蔽 `fallback_profile`，探活时可能触发回退或报错。 | `FallbackProvider` 改为懒加载备用客户端（首次拒答时才初始化）；`probe.ping` 与 `_print_remote_models` 清除 `fallback_profile=None`。 |
| P1-2 | 可靠性 / LLM | `src/lecture_analyzer/llm/gemini_adapter.py:26-34, 65-72`、`llm/base.py:299-307` | `GeminiProvider` 未包裹 `retry_transient`，且 `google.genai.errors.APIError` 的状态码字段叫 `.code` 而非 `.status_code`，导致 `retry_transient` 无法识别 Gemini 503/429，且 SDK 默认内部会做 5 次重试与外层语义不一致。 | `retry_transient` 同时识别 `status_code` 与 `code`；`GeminiProvider` 禁用 SDK 内部重试（`HttpRetryOptions(attempts=1)`）并接入统一 `retry_transient`。 |
| P1-3 | 安全 / 正确性 | `src/lecture_analyzer/llm/anthropic_adapter.py:110-112` | `_use_refusal_fallback` 使用 `OFFICIAL_HOST in base_url` 子串判断，形如 `https://api.anthropic.com.relay.example/v1` 的中转域名会被误判为直连官方接口而错误注入 beta 头。 | 改为 `urlsplit(base_url).hostname` 精确大小写无关相等比较。 |
| P1-4 | 正确性 / 计费 | `src/lecture_analyzer/pipeline/extract.py:62-97`、`slides.py:121-150`、`run.py:239,264`、`exam_plan.py:186`、`llm/openai_compat.py:72-95` | 1）`extract_notes` 与 `match_slides` 丢弃了 `result.profile`，拒答回退到备用模型时 `run.py` 仍按主档案计费和记录 `notes_model`；2）`exam_plan.py:186` 返回的 `PlanAnalysis.model` 误用了 `profile.model` 而非 `used.model`；3）`openai_compat` 遇到空 `choices` 重试时丢弃了第一次调用已产生的 `usage`。 | `ExtractResult` / `MatchResult` 透传 `used_profile` / `fallback_used`，`run.py` 使用 `billed_profile`；`exam_plan.py` 返回 `used.model`；`openai_compat` 累加空选择重试的 `usage`。 |
| P1-5 | 正确性 / 原话定位 | `src/lecture_analyzer/pipeline/exam_plan.py:200-201`、`pipeline/validate.py:41-58` | 考试方案原话通过 `window` / `fuzzy` 核对后，代码调用 `index.locate(...)` 做**全文首次匹配**，当同一短语（如“简答题四道”）在前面出现过时，录音跳转时间会错跳到第一次出现的无关位置。 | 在 `TranscriptIndex` 增加 `locate_in_window(quote, segment_ids)`，仅在当前引用窗口内精确收缩句子范围。 |
| P1-6 | 正确性 / 预算裁剪 | `src/lecture_analyzer/pipeline/memory.py:150-176` | `assemble()` 初始估算忽略段间 `\n\n` 与整段取整误差，且逐条扣减时未计入空段替换成 `empty_text`（可能比被删条目更长）的变化，导致估算漂移后提前 `break`，明明还有可裁剪条目却误报 `BudgetError`。 | 按段精确维护非空条目数与渲染字数（O(1) 更新，含 `\n`、`empty_text` 与 `\n\n`），使循环判断与最终 `estimate_tokens` 完全一致。 |
| P1-7 | 正确性 / 数据清理 | `src/lecture_analyzer/pipeline/organize.py:174-184` | `delete_course` 未删除 `data/qa_attachments/<course_id>/` 目录，也未清理课程/讲/资料对应的 `.lock` 文件，长期留下孤立附件和锁文件。 | 删除课程时一并删除该课程的问答附件目录并 `discard` 全部相关锁文件。 |
| P1-8 | 正确性 / 数据一致性 | `src/lecture_analyzer/web/settings_page.py:113-116`、`config_edit.py:115-163` | 设置页保存档案时先调用 `set_profile_model` 落盘，再调用 `set_profile_base_url` 校验 URL；URL 非法时虽弹出错误提示，但 `model` 已被写入 `.env`/`config.toml` 造成半写入状态；且 `set_env` 直接 `write_text` 非原子写入。 | 新增 `config_edit.set_profile` 先校验两项再落盘；`set_env` 改用同目录临时文件 + `os.replace` 原子替换。 |
| P1-9 | 错误处理 / 安全 | `src/lecture_analyzer/web/exam_plan_page.py:204-214`、`timecode.py:18-33`、`web/rendering.py:32`、`render.py:100`、`web/views.py:476` | 1）`_number` 和 `parse_timecode` 接受 `"nan"`/`"inf"`，后续 `int(number)` 直接触发 HTTP 500；2）`render_markdown` 还原占位符未做下标边界检查，正文含 `\ue00099\ue001` 时抛 `IndexError`；3）`render_references` / `_references` 未检查 `1 <= ref.note_index`，`0` 会负索引命中最后一个知识点。 | 增加 `math.isfinite` 校验；占位符还原加下标范围保护；引用下标改为 `1 <= ref.note_index <= len(...)`。 |
| P1-10 | 正确性 / 附件与图像 | `src/lecture_analyzer/qa_attachments.py:143-157, 179-182, 232-235, 319-330, 353-362`、`documents.py:124-136` | 1）`get_attachment` 未校验 `meta.json` 中的 `suffix` 与字段类型，损坏元文件可能绕过类型白名单或抛未处理异常；2）`_read_pdf` 直接打开磁盘路径文件在 Windows 下可能持锁；3）`_read_pptx` 在文本预算耗尽时误判 `has_text=False`；4）`read_attachments` 在图片预算为 0 时未正确标记截断；5）极窄长图缩放 `round(height * scale)` 可能为 `0` 导致 PyMuPDF 崩溃，且 `_pixmap_png` 与 `documents.py` 存在重复缩放逻辑。 | 抽出公共 `documents.pixmap_to_png` 并以 `max(1, round(...))` 防零宽/高；`get_attachment` 严格校验 `suffix` 与类型；PDF 读入内存打开；修正 PPTX `has_text` 与图片预算截断分支。 |
| P1-11 | 安全 / 协议 | `src/lecture_analyzer/web/forms.py:43-50`、`web/qa_page.py:27-49, 69-70, 93-94, 102-103` | 1）`read_form` 依赖 `request.body()`，对分块传输（无 `Content-Length`）请求会先把整包读入内存才判超限，且与 `qa_page._body_limited` 代码重复；2）`upload_attachment`/`remove_attachment` 在课程不存在时抛出 `LibraryError` 返回 400 HTML 而非注释承诺的 404 JSON；3）附件下载响应缺少 `X-Content-Type-Options: nosniff`。 | 抽取 `forms.read_body_limited` 供两者共用；捕获 `LibraryError` 返回 404 JSON / 404 HTTPException；附件下载补齐 `X-Content-Type-Options: nosniff`。 |

---

### P2 — 中优先级问题（性能、并发一致性、错误处理与前端体验）

| 编号 | 类别 | 位置 | 问题描述与影响 | 建议修法 |
| --- | --- | --- | --- | --- |
| P2-1 | 性能 / N+1 查询 | `src/lecture_analyzer/web/views.py:155-181, 333-339`、`cli_materials.py:76`、`materials.py` | `material_rows` 与 `cli_materials.material_list` 为每份资料调用 `len(store.pages(m.id))` 和 `len(store.items(m.id))`，在 SQLite 层做 N+1 次 `SELECT *` 并全量反序列化正文与题目 JSON；`_kp_chapters` 为每个知识点线性过滤全课 `history`（O(K×S)）。 | 在 `MaterialStore` 新增 `page_counts(course)` / `item_counts(course)` 使用 SQL `COUNT(*) GROUP BY`；`_kp_chapters` 预先按 `signal.key` 分组。 |
| P2-2 | 性能 / 并发落库 | `src/lecture_analyzer/pipeline/ingest.py:229-238` | `digest_material` 并发识别各页批次后按提交顺序 `for future in futures:` 等待，若首批次极慢或重试，后续已完成的批次无法及时落库与汇报进度。 | 改为 `for future in as_completed(futures):` 完成即落库并汇报进度。 |
| P2-3 | 并发 / CLI 锁 | `src/lecture_analyzer/cli.py:305-318, 481, 487` | `lec lecture add-part` 未持有 `lecture_lock`（网页端已加锁），可能与后台转写并发改节；`lec review` 导出时未持有 `memory_lock`（后台任务已加锁）；`books.remove_report` 删报告并 `export_course` 未加 `memory_lock`。 | 补齐与网页/后台任务一致的 `lecture_lock` 与 `memory_lock`。 |
| P2-4 | 错误处理 / 健壮性 | `src/lecture_analyzer/config.py:156-163`、`media.py:24-31`、`books.py:28-33`、`asr/models.py:92-97`、`pipeline/report_render.py:86`、`cli_plan.py:62`、`cli.py` | 1）`load_config` 未捕获 `tomllib.TOMLDecodeError` / `OSError`，配置文件语法错时直接抛未包装堆栈；2）`run_ffmpeg` 未传 `-nostdin` / `stdin=DEVNULL`；3）`books.safe_name` 未转义 Windows 保留设备名（`CON`/`NUL`/`COM1` 等）；4）`asr/models._progress_text` 扫描目录时未捕获并发重命名导致的 `OSError`；5）`report_render._plan_lines` 未用 `table_cell` 转义 `|` 和换行；6）`cli_plan.analyze` 读取 `--text-file` 用 `utf-8` 未兼容 Windows BOM（`utf-8-sig`）；7）CLI 多处 Rich 输出未 `escape()` 课程名/学科名。 | 逐一补齐异常包装、`-nostdin`、保留名转义、`OSError` 保护、`table_cell` 转义、`utf-8-sig` 与 `escape()`。 |
| P2-5 | 前端 / 桌面端边界 | `src/lecture_analyzer/web/static/app.js:208, 460`、`reader.js:24-40`、`desktop.py:139-146, 264-277` | 1）`app.js:208` 跳转前仅判断 `startsWith("/")`，未拦截协议相对路径 `//evil`；2）`app.js:460` 在没有 `#theme-menu` 的页面访问 `menu.hidden` 抛 `TypeError`；3）`reader.js` 连续跨节 `seek()` 会叠加多个 `loadedmetadata` 监听器导致回跳；4）`desktop.py` 启动失败未关闭已绑定的 `socket`，重复 `start_server` 未先停旧服务。 | 补齐 `!startsWith("//")`、`if (menu)`、清理旧 `pendingLoad` 监听、关闭失败套接字及停旧服务。 |
| P2-6 | 构建 / CI / 文档 | `pyproject.toml:24, 54`、`.github/workflows/`、`README.md:168-174` | `httpx>=0.28.1` 在 `dependencies` 与 `[dependency-groups] dev` 中重复声明；仓库仅有发布打包工作流 `release.yml`，缺少 PR/Push 自动测试 CI；`README.md` 未说明基础 `uv sync` 运行测试时 ASR 用例的跳过行为。 | 清理重复依赖并同步 `uv.lock`；新增 `.github/workflows/ci.yml`；更新 `README.md` 测试说明。 |

---

### P3 — 低优先级 / 架构演进项（留待后续迭代）

| 编号 | 类别 | 位置 | 问题描述与影响 | 建议方向 |
| --- | --- | --- | --- | --- |
| P3-1 | 数据库性能 | `src/lecture_analyzer/library_schema.py:67-90, 134-148, 215-242` | `signal(lecture_id)`、`llm_call(course_id)`、`kp_signal(kp_id)`、`lecture_page(material_id)`、`report(course_id)`、`job(course_id, kind, state)` 缺少二级索引，数据量增大后统计与删除级联扫描变慢。 | 随下一版本 Schema 升级（`user_version = 6`）统一追加索引，走既有备份迁移机制。 |
| P3-2 | 代码结构 / 重复代码 | `web/views.py`（488 行）、`pipeline/review.py`、`library.py`、`cli.py`、`web/static/app.js`（800 行）；多模块重复的 `_now()`、`_*_or_404` | 若干核心文件偏大；`_now()` 在 5+ 个存储模块重复定义；`web/app.py` 与 `web/actions.py` 存在重复查找函数。 | 后续按功能域渐进拆分模块并抽取公共时间与 Web 查找工具。 |
| P3-3 | 功能边界 | `src/lecture_analyzer/web/qa_page.py:160-170`、`web/manage_lectures.py:106-128`、`pipeline/materials.py`、`pipeline/ingest.py:235` | 1）删除问答对话时未取消仍在排队的该对话 `qa` 任务；2）网页修改课件页（`set_pages`）未持有 `lecture_lock`；3）资料拆题与逐页识别尚未透传 `FallbackProvider` 的 `billed_profile`。 | 结合任务取消策略和资料流程单测一并补齐。 |
| P3-4 | 渲染与安全增强 | `src/lecture_analyzer/web/rendering.py:14, 18, 31`、`web/templates/base.html:30-32`、`web/templates/qa.html:25-28`、`web/app.py:131-137` | 1）`_MATH` / `_LEFTOVER_BOLD` 正则会触及行内代码与代码块内的 `$` 和 `**`；2）KaTeX / Google Fonts 走外部 CDN 且未带 SRI（离线时公式不渲染）；3）`/api/*` 路径异常目前也渲染 HTML 错误页而非 JSON。 | 改用 markdown-it 插件或屏蔽代码块后再替换；内置本地 KaTeX 静态资源；对 `/api/` 前缀统一返回 JSON 错误。 |
| P3-5 | 缓存写入与 JSON 解析 | `src/lecture_analyzer/asr/llm_backend.py:134, 168`、`llm/base.py:261-267` | 1）大模型转写生成 `.mp3` 与写切片缓存 JSON 非原子写入，中途强杀可能留下残缺 mp3（JSON 读回已容错）；2）`extract_json_text` 用首个 `{` 到末个 `}` 切片，前后说明文字若带多余花括号会解析失败。 | `.mp3` 与缓存 JSON 改用临时文件原子落盘；`extract_json_text` 优先用 `json.JSONDecoder().raw_decode` 扫描合法对象。 |
| P3-6 | 维护清理 | `packaging/upload_release_asset.py:12-15` | 遗留的一次性脚本硬编码了作者本机路径（`F:\claude工作区\...`）与旧版本号 `v0.1.0`，如今已有 `release.yml` 自动上传。 | 需维护者确认是否删除或改为 CLI 参数化脚本。 |

---

## 三、本次已实施的改进项

本次遵循「高价值、低风险、不破坏现有对外行为、代码注释使用中文」的原则，已落地全部 **P0、P1 及绝大部分 P2** 改进项，并补充完整回归测试：

1. **ASR 延迟加载与基础环境修复**（P0-1、P2-4）
   - `asr/funasr_backend.py`、`asr/vad.py`、`asr/__init__.py`、`asr/qwen3_backend.py`、`asr/models.py`：移除顶层 `numpy` 硬依赖、捕获 `torch` 缺失回退 CPU、补全无 `transformers` 时的语种兜底表、扫描下载进度容忍并发文件消失；
   - `tests/test_funasr_backend.py`、`tests/test_qwen3_backend.py`：仅在真正需要 `numpy`/`torch`/`modelscope` 的用例内使用 `pytest.importorskip`，消除基础环境测试收集报错。
2. **LLM 懒加载回退、Gemini 重试与拒答计费**（P1-1、P1-2、P1-3、P1-4）
   - `llm/base.py`、`llm/fallback.py`、`llm/registry.py`、`llm/probe.py`、`llm/gemini_adapter.py`、`llm/anthropic_adapter.py`、`llm/openai_compat.py`；
   - `pipeline/extract.py`、`pipeline/slides.py`、`pipeline/run.py`、`pipeline/exam_plan.py`。
3. **运行时并发与输入边界加固**（P0-2、P0-3、P0-5、P1-9、P1-10、P2-3、P2-4）
   - `locks.py`、`job_runner.py`、`archive.py`、`config.py`、`media.py`、`documents.py`、`qa_attachments.py`、`books.py`、`timecode.py`。
4. **流水线死锁消除、资源清理、原话定位、预算精确计算与 CLI 加固**（P0-4、P1-5、P1-6、P1-7、P1-9、P2-2、P2-3、P2-4）
   - `pipeline/organize.py`、`pipeline/validate.py`、`pipeline/exam_plan.py`、`pipeline/memory.py`、`pipeline/ingest.py`、`pipeline/report_render.py`、`render.py`、`cli.py`、`cli_plan.py`。
5. **Web 查询性能优化、表单原子性与前端边界修复**（P1-8、P1-9、P1-11、P2-1、P2-5）
   - `materials.py`、`web/views.py`、`cli_materials.py`、`web/forms.py`、`web/qa_page.py`、`config_edit.py`、`web/settings_page.py`、`web/exam_plan_page.py`、`web/rendering.py`、`web/static/app.js`、`web/static/reader.js`、`desktop.py`。
6. **测试覆盖、依赖整理与 CI 工作流**（P2-6）
   - 新增 `tests/test_review_fixes.py` 覆盖以上全部修复的关键路径与边界；
   - 清理 `pyproject.toml` / `uv.lock` 重复 `httpx` 依赖；新增 `.github/workflows/ci.yml`；更新 `README.md` 测试说明。

---

## 四、第一轮留待后续的改进项与当前状态

| 编号 | 改进项 | 第二轮状态 | 说明 / 暂缓原因 |
| --- | --- | --- | --- |
| P3-1 | SQLite 补齐外键/查询二级索引 | 仍暂缓 | 涉及递增 `user_version` 与数据库迁移备份，应与下一次功能性表结构变更合并发布，避免无必要的迁移版本膨胀。 |
| P3-2 | 大模块拆分与公共工具去重 | 仍暂缓 | 属于结构性重构，改动面广且短期不直接修复缺陷，留待专项重构 PR 避免与缺陷修复混杂。 |
| P3-3 | 删对话取消在途问答任务、`set_pages` 加锁与资料步骤拒答计费 | **已全部落地** | 已在第二轮分别落地（`9749e9e`、`c15b2c6`、`0916ddd`、`1a3c3f0`），并补充并发与回退/失败计费测试。 |
| P3-4 | Markdown 代码块保护、本地化 KaTeX、`/api/*` 统一 JSON 错误 | **部分落地** | 代码段/属性保护、围栏线性扫描与 `/api/*` JSON 错误已在第二轮落地（`6ecb38a`、`6dce815`、`1a3c3f0`）；KaTeX 与 Google Fonts 本地化涉及引入前端静态资产及许可证声明，留待后续专项处理。 |
| P3-5 | 大模型转写 `.mp3`/缓存原子落盘与 `raw_decode` JSON 抽取 | **已落地** | `extract_audio`/`encode_mp3` 原子落盘与 `extract_json_text` 抗花括号干扰已在第二轮落地（`91f3f00`）；切片缓存 JSON 读回已捕获 `ValueError` 容错，无需改造。 |
| P3-6 | 清理或参数化 `packaging/upload_release_asset.py` | 仍暂缓 | 属于维护者个人发布辅助脚本，删改前应由仓库维护者确认是否仍在本地使用。 |

---

## 五、第二轮改进记录（`improve/round2-2026-11-25`）

> 基准提交：`0458b50`（第一轮 `improve/review-2026-11` 末端）  
> 涵盖提交：`9cd7c23` ~ `1a3c3f0`（共 13 个功能与收口提交，修改 30 个文件）  
> 执行情况：方案讨论由 Antigravity（gemini-4-neon xhigh）完成；第一批由 DeepSeek Harness 实施；第二批中因 DeepSeek Harness 不可用，T8 由 DeepSeek 写完后统筹方代为提交，T6/T7/T9/T10/T11 由 Antigravity 执行会话完成；统筹方（Claude）负责核验、review 与合入。

### 5.1 已落地改进（按功能域）

1. **压缩包与解压安全加固**（`9cd7c23`、`6dce815`、`1a3c3f0`）
   - `src/lecture_analyzer/archive.py`：
     - 修复上一轮引入的 P0 回归：`_check_extracted` 改用 `os.scandir` 栈式遍历，先判 `entry.is_symlink() or entry.is_junction()` 并用 `Path.unlink(missing_ok=True)` 只删链接本身、绝不穿透进入 Windows 目录联接点（junction），同时移除原先会误删外部文件的 `resolve().is_relative_to` + `shutil.rmtree` 分支；
     - `expand` / `_expand_item` / `_expand_folder` 统一用 `_is_link` 跳过符号链接与目录联接点（同时消除指向祖先目录的环路重复导入）；
     - `extract_zip` 按条目将目录/文件同名冲突与路径过长抛出的 `OSError` 包装为 `ArchiveError`；
     - `find_tool` 优先使用安装包随附的 `BUNDLED_7ZIP`（`tools\7zip\7z.exe`），避免被用户 PATH 中旧版 7-Zip 覆盖；
     - `extract_with_tool` 捕获启动外部工具时的 `OSError` 并转为友好 `ArchiveError`（提示无法启动解压工具，且不自动回退系统旧版 7-Zip）。
   - 回归测试：`tests/test_round2_boundaries.py`（`test_check_extracted_removes_junction_without_touching_external`、`test_expand_archive_junction_end_to_end`、`test_expand_skips_junctions_in_folder`、`test_extract_zip_wraps_file_dir_collision_as_archive_error`、`test_find_tool_prefers_bundled_7zip`、`test_extract_with_tool_wraps_launch_failure`）。

2. **问答附件与 PDF 页面渲染资源上限**（`97d954b`、`9cf38c6`、`1a3c3f0`）
   - `src/lecture_analyzer/documents.py`、`src/lecture_analyzer/qa_attachments.py`：
     - 抽取 `documents.render_page_png`，在渲染 PDF 页面前先按 `max_side` 夹紧缩放再渲染，`iter_pages` 与 `qa_attachments._page_png` 共用，避免超大页面先分配数百 MB 像素图或触发 `FzErrorLimit`；
     - `_read_pdf` 与 `_read_legacy_office` 先判断是否需要附图，图片额度耗尽或部分耗尽时正确置 `truncated = True`，纯扫描 PDF 不再误报“没有可读取的内容”；
     - `_read_docx` / `_read_pptx` 统一通过 `_read_xml_member` 读取 XML，先校验中央目录 `file_size <= MAX_XML_BYTES`（20 MB）再以 64 KB 分块限流读出，防止高压缩比或伪造小 `file_size` 的压缩炸弹撑爆内存。
   - 回归测试：`tests/test_round2_boundaries.py`（`test_scanned_pdf_attachment_marks_truncated_when_image_budget_exhausted`、`test_legacy_office_marks_truncated_without_office`、`test_docx_and_pptx_reject_oversized_xml_member`、`test_pdf_page_render_respects_max_image_side`（加强覆盖 14400×14400pt 最大页面尺寸）、`test_docx_forged_file_size_uses_bounded_memory`）。

3. **数据一致性、并发原子性与跨进程锁补齐**（`18aaf96`、`6020565`、`9749e9e`、`c15b2c6`、`4b731c9`、`9cf38c6`）
   - `src/lecture_analyzer/pipeline/organize.py`、`src/lecture_analyzer/library.py`：
     - 跨课移动讲时清空原课程的 `lecture_page`（含手动指定页）并无条件重置 `slides` 步骤；
     - 删除讲时保留课程账面费用（`course_id = COALESCE(course_id, ?), lecture_id = NULL`），删除课程时解绑课/讲级 `llm_call`，新增课/讲时解绑旧库遗留的孤儿 `llm_call`，避免 SQLite 复用自增编号导致费用串账。
   - `src/lecture_analyzer/library.py`、`src/lecture_analyzer/materials.py`：
     - 新增 `Library.immediate()`，以 `BEGIN IMMEDIATE` 保护讲与资料的 `status` 读-改-写，消除多进程/多连接并发更新丢写，同时保持原有即时提交语义并在 docstring 中明确说明。
   - `src/lecture_analyzer/web/qa_page.py`、`src/lecture_analyzer/job_handlers.py`：
     - 删除问答对话时通过统一的 `_job_thread` 找出并取消该对话尚未结束的 `qa` 任务，同步清理已答轮次及排队/失败任务的附件；`job_handlers._qa` 在写回前复核任务仍为 `running`，避免已删对话被迟到结果复活。
   - `src/lecture_analyzer/web/manage_lectures.py`、`src/lecture_analyzer/cli_materials.py`、`src/lecture_analyzer/pipeline/organize.py`：
     - 网页 `set_pages` 与命令行 `lec lecture pages` 写分支补齐 `lecture_lock`；
     - `delete_course` 按 `locks.py` 顺序（且讲/资料按 `id` 升序）加齐 `lecture_lock`、`material_lock`、`materials_lock`、`memory_lock` 后再复查讲与资料集合，拦截命令行并发处理及等锁期间新导入资料导致的磁盘残留。
   - 回归测试：`tests/test_round2_consistency.py`（共 10 个测试函数 / 11 个用例）。

4. **Markdown / 文本清理代码段保护与 Web 安全边界**（`6ecb38a`、`9cf38c6`、`6dce815`、`1a3c3f0`）
   - `src/lecture_analyzer/textnorm.py`、`src/lecture_analyzer/web/rendering.py`：
     - 在 `textnorm.py` 中统一提供 `MATH` 与 `split_code`（内部用 `_INLINE_CODE` 切出行内代码），由 `textnorm._strip_asides` 与 `web/rendering.render_markdown` 共用；
     - 围栏识别符合 CommonMark（兼容 CRLF、反引号信息串不含反引号、结束围栏长度允许不短于开栏且行尾仅有空白），`split_code` 改为逆序预统计后续结束围栏最大长度的线性逐行扫描，对大量未闭合开栏行保持 O(n) 复杂度，围栏之外再用 `_INLINE_CODE` 切出行内代码；
     - `strip_foreign_asides` / `strip_citations` 不再破坏围栏代码块里的函数参数或受孤立 `$` 跨反引号影响；
     - `render_markdown` 改用每次调用随机 8 位十六进制 `nonce` 占位符（不含 `:` 等行内终止符以保性能），`_fix_leftover_bold` 跳过 HTML 标签属性与 `<code>` 内文本。
   - `src/lecture_analyzer/web/app.py`：
     - `/api/*` 路径上的 `LibraryError` 与 `StarletteHTTPException` 统一返回 JSON `{"error": ...}`（保留 405 `Allow` 等响应头）；
     - `/media/{lecture_id}/{part}` 仅对 `audio/*`、`video/*` 保留原 MIME，其余误登记文件（如 `.html`、`.svg`）一律降级为 `application/octet-stream` 并附带 `X-Content-Type-Options: nosniff`。
   - 回归测试：`tests/test_web.py`（新增 `test_media_never_served_as_html`、`test_api_errors_are_json`、`test_markdown_leaves_code_and_attributes_alone`、`test_markdown_placeholder_cannot_collide`）、`tests/test_foreign_language.py`（新增 `test_foreign_asides_keep_code_intact`）、`tests/test_round2_boundaries.py`（新增 `test_split_code_is_linear_on_unclosed_fences`）。

5. **音视频原子写、大模型计费精确化、执行器与锁健壮性及测试补齐**（`91f3f00`、`0916ddd`、`1a3c3f0`）
   - `src/lecture_analyzer/media.py`：`extract_audio` 与 `encode_mp3` 共用 `_ffmpeg_to`（固定 `<stem>.partial<suffix>` 临时文件 + `os.replace`），进程被强杀不再留下被永久复用的半截音频。
   - `src/lecture_analyzer/asr/base.py`、`src/lecture_analyzer/asr/llm_backend.py`、`src/lecture_analyzer/asr/fallback.py`、`src/lecture_analyzer/pipeline/run.py`：
     - `ASRBackend` 新增 `on_usage` 钩子，`LLMTranscriber` 每次非 0 用量请求后立即回调记账（按 `billed_profile`），中途失败或进程被强杀不丢账；`_request_any` 统一捕获 `LLMError`，将输出截断等普通失败请求消耗的用量同样计入 `usage_by_model` 并回调 `on_usage` 后原样抛出；
     - `fill_skipped` 透传 `on_usage`（补齐 `llm:<档案>` 补转记账）并捕获 `(RuntimeError, ConfigError)`。
   - `src/lecture_analyzer/llm/base.py`：`extract_json_text` 在首个 `\{[ \t\r\n]*"` 处尝试 `raw_decode`（捕获 `ValueError` / `RecursionError`），容忍前后说明文字里的花括号，同时对截断 JSON 与超深嵌套保持原有回退与重试行为。
   - `src/lecture_analyzer/pipeline/page_digest.py`、`src/lecture_analyzer/pipeline/ingest.py`、`src/lecture_analyzer/pipeline/materials.py`：
     - 资料逐页识别、拆题、对应知识点三步通过可选 `on_call` 回调按实际作答的 `billed_profile` 记账并记录真实模型名（不改变二元组返回契约）；
     - `split_items`、`map_items`、`digest_pages` 在 `generate_structured` 抛带非 0 用量的 `LLMError` 时按调用档案回调 `on_call`；`ingest._digest_batch` 将失败批次已收集的用量挂到异常上带回主线程，由 `digest_material` 先记账再跳过落页。
   - `src/lecture_analyzer/job_runner.py`、`src/lecture_analyzer/locks.py`：`JobRunner.start()` 在 `pause_interrupted()` 失败时释放 `_owner_lock`，`_run()` 未启动子进程时清理 0 字节空日志；`locks._lock_file` 分拆 `open` 与 `lock_file` 并为打开失败设置 `OPEN_RETRY_SECONDS = 10.0` 上限。
   - 回归测试：`tests/test_round2_reliability.py`（共 15 个测试函数，含 T6/T7 验收、失败请求用量记账 `test_transcriber_logs_usage_of_failed_request` / `test_material_steps_log_usage_of_failed_batch` 与第一轮 P0-1/P0-3/P0-4 缺口补齐）、`tests/test_review_fixes.py`（加强 `test_locks_release_in_process_guard_on_open_error_and_retry_permission` 覆盖重试循环与 `sleep`）。

### 5.2 第二轮测试统计

- 验证命令：`uv run pytest -m "not office" -q -p no:cacheprovider`
- 测试结果：**514 passed, 9 skipped, 2 deselected**（较第一轮基线 `471 passed, 9 skipped, 2 deselected` 净增 43 个回归用例）。

### 5.3 第二轮结束后仍留待后续的项目

| 编号 / 来源 | 项目 | 暂缓理由 |
| --- | --- | --- |
| P3-1 | SQLite 二级索引（`signal`、`llm_call`、`kp_signal`、`lecture_page`、`report`、`job`） | 需递增 `user_version = 6` 并触发书库备份迁移，留待下一次功能性表结构变更合并实施。 |
| P3-2 | 大模块拆分与公共工具去重（`web/views.py`、`pipeline/review.py`、`library.py`、`cli.py`、`web/static/app.js`、重复 `_now()`） | 纯结构性重构，留待专项 PR，避免与缺陷修复混杂。 |
| P3-4（剩余） | KaTeX 与 Google Fonts 本地静态化（离线渲染与去外部 CDN 依赖） | 涉及引入前端静态资产与 `packaging/notices.py` 头部许可证维护，需结合桌面打包专项推进。 |
| P3-6 | 清理或参数化 `packaging/upload_release_asset.py` | 个人发布辅助脚本，需仓库维护者确认是否仍在本地使用。 |
| 已知局限 | `extract_lecture` 多窗口部分回退、`FallbackProvider` 拒答输入与备用输出按备用单价混算 | 计费偏差极小且仅在拒答回退时触发，保持单条汇总逻辑简洁，后续如细化账单再拆分。 |
| 已知局限 | `FallbackProvider` 中备用档案也失败时的记账归属与主模型拒答用量合并 | 主模型拒答后若备用档案也失败，异常上的用量按主档案记账且主模型拒答那次的用量未并入（需调整 `src/lecture_analyzer/llm/fallback.py` 与异常契约，留待后续计费细化）。 |
| 后续观察 | 图片附件与资料图片（`qa_attachments._image_file_png` / `documents._load_image_png`）解码前校验像素数上限 | 单机本地场景已有 50 MB 附件上限，且 PyMuPDF 单图文档不直接暴露原始像素宽高，需自写 PNG/JPEG/BMP 头部解析，收益风险比不高，先作观察项。 |
