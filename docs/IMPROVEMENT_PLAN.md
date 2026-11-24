# lecture-analyzer 代码评审与改进方案

> 评审基准分支：`master`（版本 `0.3.5`，提交 `37cb789`）  
> 改进分支：`improve/review-2026-11`

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

## 四、留待后续的改进项与原因

| 编号 | 改进项 | 暂缓原因 |
| --- | --- | --- |
| P3-1 | SQLite 补齐外键/查询二级索引 | 涉及递增 `user_version` 与数据库迁移备份，应与下一次功能性表结构变更合并发布，避免无必要的迁移版本膨胀。 |
| P3-2 | 大模块拆分与公共工具去重 | 属于结构性重构，改动面广且短期不直接修复缺陷，留待专项重构 PR 避免与缺陷修复混杂。 |
| P3-3 | 删对话取消在途问答任务、`set_pages` 加锁与资料步骤拒答计费 | 涉及调整 `JobRunner` 取消联动与资料流水线返回契约，影响面较大，建议单独设计和补充并发集成测试。 |
| P3-4 | Markdown 代码块保护、本地化 KaTeX、`/api/*` 统一 JSON 错误 | 引入静态前端资产或改动公共异常处理器可能改变现有模板渲染与错误断言，需结合前端构建规范统一推进。 |
| P3-5 | 大模型转写 `.mp3`/缓存原子落盘与 `raw_decode` JSON 抽取 | 需要针对多模型输出样例（含代码块与嵌套文本）做回归验证，避免放宽或收紧既有 JSON 提取行为。 |
| P3-6 | 清理或参数化 `packaging/upload_release_asset.py` | 属于维护者个人发布辅助脚本，删改前应由仓库维护者确认是否仍在本地使用。 |
