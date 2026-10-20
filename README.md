# lecture-analyzer

课程录音分析与考点预测工具，个人本地使用。任何课程都能用：放进录音录像、课件（PPT、PDF 等）和复习资料（往年真题、模拟题、复习提纲……），授课和笔记可以是任何语言。

- 把老师上课的录像/录音转写成带时间戳的文字，自动把课件匹配到各讲，整理每节课的结构化笔记
- 分析新课时自动带入前几节课的记忆（知识点树 + 历史笔记），而不是把原始音视频塞进上下文
- 考前综合课堂内容、课件和模拟题等资料，给出"带证据的考点优先级"：每条结论都能追溯到"第几讲 + 时间点 + 原话"、"哪份课件第几页"或"哪份资料第几题"
- 模型层统一接口：Claude、Gemini、GPT 和国产模型（DeepSeek、通义千问、Kimi、GLM 等）都能接，每一步可以分别指定模型

> 当前进度：**M4**。做成了双击就能用的桌面程序（安装包 setup.exe，自带 Python、PyTorch 和本地语音识别）：建课、加讲、处理、导入资料、设置模型和密钥都在界面里完成；新增按课程的连续问答（回答附出处）；后台任务改为独立进程，可以取消；韩语、英语等外语授课的课程用原文理解、用中文总结；分析结果（笔记、课程记忆、总复习、考前报告、问答）一律用中文；考前报告直接列出考点和要点，不写推断过程。M1–M3 的内容见 `docs/`。

## 安装版（推荐）

1. 运行 `LectureAnalyzer-Setup-<版本>.exe`。按当前用户安装，不需要管理员；没有代码签名，Windows 可能提示"已保护你的电脑"，点「更多信息」→「仍要运行」。
2. 打开「课程书架」。第一次启动选择书库文件夹：新建（默认"文档\课程书架"），或选以前用过的书库文件夹（里面有 `config.toml`）。
3. 新建的书库会打开设置页：填写各模型的型号和密钥（走中转站时再填接口地址），点「测试连接」。密钥保存在 Windows 凭据管理器里，只有当前 Windows 账号能读取，页面上不会显示。
4. 回到书架：新建课程（勾选老师讲课用的语言：中文、韩语、英语、日语，或填其他语言代码）→ 添加一讲（选录音录像，一讲分几个文件时按顺序选）→ 处理。处理在后台进行，顶栏「后台任务」显示进度，可以随时取消，再次处理会接着做。
5. 转写方案默认是「自动」：按课程的授课语言用本地模型（只有中文用 Paraformer，韩语、英语等用 Qwen3-ASR）。模型（中文约 2GB、Qwen3-ASR 约 4GB）第一次用到时自动下载，也可以在设置页提前下载。有能听音频的大模型（如 Gemini）时，建议在设置页把转写方案改成大模型转写：更准，也不占显卡。

- 需要 Windows 10/11（64 位）和 Microsoft Edge WebView2 运行时（Windows 11 自带；没有时书架会改用浏览器打开，并提示下载地址）。本地转写需要 NVIDIA 显卡，没有显卡时自动改用 CPU（很慢）。
- 命令行：开始菜单里的「课程书架命令行」，命令与下面的源码版相同（`lec models`、`lec run 课程名`……）。
- 卸载不会删除书库文件夹；重新安装后选同一个文件夹即可。

## 从源码运行：环境要求

- Windows 10/11，Python 由 [uv](https://docs.astral.sh/uv/) 管理（项目固定 Python 3.12）
- 本地转写需要 NVIDIA 显卡（6GB 显存可用）；没有显卡可以改用大模型转写
- 导入 PPTX/DOCX 需要本机安装 Microsoft Office（转 PDF 用）；PDF、图片、文本不需要
- 导入 rar、7z 压缩包需要本机装有 7-Zip 或 WinRAR；zip 不需要
- 不需要单独安装 ffmpeg（使用 imageio-ffmpeg 自带的版本）

## 安装与配置

```bash
uv sync                      # 基础依赖
uv sync --extra asr          # 本地语音识别（PyTorch CUDA 版 + FunASR，约 3GB）
copy config.example.toml config.toml
copy .env.example .env       # 填入要用的模型密钥；.env 和 config.toml 不会被提交
uv run lec models            # 查看模型档案、能力和状态
uv run lec models --ping     # 给已配置的模型各发一个最小请求，检查密钥和端点
uv run lec models --remote gemini   # 列出服务端可用型号，用来填写 config.toml
```

## 使用流程

一门课是一本书，一讲是书里的一章；一讲可以由多节录音录像组成，处理时合并成一讲。

```bash
# 1. 建课程（一本书）：学科、授课语言（ISO 639 代码，任何语言都可以）；可附上课程提示词。笔记和分析一律用中文
uv run lec course add 药理学 --subject 药理学 --teaching zh
uv run lec course add 病毒学 --subject 病毒学 --teaching ko --interpreted-to zh   # 韩语授课、现场翻译成中文
uv run lec course add Biology --subject Biology --teaching en,fr                 # 老师混用英语和法语
uv run lec course set 病毒学 --instructions-file "D:\录课\病毒学提示词.txt"

# 2. 课件：分析开始前必须先选择是否提供课件。提供了就导入，处理时自动匹配到各讲；没有就声明不提供
uv run lec material add 病毒学 "D:\资料\病毒学课件.zip" --kind slides     # 文件、文件夹、zip/rar/7z 都可以
uv run lec course set 微生物 --no-slides

# 3. 加讲：原始音视频不会被复制，只记录路径；一讲有多节就重复 --media
uv run lec lecture add 药理学 --seq 3 --media "D:\录课\药理学3.m4a"
uv run lec lecture add 病毒学 --seq 1 --media 病毒学1.1.mp4 --media 病毒学1-2.mp4 --media 病毒学1-3.mp4
uv run lec lecture add-part 病毒学 1 --media 病毒学1-4.mp4    # 追加一节，已转写的节不会重新转写

# 4. 处理：抽音频 → 转写 → 匹配课件 → 整理笔记 → 校验原话 → 整合课程记忆 → 上架，中断后再运行会接着做
uv run lec run 病毒学              # 按讲次依次处理所有没完成的讲
uv run lec lecture pages 病毒学 1 --pages "3:1-30"   # 自动匹配不准时手动指定（资料编号:起止页）
uv run lec run 药理学 3 --force extract --model gemini   # 换个模型重新整理笔记（依赖它的步骤自动重跑）
uv run lec rebuild 病毒学 --from 2  # 从第 2 讲起重新整合课程记忆（前面的讲直接重放已有结果）

# 5. 复习资料：往年真题、模拟题、习题、复习提纲、考试通知，导入后拆成题目并对应到课程知识点
uv run lec material add 病毒学 "D:\资料\期中模拟题.pdf" --kind mock_exam
uv run lec material list 病毒学

# 6. 总复习与考前报告
uv run lec review 病毒学                              # 按章节生成复习讲义，内容没变的章节不会重复生成
uv run lec report 病毒学 --models claude,gemini       # 多个模型分别排名后取共识，列出考点和要点
uv run lec report 病毒学 --lectures 1,2 --materials 12   # 只分析部分讲和部分资料

# 7. 打开书架：桌面窗口（与安装版相同），或在浏览器里打开（没有原生的选文件对话框，手动粘贴完整路径）
uv run lec desktop
uv run lec serve

# 其他
uv run lec lecture list                       # 各讲的处理进度
uv run lec lecture move 微生物 3 --to 病毒学    # 把放错课程的一讲移过去
uv run lec cost                               # 按步骤和模型汇总 token 用量与估算费用
```

**课程记忆**：每处理完一讲，模型把它的知识点归并进整门课的知识点体系（按章节分组），并找出与前几讲的联系（例如老师回顾上次的内容）。阅读页会显示"与前几讲的联系"，点击就跳到前一讲对应的时间点。前面某一讲重新处理后，后面的讲会标记为"课程记忆待更新"，`lec rebuild` 或再次 `lec run` 时自动更新。

**总复习**：模型按章节把各讲的讲解合并成复习要点；老师原话和出处由程序从已核对的考点信号里填入，重要程度（★★★ 提示考试或明确划重点，或出现在往年真题、模拟题、复习提纲里 / ★★ 强调、易错或多次讲到 / ★ 一般 / 不考）由程序计算，不靠模型打分。

**课件匹配**：导入的课件和教材先逐页识别（按页缓存，中断后接着做）；处理每一讲时，模型根据全部页面的目录和这一讲的转写找出讲到的页，程序校验页码、补上中间被跳过的几页。课件和授课语言不同也能匹配。之后再导入课件，只重新匹配和整理笔记，不重新转写。

**外语授课的课程**：老师讲韩语、英语等外语时，转写、课件和资料都保留原文，模型直接读原文理解，再用中文总结。笔记、知识点名称、课程记忆、总复习、考前报告和问答一律用中文，不在括号里附外语原词（DNA、PCR 这类通用缩写照常用）；外语原词只在后台作为知识点别名，用来对上外语出的考题。老师的原话先显示中文意思，原文收起，点开可核对；现场翻译只用来帮助理解，以老师的原话为准。

**问答**：在课程页点「提问」，可以连续追问、限定某一讲。程序先找出相关的知识点，整理好带编号的证据（笔记要点、转写片段、老师原话、资料题目、课件页、例题），再由模型用中文回答（用外语提问也一样）；回答里的出处可以点击，原话跳到录音对应的时间点，课件页弹出页面图片。课程内容里没有的，回答会直接说明。

**考前报告**：直接列出判断后的知识点本身。判断在后台进行：程序先按课堂信号、资料题目和讲解时长给每个知识点打参考分（权重可在 `[scoring]` 里调），整理好带编号的证据；每个模型分别排出最可能考的考点，程序用 Borda 计数取共识。报告按可能考到的程度列出考点，每个只有名称、星级、可能题型和知识点要点（中文，取自总复习，生成报告时会先更新相关章节），以及一个到课程页核对出处的链接；不写理由、分数和推断过程。报告可以打印成 PDF、下载 Markdown，也可以删除。

每门课还会导出到 `data/books/<课程>/`：`book.json`（整本书：目录、知识点体系、跨讲联系、总复习、资料、报告）、`课程总览.md`、`总复习.md`、`考前报告-时间.md`，以及每讲一个文件夹（`lecture.json` 和可以直接用 Obsidian 或 Typora 阅读的 `笔记.md`）。

用大模型转写长课时，音频每 10 分钟一片、结果逐片缓存，中断后重跑不会重复请求。某一片被上游拦截（返回空结果）时自动对半切开重试；切到 2.5 分钟仍被拦截的片段，先换用 `[asr] llm_alternates` 里的其他型号，还不行再用本地模型补转（`[asr] fallback`），处理记录会注明补转了哪些区间。之后想让这些区间重新交给大模型，运行 `lec run 课程 讲次 --force transcribe`：已转写好的片段直接复用，只重试缺的部分。

## 密钥

密钥优先从 Windows 凭据管理器读取（在设置页保存），没有时用环境变量或 `.env`；命令行和书架用的是同一份。设置页有「把 .env 里的密钥移到凭据管理器」，点了之后 `.env` 里的明文密钥会被清空。

## 使用中转站

在设置页填写，或在 `.env` 里填写接口地址和各接口的型号，填了的值会覆盖 `config.toml` 和系统环境变量里的同名设置：

```
ANTHROPIC_BASE_URL=https://你的中转站/
GEMINI_BASE_URL=https://你的中转站/
OPENAI_BASE_URL=https://你的中转站/v1     # OpenAI 兼容接口要以 /v1 结尾
CLAUDE_MODEL=claude-opus-5-5
GEMINI_MODEL=…
GPT_MODEL=…
```

同一个中转地址可以请求所有模型，靠型号区分。`uv run lec models` 会列出每个档案实际使用的型号和地址，`uv run lec models --remote <档案名>` 可以查询中转站提供的型号。

中转站前面有 Cloudflare 时，耗时长的请求可能因为 120 秒内没有响应而失败（524）。OpenAI 兼容接口的档案可以打开流式请求：

```toml
[models.gemini.options]
stream = true
```

## 新增模型

- **同一接入方式下的新模型**（如新的国产模型）：在 `config.toml` 加一段 `[models.<名称>]`，写明 `adapter`、`model`、`base_url`、`api_key_env` 和能力，不用改代码。
- **新的接入方式**：继承 `lecture_analyzer.llm.base.LLMProvider` 实现 `_complete`，在 `llm/registry.py` 注册一行。

## 目录

```
src/lecture_analyzer/
  llm/          统一模型接口与 anthropic / gemini / openai_compat 三种适配器
  asr/          语音识别：本地 FunASR / Qwen3-ASR、大模型转写及本地补转
  pipeline/     处理流程：整理笔记、原话校验、跨讲整合、总复习、输入预算、移动讲、
                资料入库、课件匹配、拆题与知识点对应、打分、考前报告
  prompts/      各步骤的提示词
  web/          本地网页书架（页面、写操作、设置、问答、播放器、首次设置页）
  library.py    书库（SQLite），表结构与迁移在 library_schema.py
  course.py     课程设置：学科、授课语言、现场翻译（笔记和分析一律中文）
  knowledge.py  课程记忆（L2）与总复习的存储
  materials.py  课程资料、题目、每讲对应的课件页的存储；reports.py 考前报告的存储
  archive.py    文件、文件夹、压缩包展开（文件名编码识别、防路径穿越）
  jobs.py       后台任务表；job_runner.py 子进程执行与两条队列；job_handlers.py 各类任务；worker.py 子进程入口
  desktop.py    桌面程序（pywebview 窗口、原生对话框、首次设置、单实例、关窗确认）
  config_edit.py 设置页改配置（保留注释）；credentials.py 密钥（凭据管理器）；apphome.py 书库位置
  conversations.py 问答记录的存储
  books.py      导出 book.json / lecture.json / Markdown
  documents.py  课件和资料拆页（PDF / PPTX / DOCX / 图片 / 文本）
  media.py      抽音频、切片、转码
packaging/      打包成 setup.exe（build.ps1、启动程序、Inno Setup 脚本、图标）
experiments/    M0 对比脚本（见 experiments/README.md）
docs/           设计文档、M0 结论与 M2–M4 验收记录
tests/          pytest 测试
```

## 测试

```bash
uv run pytest
```

需要 Office 的转换测试标记为 `office`，可用 `uv run pytest -m "not office"` 跳过。

## 打包成 setup.exe

需要 [Inno Setup](https://jrsoftware.org/isdl.php) 6.5 以上和 [7-Zip](https://www.7-zip.org/download.html) 的 x64 MSI（只用来解出 7z.exe 随附进安装包，不会安装）：

```powershell
powershell -ExecutionPolicy Bypass -File packaging\build.ps1 -InnoSetup <ISCC.exe 路径> -SevenZipMsi <7z*-x64.msi 路径>
```

生成 `dist\LectureAnalyzer-Setup-<版本>.exe`（程序目录约 3.7GB，大部分是 PyTorch 的显卡运行库）。加 `-SkipInstaller` 只生成 `build\LectureAnalyzer`，可以直接双击里面的 `LectureAnalyzer.exe` 试用。Inno Setup 6.x 没有自带简体中文，把它官方仓库里的 `Files\Languages\Unofficial\ChineseSimplified.isl` 放进 Inno Setup 的 `Languages` 文件夹，安装向导就是中文。

## 隐私与合规

录音、转写、课件和资料只用于个人学习，不要公开传播。它们都保存在本地 `data/` 和 `experiments/output/` 目录（已被 git 忽略）；只有调用云端模型时，相关内容才会发给对应的模型服务商。
