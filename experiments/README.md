# M0 可行性验证

目标：用 2–3 节真实课回答三个问题，结论写进 `docs/m0-findings.md`：

1. 哪种转写方案在你的课堂录音上最准、速度能接受？
2. 页面识别和单课抽取，各模型的效果和费用如何？每一步默认用哪个模型？
3. 每节课实际消耗多少 token、多少钱？

## 准备

```bash
uv sync --extra asr                 # 安装本地语音识别依赖（PyTorch CUDA 版 + FunASR）
copy config.example.toml config.toml
copy .env.example .env              # 填入要用的模型密钥
uv run lec models --remote gemini   # 查 Gemini 可用型号，填进 config.toml 的 [models.gemini]
uv run lec models --ping            # 逐个检查模型能否调用
```

第一次运行本地转写时会自动下载模型（约 3GB，存放在 `%USERPROFILE%\.cache\modelscope`）。

## 步骤

课程素材请放在 `materials/` 目录，所有输出都在 `experiments/output/`，两者都已被 git 忽略，不会被提交到公开仓库。下面示例中的 `第3讲.mp4` 等文件名请换成 `materials/` 下的实际文件。

1. **课件识别，顺便生成热词表**

   ```bash
   uv run python experiments/llm_trial.py digest --slides 第3讲.pptx --models gemini,claude
   ```

   每个模型输出 `digest_<模型>.md`（人工检查公式和图表是否识别正确）和 `terms_<模型>.txt`；第一个成功的模型的术语表另存为 `terms.txt`。

2. **转写对比**（先截 15 分钟，省时间也省钱）

   ```bash
   # 中文课
   uv run python experiments/asr_compare.py materials/药理学3.m4a --start 30:00 --minutes 15 --teaching zh --backend funasr:paraformer --backend funasr:sensevoice --backend qwen3:1.7b --out experiments/output/asr_pharmacology
   # 韩语授课 + 现场翻译的课
   uv run python experiments/asr_compare.py materials/病毒学2-1.mp4 --start 20:00 --minutes 15 --teaching ko+zh --backend funasr:sensevoice --backend qwen3:1.7b --backend qwen3:0.6b --out experiments/output/asr_virology
   ```

   - `--teaching`（加上可选的 `--interpreted-to`）决定语种：录音里只有一种语言（如 `zh`、`ko`）时强制该语种；几种语言混用（如 `ko,en`）或有现场翻译（如 `ko+zh`，即 `--teaching ko --interpreted-to zh`）时逐句自动识别。有 Gemini 密钥后可以再加 `--backend llm:gemini`。
   - 想看热词的作用，加上 `--hotwords experiments/output/llm/terms.txt`，把 `funasr:paraformer` 再跑一次（输出目录用 `--out` 区分）。
   - 结果中每句会标出语种，例如 `[21:42][ko] …`。

3. **人工校对参考稿**：从某个结果里导出 10 分钟初稿，边听录音边改成参考稿

   ```bash
   uv run python experiments/cer.py draft experiments/output/asr/funasr_paraformer.json --start 03:00 --end 13:00 --out experiments/output/ref.txt
   ```

4. **计算字错率**

   ```bash
   uv run python experiments/cer.py score --ref experiments/output/ref.txt --start 03:00 --end 13:00 experiments/output/asr/*.json
   ```

   韩语加翻译的课，韩语和中文分开校对、分开计算：`draft` 和 `score` 都加上 `--lang ko` 或 `--lang zh`。

5. **单课抽取对比**（用同一份课件识别结果，保证对比公平）

   ```bash
   uv run python experiments/llm_trial.py extract --transcript experiments/output/asr/funasr_paraformer.json --digest experiments/output/llm/digest_gemini.json --models claude,gemini
   ```

   检查 `notes_<模型>.md`：知识点是否完整、考点信号是否有据可查。结果表中的"原话命中"表示引用的原话能在转写中找到的比例，"token/字" 填进模型档案的 `tokens_per_char`。

## 说明

- 有 GPT 或国产模型的密钥时，在 `config.toml` 填好型号，加进 `--models` 一起比较即可。
- 本目录只放 M0 的对比脚本，正式流程从 M1 起在 `lec` 命令里实现。
