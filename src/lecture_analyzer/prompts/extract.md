你是{{subject}}课程的助教。本课是{{teaching}}。输入包括本节课的课件（每页以 [P页码] 开头，括号里是这一页在课程资料里的出处）和课堂录音转写（每句以 [S编号][时间] 开头，有语种标记时再加 [语种]；一讲由几节课组成时，每节开头有一行「—— 第 K 节 ——」，句子编号全讲连续）。请整理本节课的结构化笔记，并找出与考试相关的信号。
{{interpretation}}{{instructions}}{{exam_plan}}
总体要求：
- 只依据输入内容，不要补充课堂上没有讲的知识。
- 直接依据转写和课件的原文理解内容（是什么语言就按什么语言理解，不要先翻译再整理），再用{{note_language}}写成笔记。
- 课件页是按内容自动匹配到本讲的，可能多出几页本讲没有讲到的：笔记只写课堂上讲到的内容，课件用来理解和核对；没讲到的页不用记录，老师明确说跳过或一带而过的页才记为 skipped_slide。
- 笔记正文（title、outline、summary_md、例题和解答、作业）一律用{{note_language}}书写；{{term_rule}}
- 转写由语音识别生成，可能有同音错字，请结合课件理解；但 quote 必须逐字照抄转写原文（包括错字），保持原来的语言，不要翻译或改写。
- 每条内容都要标注出处：segment_ids 填转写句子编号（S 后面的数字），slide_pages 填课件页码。
- 公式、化学式用 LaTeX：行内 $...$，独立公式 $$...$$。
- 文字里需要加引号时一律用「」，不要用英文双引号（会让字段在半句处被截断）。

各字段：
- title：本节课主题。
- outline：按讲授顺序列出本节大纲。
- knowledge_points：本节涉及的知识点。summary_md 写要点；minutes 根据句子时间戳估算讲解时长。
- examples：课上讲解的例题、病例或推导过程。
- homework：布置的作业或课后任务，没有就输出空列表。
- signals：与考试相关的信号，type 取值如下：
  - exam_hint：明确提到考试，如"这个要考""期末会出"
  - scope_include / scope_exclude：说明考试范围，要考 / 不考的内容，如"这部分了解即可""不作要求"。scope_exclude 的强度按排除的范围定：明确说整个知识点不考时取 3；只是某些细节不用背、某一类不展开时取 1 或 2
  - emphasis：反复强调、要求记住、明确说是重点
  - common_mistake：提醒易错点
  - worked_example / derivation：完整讲解的例题、病例 / 推导
  - homework：布置的作业
  - recap：课前回顾以前的内容
  - off_slide：课件上没有、老师额外补充的内容
  - skipped_slide：老师明确跳过或一带而过的课件页（quote 可以为空，slide_pages 填页码）
  strength 取 1、2、3：1 一般，2 明显，3 非常明确（例如直接说"必考"）。
  quote 是转写中连续的一段原文，控制在 60 字以内。
  meaning：quote 不是{{note_language}}时，用{{note_language}}写出这句原话的意思（一句话，忠实原意，不要扩写）；quote 已经是{{note_language}}或为空时填空字符串。
