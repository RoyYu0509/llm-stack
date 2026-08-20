# Assignment B — Closing the Loop

一份 CS336 assignment handout 风格的**回顾式讲义**，覆盖「语料混合 → 预训练 → 指令微调 ablation → KV cache → serving 集成 → 系统 benchmark」这条链路。中文正文，技术术语保持英文。

配套的姊妹篇是 `llm-serving/docs/handout/assignment-a.tex`（《Inference Serving Engine》）。没读过姊妹篇也能读这份，但第 4 章会少掉一半的「为什么是这样」。

## 构建

需要 XeLaTeX（`ctex` + `xeCJK`）。macOS 上装 MacTeX 即可，不需要 `-shell-escape`。

```bash
latexmk -xelatex assignment-b.tex
```

不带参考答案的**空白自测版**：

```bash
latexmk -xelatex -jobname=assignment-b-quiz -pretex='\def\HANDOUTQUIZ{}' -usepretex assignment-b.tex
```

## 文件

| 文件 | 作用 |
|---|---|
| `assignment-b.tex` | 主文档，只负责 `\include` 各章 |
| `handout.sty` | 共享样式。**与姊妹篇是同一份的两个副本**，顶部有 `CANONICAL` 标记和版本串 |
| `sections/*.tex` | 每章一个文件 |
| `xref-a.tex` | 姊妹篇导出的稳定锚点，**手工同步的副本**（见下） |
| `refs.bib` | 参考文献 |

## 跨讲义引用

正文里的 `\extref{a:some-key}` 指向姊妹篇的小节。机制是：姊妹篇构建时生成 `assignment-a.xrefs`，把那个文件复制过来覆盖 `xref-a.tex`（保留文件头的说明）。

不用 `xr-hyper`，因为那要求构建时能读到姊妹篇的 `.aux` —— 两份讲义在不同的 git 仓库里，只 clone 了这一个的人拿不到另一个。引用了未同步的 key 时，构建会警告并把它排成 `[??: key]`，所以漂移在日志里看得见，不会静默。

## 一条读这份讲义的前提

这份讲义里的每张结果表都标注了出处（哪个 JSON、多少 batch、什么 seed）。这不是排版洁癖——第 5 章整章在讲「怎么才有资格说一个性能结论」，所以讲义自己也守同一条规矩。**看到没有出处的数字，就当它不存在。**
