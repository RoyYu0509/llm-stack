# Assignment A — Inference Serving Engine

一份 CS336 assignment handout 风格的**回顾式讲义**，覆盖本仓库 Week 1–3。中文正文，技术术语保持英文。

配套的姊妹篇是 `LLM-system-project/docs/handout/assignment-b.tex`（《Closing the Loop》），两份可以独立阅读。

## 构建

需要 XeLaTeX（`ctex` + `xeCJK`）。macOS 上装 MacTeX 即可，不需要 `-shell-escape`。

```bash
latexmk -xelatex assignment-a.tex
```

不带参考答案的**空白自测版**：

```bash
latexmk -xelatex -jobname=assignment-a-quiz -pretex='\def\HANDOUTQUIZ{}' -usepretex assignment-a.tex
```

清理中间产物：

```bash
latexmk -c
```

## 文件

| 文件 | 作用 |
|---|---|
| `assignment-a.tex` | 主文档，只负责 `\include` 各章 |
| `handout.sty` | 共享样式。**与姊妹篇是同一份的两个副本**，顶部有 `CANONICAL` 标记和版本串 |
| `sections/*.tex` | 每章一个文件，被 `\include`（支持 `\includeonly` 单章快编） |
| `refs.bib` | 参考文献，仓库内部文档也作为一等公民引用 |
| `assignment-a.xrefs` | 构建时生成的稳定锚点导出，供姊妹篇同步 |

## 字体

`handout.sty` 显式指定 Songti SC / Heiti SC / Menlo / TeX Gyre，并用 `fontset=none` 关掉 ctex 的自动选择——缺字体会在构建时报错，而不是静默替换成别的。macOS 之外的机器需要改那几行。

## 跨讲义引用

姊妹篇用 `\extref{a:some-key}` 引用这份讲义的小节。机制是：这里构建时导出 `assignment-a.xrefs`，把它复制到 `LLM-system-project/docs/handout/xref-a.tex` 即可。

不用 `xr-hyper`，因为那要求构建时能读到对方的 `.aux` —— 两份讲义在不同的 git 仓库里，clone 其中一个的人拿不到另一个。改了本讲义的章节结构之后记得重新同步；忘了同步的话，姊妹篇构建时会警告并把引用排成 `[??: key]`。
