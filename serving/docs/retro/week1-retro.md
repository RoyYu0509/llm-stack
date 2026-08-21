# Retro Week 1
   
## What Went Well
1. Request Receiver 的基本功能实现了。能并发接收多个 request，正确 tokenize，并放入 pending queue 里了。
2. 知道写 design doc 的时候, 重要的不止是写出design, 更重要的是 写清楚其他的 alternatives 和 为什么不选那些 alternatives.
3. 知道怎么写 test 需要测试 edge case, 怎么读 pytest 的 console log, 怎么 debug test 里报错的 assert statement.
4. 知道了 asyncio 中 create_task 和 await 的区别, 以及 event loop 的调度模型, gather 是在干什么, 以及整体的 concurrency 是什么 + 为什么可以让代码跑的更快. (都记在notion中了)

## What did not go well
### 问题1. Test 没跑就 Push
**Description:**
    多次提交没跑过的 test。 缺 test_ prefix、try/except 空路径、验证 coroutine 对象而不是返回值。这些都是跑一遍就能发现的。养成 "写完跑，绿了再交" 的习惯。

**Root Cause:**
没有 pre-commit checklist

**Actions:**
1. 每次 push 前，本地跑 pytest -v 截图绿色输出再提交
2. Test 报错之后，先自己 debug. Narrow down 到具体的 Functionion/Lines + Error Type 之后再提问。
3. 从 week 2 开始, keep 一个 pre-commit checklist, push 之前过一遍, 确保每条都满足了.

**Week 2 Check:**
Week 2 retro 里不会出现没跑就push的情况，目标 0 次.
Week 2 中所有的提问都是 具体的, narrow down 到 function/line + error type 的。

### 问题2. Design Doc 和代码不一致
**Description:**
Desgin Doc / ADR 和代码不一致。 

**Root Cause:**
根本原因是没有 根据 Design Doc 来写代码, 还是自己怎么想怎么写. 要记住 engineer 是一个团队, Design Doc 就是整个 team 唯一的 reference.

**Actions:**
1. 在和 manager 讨论并决定好最终的 design 后, 才写进 design doc 里.
2. 写代码的时候要照着, 不能自己想怎么写怎么写, 不然和 manager 的讨论就会 diverge, 代码和 design doc 就不一致了.
3. 如果中途发现 design doc 有之前没考虑到的地方, 要重新和 manager 讨论, 决定好之后再更新 design doc. 不能自己擅自改 design doc 了, 还不告诉 manager, 这样就完全失去 design doc 的意义了.
4. 代码写完了, 再回过头来检查一遍 design doc, 确保 design doc 和代码是一致的. 这样和 整个 team 都是 aligned 的.

**Week 2 Check:**
Week 2 retro 里不会出现 design doc 和代码不一致的情况，目标 0 次.
如果要改 design doc, 先和 manager 讨论，达成一致后再改 design doc. 不能自己擅自改 design doc 了, 还不告诉 manager.

### 问题3. Design Doc 中写 alternatives 的时候没有写为什么不选那些 alternatives
**Description:**
在写 alternative 的时候, 最重要的就是写出来, 每一个 alternative 为什么不行. 因为 一个 NO 的理由, 比 十个 YES 的理由 更直接, 更能 justify 我们的选择.

**Root Cause:**
觉得 final decision 是更关键的, 把 design doc 当作 function annotation 了. 但是其实 Design Doc 最重要的作用是: 记录我们为什么做这个 design decision, 以及为什么不选其他的 alternatives. 而不是去记录一个 function 怎么用.

**Actions:**
1. 永远要写 为什么不选 Alternative, 这样自己后面一眼就能够看懂. 并且能在后面面试的时候, 说出来每一个 option 的 trade-off.

**Week 2 Check:**
Week 2 的 design doc 里面, 每一个 alternative 都有写清楚为什么不选这个 alternative. 不能有任何一个 alternative 没有写为什么不选.

## One improvement
永远先自己跑一遍test,再问, 等绿了再提交。

## Behavioral Improvements
### 问问题的时候要具体
**Description:**
在有看不懂的内容的时候, 不要直接求助: 
>”我不知道这个代码在干什么“, 而是要说 ”代码中的某一个 function/line 我不太理解, 我现在对它的理解是 ..., 请问我理解有问题吗?“ 

这样子提问更具体, 而且直接narrow down到具体的问题, 节约了一轮沟通轮次.

