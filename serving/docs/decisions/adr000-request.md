# ADR-000: Request & TokenizedData 设计结构

**Date**: [04-11-2026] | **Status**: Accepted

## Context
Request 的 data class 怎么设计?

## Decision
### TokenizedData 和 Request class 的关系
**✨ Final Decision**
把 Tokenized Text 从 Request Class 中分出来 单开一个 TokenizedData Class. 这样区分 Receiver层 和 Engine层.

**NOTE:** Both Request层 和 Engine层 都需要 import Request Class 和 TokenizedData Class. 所以虽然是 data class 分开了, 但是 在实际的 module 层, Request Receiver module 和 Engine module 都需要 import 这两个 class 来进行 request 的接收, tokenization, 和 inference.

**Alternatives**

> 1. 把 Tokenized 作为一个 member data 直接放在 Request Class 里. 
**问题:** 但这样 Request Class 就包含了两层含义：既有原始的 text data interaction，也有 tokenized data interaction. 不够清晰.
**好处**: 结构简单, 不需要额外的 mapping, 而且在做 inference 的时候, 不用额外增加一个 logics 去 update Request Class 里的 status.

### Request Class 的 member data 设计
**✨ Final Decision**
User input 的是 prompt_text 和 user_id, RequestReceiver 在收到这两个输入后, 会在内部创建一个 RequestData 对象. 

Request Class 中包括下面👇这些 member data:
> a. user_id (str): The ID of the user who sent the request.
	b. request_id (str): The unique ID of the request. (用 RequestReceiver 的 class variable 来做到)
	c. timestamp (float): The time when the request was created.
	d. status (Enum): The current status of the message.
	e. prompt_text (str): The raw text prompts associated with the message.

**Alternatives**
> 1. status 不用 Enum 而是用 String 或者 int 来表示.
**问题:** 用 string 的话不安全, 因为 Enum 本身是可以  validate input 的, 而 string 的话必须要额外设置 validation. 用 int 的话不够直观, 需要额外的 mapping 来解释每个 int 代表什么状态.
> 2. User 的 prompt 在外部就被 wrap 好, RequestReceiver 直接接收 RequestData.
**问题:** 在外部调用方创建 RequestData 的时候还没有 request_id. 这样就等于传一个不完整的 RequestData，再在 receiver 里 overwrite request_id。这很别扭。


### TokenizedData Class 的设计
**✨ Final Decision**
TokenizedData Class 中包括下面👇这些 member data
> a. request_id (str): The unique ID of the request.
    b. tokens (List[int]): The list of token IDs after tokenization.

**Alternatives**
> 1. TokenizedData Class 中包含 Request Class 的所有 member data.
**问题:** TokenizedData Class 只应该专注于 存储 tokenized data, 不包含其他的 field, 比如 timestamp, user_id 等等. 只保留一个 request_id 来 map 对应的Request 就够了. 因为 TokenizedData 是要进 Engine 的, 只需要关注 tokenized data 就好, 不需要其他的 field. 这样职责更清晰.
> 2. TokenizedData Class 中加一个 token length field 来表示 token 的长度.
**问题:** 不需要, 直接 overwritten __len__ method 就可以了, 没必要额外加一个 field 来存 token length. 这样设计更简洁.


### Request Class 的 serialization 和 deserialization
**✨ Final Decision**
Request Class 的 serialization 和 deserialization 用 Pydantic 来完成 (因为自带 type checking).

**Alternatives**
> 1. 用 Python 内置的 dumping JSON 来做 serialization 和 deserialization.
**问题:** 没有 type checking, 必须要我们自己额外做 type validation.

> 2. 用 Python Dict 来做 serialization 和 deserialization.
**问题:** 同样没有 type checking, 需要额外做 validation. 而且 off board 还要额外做 dict to json 的转换, 不够直接.

## Consequences
1. Request Receiver 层需要 import Request class 和 TokenizedData class 来进行 request 的接收和 tokenization. 
2. Engine 层需要 import Request class 和 TokenizedData class 来进行 inference, TokenizedData 用来做 computation, Request class 用 reference 用来 update request 的 status (pending → prefilling → decoding → completed).
