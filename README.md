# 电商智能客服 Agent

一个**离线可跑**的电商售后客服系统：政策不再是被检索的文本，而是**结构化的可执行条件**；模型只负责把判定结论说成人话。生成之后还有一道核验（claim 逐句核验、数字溯源、合规拦截），最后按校准概率决定「自动答复 / 追问 / 转人工」。

> 默认零外部依赖：自研状态图引擎 + 内存向量库 + 规则/模板作答，`python -m ecom_cs.cli demo` 直接出结果。
> 配好 OpenAI 兼容 Key / Milvus / BGE 后即可切真实模型与真实向量库，代码路径不变。

---

## 为什么不是「RAG + Prompt」

简历里那版系统优化的是**信息获取**：混合检索、rerank、评测集，Recall@5 98%。但客服业务买的是**处置结论**——「这单能不能退、退多少钱、多久到账、能不能改地址」。这五个问题里，只有第一个是检索问题，另外四个是**规则判定 + 写操作 + 时效约束**。

所以这一版把重心放在：

| 问题 | 这一版的做法 |
| --- | --- |
| 政策是长文本，检索到也说不准 | 条款拆成 `conditions/exceptions/action` 的 DSL，规则引擎三值求值（真/假/**未知**） |
| 只会答不会办 | 6 个工具，其中 3 个写操作带二次确认、幂等键、审计、失败补偿 |
| 生成后没人管 | claim 逐句核验 + 数字必须可溯源 + PII/承诺/提示注入拦截 |
| 只会答或转人工 | 三选一：`answer / clarify / escalate`，缺哪个事实就问哪个 |
| 置信度是拍的 | Platt 校准 + 保形阈值（`alpha` 控制错误率上界）+ 四信号融合 |
| 成本不可控 | 模型级联 L0 模板 / L1 小模型 / L2 大模型 + 语义缓存（仅政策类） |
| 评测只看检索 | 端到端 30 条用例：decision / intent / clause / verify 四个通过率 + 成本与延迟 |

## 架构

```
用户输入
  │
  ├─ guard        注入/批量数据请求拦截
  ├─ confirm      上一轮写操作的「确认 / 取消」
  ├─ understand   归一化 → 别名改写 → 意图(9类) → 槽位 → 路由(5条)
  ├─ cache        仅政策类、无订单号时查语义缓存
  ├─ dispatch ─┬─ tools     订单/物流/售后 只读工具
  │            ├─ retrieve  BM25 + 向量 → RRF → 重排
  │            ├─ smalltalk
  │            └─ escalate
  ├─ evaluate     事实 → 规则引擎(三值) → 主题收窄 → 置信度 → 校准 → 决策
  ├─ clarify / escalate / generate
  ├─ generate     级联选层 → 规则/模板 或 LLM 生成
  ├─ verify       claim 核验 → 数字守卫 → 合规复检（不过则转人工）
  └─ finalize     上下文压缩、成本记账、未解决问题入 gap 池、审计
```

## 快速开始

```bash
cd ecom-cs-agent
python -m ecom_cs.cli demo        # 9 组脚本化场景（改地址二次确认、已发货拒绝、注入、投诉…）
python -m ecom_cs.cli chat        # 交互式对话（/trace 看上一轮完整轨迹）
python -m ecom_cs.cli eval        # 跑评测集并打印指标
python -m ecom_cs.cli calibrate   # 用评测集拟合置信度校准与保形阈值
python -m ecom_cs.cli serve       # 起 FastAPI：/chat、/chat/stream、/stats
python -m unittest discover -s tests   # 54 个测试
```

要求 Python ≥ 3.10，只用标准库 +（可选）FastAPI/httpx/numpy。没有 Key、没有 Milvus、没有 GPU 也能跑通全链路。

## 评测（当前仓库状态）

`configs/eval/scenarios.jsonl`，30 条：退货资格 ×6、运费、物流 ×2、订单状态、越权、退款时效与渠道、多轮指代、发票 ×2、改地址（含确认轮）、质量问题、注入 ×2、投诉、闲聊、超范围、会员、价保、缓存命中。

| 指标 | 值 |
| --- | --- |
| decision_accuracy | 1.000 |
| intent_accuracy | 1.000 |
| clause_hit | 1.000 |
| verify_pass_rate | 1.000 |
| escalate_rate | 0.200 |
| clarify_rate | 0.167 |
| 平均延迟 (mock) | ~5 ms |

> 这是**自建小评测集**，不是公开 benchmark；它的价值是「改一行规则就能看到哪类问题退化」，而不是对外宣称的 SOTA。指标口径见 `docs/DESIGN.md`。

### 同一套用例接真实模型（deepseek-chat）

| 指标 | mock | deepseek-chat |
| --- | --- | --- |
| decision_accuracy | 1.000 | 1.000 |
| intent_accuracy | 1.000 | 1.000 |
| clause_hit | 1.000 | 1.000 |
| verify_pass_rate | 1.000 | 1.000 |
| 平均延迟 | ~5 ms | ~1.47 s（p50 ≈ 6 ms，绝大多数轮次被 L0/L1 吃掉） |
| 33 轮成本（按内置价格表估算） | 0 | ≈ $0.0077（约 $0.00023/轮） |

接真实模型的第一轮**并不**是满分（decision 0.967 / verify 0.833）：模型会在正确结论后面补一句「信息不足，建议转人工」、会把结构化字段名写进回复、会把引擎算出的动作短语说得不像模板。修法分两层——生成侧删掉越权话术（`clean_llm_reply`），核验侧把动作短语登记为可信事实（`trusted`）。这两处修复的来龙去脉写在 `docs/DESIGN.md` 第 4 节。

## 目录

```
ecom_cs/
  config.py           配置（全部可被 ECOM_CS_* 环境变量覆盖）
  types.py            意图/路由/决策/信号/会话状态
  kb/                 政策 DSL、规则引擎、父子块切分、事实构建、知识闭环
  retrieval/          BM25、向量库、RRF 融合、重排、改写/指代消解、检索服务
  nlu/                意图词典+质心分类、注入检测、槽位抽取
  tools/              只读工具 ×3、写操作 ×3（幂等/确认/审计/补偿）
  graph/              自研状态图引擎（节点返回下一跳，router 只读状态）
  orchestrator/       节点、图装配、依赖装配、会话存储、对外的 ChatService
  verify/             claim 核验、数字守卫、合规、置信度、校准与保形阈值
  cost/               级联、语义缓存、成本账本
  eval/               评测执行器与指标
  api/                FastAPI 接入
configs/
  policies/*.json     8 个政策文档、23 条结构化条款
  eval/scenarios.jsonl
docs/DESIGN.md        设计取舍与踩坑记录
docs/INTERVIEW.md     面试问答手册
```

## 切真实模型 / 真实向量库

```bash
cp .env.example .env      # 填 ECOM_CS_LLM_PROVIDER=openai 等
```

| 能力 | 离线默认 | 生产 |
| --- | --- | --- |
| 生成 | `HeuristicAnswerer`（模板，引用条款 ID） | `LLMAnswerer`（OpenAI 兼容，强约束 prompt） |
| 向量 | `HashingEmbedder`（512 维特征哈希） | `BGEEmbedder`（bge-m3 / bge-small-zh） |
| 向量库 | `MemoryVectorStore` | `MilvusVectorStore` |
| 重排 | 覆盖率 + 关键词 + 数字一致性 | `CrossEncoderReranker`（bge-reranker） |

`Settings.ablation(**flags)` 支持关掉任一环节（`enable_rerank=False`、`enable_claims=False`…），用来做消融对照，也是面试里「你怎么证明这层有用」的答案。

## 设计文档与面试手册

- [`docs/DESIGN.md`](docs/DESIGN.md)：七个改进逐条讲「原方案的问题 → 这一版怎么做 → 代价是什么 → 怎么验证」，含指标口径、已知坑、演进方向。
- [`docs/INTERVIEW.md`](docs/INTERVIEW.md)：按「项目讲什么 / 为什么这么做 / 数字怎么来的 / 哪里不行」整理的答辩问答。

## 已知边界

- 会话状态在进程内存，单机可用；多副本要换 Redis。
- 政策和订单是种子数据（`ecom_cs/db/seed.py`，以运行当天为基准动态生成），接真实库只需实现 `repo.upsert_*` 的导入脚本。
- 没有 Web 前端（按需求只交付服务端与 CLI）。
- 中文分词用 bigram，够用但不等于领域分词；上线前建议换 jieba/pkuseg 并重训意图词典。
