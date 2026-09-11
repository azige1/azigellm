# General Agent 技术体系全解析

## 一、方向定义

### 1.1 General Agent 是什么

General Agent 是基座模型 agent 能力矩阵中的一条独立能力线，指模型在通用工具环境中完成多步专业事务的能力：调用工具（function calling / MCP server）、操作文件系统、检索与核验网页信息、执行脚本、生成结构化产物（报表、文档、配置），并在长程执行中维持规划、从错误中恢复。任务域覆盖办公事务、信息检索、数据分析、调度自动化、系统运维、沟通协作等非编码为主的专业工作。

基座模型的 agent 能力通常拆为四条并行能力线，General Agent 与其余三条的分界如下：

| 能力线 | 环境与动作空间 | 典型评测 | 与 General Agent 的区别 |
|---|---|---|---|
| Coding Agent | 代码仓库 + 终端，动作以编辑/测试为主 | SWE-bench、Terminal-Bench | 任务有测试用例作天然 verifier，环境结构单一 |
| Search Agent | 搜索引擎 + 网页抓取 | BrowseComp、GAIA（部分） | 动作空间窄（搜索/浏览），核心难点是信息核验与上下文管理 |
| GUI Agent | 屏幕截图 + 鼠标键盘 | OSWorld、AndroidWorld | 感知以视觉为主，动作是坐标级操作 |
| **General Agent** | **MCP / 多工具 + 文件系统 + 网络，动作空间开放** | **BFCL、tau-bench、GAIA、MCP 系列 bench** | **工具集合不固定、任务域开放、验证无天然信号，三者同时成立** |

该划分并非严格互斥：General Agent 通常包含浅层的搜索与终端操作，Coding / Search Agent 可视为其在特定环境上的深化。各厂 tech report 的 benchmark 分组（Coding Agent / General Agent / Search 或 Reasoning 分列）沿用同一口径。

### 1.2 该方向的三个核心矛盾

方向的全部技术工作可以还原为三个矛盾的处理：

1. **任务开放性与可验证性的矛盾**。编码任务有单元测试、数学任务有唯一答案，General Agent 的任务（生成一份分析报告、整理一批文件）没有天然验证信号。因此该方向一半以上的工作量在构造验证：为合成任务配 verification code、为不可程序化验证的维度配 rubric 与 LLM judge、训练专用 reward model。
2. **工具生态多样性与训练分布覆盖的矛盾**。真实部署时工具集合由客户端决定（不同 MCP server 组合、不同 scaffold），训练时无法穷举。因此需要工具与任务的规模化合成（task scaling / environment scaling）、格式扩增（同一轨迹渲染为多种工具调用语法）、多 scaffold 轨迹混合，使模型学到工具调用的语义抽象而非特定格式记忆。
3. **长程轨迹与信用分配的矛盾**。任务轨迹普遍 10–100 轮，outcome-level reward 对早期正确步骤连坐惩罚，信号稀疏。因此出现 turn-level reward、过程奖励模型（Agentic PRM）、失败轨迹 advantage 处理等专门机制。

### 1.3 方向兴起的原因

2024 年前基座模型竞争集中在对话与推理跑分；2025 年起各厂将 agent 能力列为基座核心卖点，训练资源向 agent 后训练倾斜。驱动因素有三：其一，token 消耗结构变化，agent 场景单任务消耗数万至数百万 token，是推理收入的主体；其二，MCP（Model Context Protocol，Anthropic 于 2024 年 11 月发布）统一了工具接入协议，工具生态的爆发使「通用工具调用」成为可规模化训练与评测的对象；其三，可验证奖励 RL（RLVR）在数学与代码上验证成功后，向开放任务域外推的自然路径就是「为开放任务构造 verifier」，General Agent 是这条外推路径的主战场。

---

## 二、训练技术体系

General Agent 的训练管线为标准多阶段：预训练/中训练注入 → 任务与环境规模化 → SFT 轨迹蒸馏 → agentic RL → 奖励模型与过程监督 → 评测闭环。以下按阶段归纳公开可考的标准做法与关键设计，公开来源见第三节厂商披露与第五节开源工作。

### 2.1 预训练 / 中训练：agent 行为先验注入

- **数据重心迁移**：中训练数据从 Long-CoT 推理为主转向 agent 轨迹为主，将工具调用、长程规划、错误恢复的行为模式写入 next-token prediction 先验。轨迹来源为强教师模型在真实 scaffold（CLI agent、IDE agent 等）中的执行记录，含 think 与 nothink 两种模式。公开实例：Tongyi DeepResearch 的 agentic CPT（AgentFounder）、Qwen3-Coder-Next 的环境反馈 mid-training、CWM 的执行轨迹 mid-training。
- **多段过滤管线**：agent 轨迹对格式错误零容忍（标签不配对、工具调用语法错乱会被当作正确模式学入），过滤管线可达十余个阶段：格式转换、去重、scaffold 来源识别、规则引擎（trajectory 级 + message 级规则，检测思考标签完整度、身份泄露、caller 字段完整性）、tool error mask、n-gram 去污染。
- **格式扩增**：准备多种工具调用模板（XML / JSON / markdown 方言），每条轨迹随机渲染多份，迫使模型将工具调用学习为语义概念而非字符串模式。
- **配比体系**：多级乘法权重（source × domain × length × 有无参考答案 × 精选度），以小模型 ablation 的量化反馈驱动采样概率调优。

### 2.2 Task Scaling：任务规模化合成

任务合成是该方向区别于 Coding Agent 的核心环节（Coding 任务可从 GitHub issue-PR 对天然获取，General 任务必须合成）。标准做法：

- **要素组合合成**：人设（如 [persona-hub](https://github.com/tencent-ailab/persona-hub) 的公开人设库）× 任务类别体系（覆盖办公/检索/分析/自动化/运维等数十小类）× 原子动作集（搜索、下载、读写文件、执行脚本等），由 LLM 组装为具体任务规范，输出任务描述、期望产物与验证点。公开的厂商实例为 Kimi K2 的工具、agent、任务、轨迹四层合成与 DeepSeek V3.2 的 1,827 环境 / 85K prompt 合成盘；开源对应物为 Toucan（真实 MCP server 执行合成）。
- **两种合成策略**：自顶向下（主题+画像派生任务，覆盖面广但可能超出环境能力）与自底向上（以环境中已沉淀的 skill 为 seed，要求新任务复用 skill 并叠加复杂度，可验证性强），实践中两者混合。
- **verification code 同步合成**：每个任务由 LLM 生成程序化验证函数，经静态编译检查 + LLM judge 逻辑审查 + 多次重试。验证粒度需与任务难度匹配——简单任务配少量断言、复杂任务配多维验证点，过紧导致 reward 偏低、过松导致 RL 学到捷径。该环节是整条管线的质量瓶颈（MiniMax M2 系披露的 artifact 对齐奖励属同类做法）。
- **多阶段筛选**：合成 → 语义去重（embedding 相似度阈值合并）→ 任务质量 judge（自洽性 / 答案泄露 / 可验证性）→ verifier 合成与校验 → 双重审查。
- **双模型交叉过滤定难度**：用一强一弱两个模型分别对每个任务多次 rollout，强模型通过率过低的任务判为任务本身有缺陷、弱模型通过率过高的判为无训练价值、弱模型反超强模型的判为 verifier 异常，仅保留强模型显著优于弱模型的能力边界任务。强模型做质量校准、弱模型做难度定位。

### 2.3 Environment Scaling：环境规模化

- **环境构成**：容器 + 领域 package + 工作文件 + memory/skill。memory 与 skill 是个性化的主要载体，重要性高于 package 安装（package 在单一领域内数量有限，且安装存在依赖冲突）；公开对应物为 Anthropic Agent Skills（文件夹加说明文件组织可复用技能）与 Manus 的文件系统作外部记忆的上下文工程。
- **环境养成流程**：确定 domain → 初始化用户偏好 → agent 执行任务并交互 → 沉淀可复用 skill（分析脚本）、沉淀 memory（偏好与历史决策）、迭代用户画像。用户偏好维度随任务交互逐次累积，而非一次性定义。
- **环境的双重用途**：作为 RL 训练的真实执行环境；作为自底向上任务合成的 seed 来源（skill 库越厚，可合成的复合任务越复杂）。
- **公开的环境合成工作**：DeepSeek V3.2（1,827 环境）与 Qwen（数百万环境口径）的管线未开源，开源对应物为 EnvScaler、Agent World Model、SETA 与 TermiGen（见 5.3）。

### 2.4 SFT：轨迹蒸馏与行为克隆

- **数据来源**：强教师模型在合成任务上的 rollout 轨迹，按 verification 结果与质量筛选；loss 仅计算 assistant / tool 消息 token。
- **失败轨迹的价值**：只保留成功轨迹存在 selection bias（困难任务因失败率高被系统性过滤，形成恶性循环），且整体失败的轨迹中多数步骤仍正确，保留一定比例失败轨迹可缓解该偏差。更进一步的推论：对 SFT 最优的数据策略可能压缩 RL 探索空间，SFT 应为 RL 保留策略多样性。
- **错误注入**：教师生成轨迹时以固定概率主动注入错误并强制恢复，使轨迹包含 error → diagnosis → correction 完整纠错环。动机是自然失败的错误类型分布不均，注入可强制覆盖全部失败模式；TermiGen 的 Generator-Critic 注错为公开来源。
- **经验性过滤的边界**：mask 错误轮次、按 tool/turn/token 分布分层筛选等精细过滤的收益需以消融验证；mask 错误轮会移除错误轮次携带的失败恢复信息。

### 2.5 RL：可验证奖励与长程优化

- **奖励来源**：verification code 的程序化判分为主，主观维度（报告质量、建议合理性）以 LLM judge / rubric 补充，形成 hybrid 判分。
- **算法与关键设计**：GRPO 系为主流。两个 agent 特有的处理——rollout 因环境故障失败的轨迹将 advantage 置零（不参与梯度），与「主动给出的差回答」区分；timeout 前被强制输出的 forced answer 对 reward 降权重塑。
- **Infra 难点**：长轨迹（128K 级 context）+ 工具调用外部依赖（超时/失败需鲁棒处理）+ 异步 rollout 与训练并行。主流开源基础为 [verl](https://github.com/volcengine/verl) + vLLM/SGLang，配 datapool、router、agent-sdk 等组件。

### 2.6 奖励模型：从 outcome 到 process

该方向奖励工程的演进主线是奖励信号从轨迹级走向轮级：

- **Self-critique / 自评估奖励**：同一模型兼任 generator 与 critic，critic 按 rubric 对不可验证任务打分，与可验证奖励联合训练（Kimi K2 的 Joint RL 为公开实例）。适用边界取决于 critic 能否从结果本身获得可靠评判信号：数学（可验算）与确定性工具调用（API/参数/返回值可检查）适用，开放式写作与领域判断不适用。已知失效模式为 critic 学会无差别差评，需以评分上下限截断修复。
- **Rubric-based RM**：为每个任务自动生成评估检查点（rubric），LLM 按 rubric 逐项打分（公开工作见 Rubrics as Rewards、Rubric Anchors）。主要风险是 length hacking（长输出骗高分），需在偏好数据采样阶段打破长度与质量的相关性。
- **Turn-level 稠密奖励**：两条路线——turn 级 on-policy 蒸馏（强模型在每轮基于相同 prefix 生成参考行为，与 policy 行为对比给分，难点是路径等价性判断）与 Agentic PRM（训练可调用工具做实际验证的专用过程奖励模型，难点是 turn 级监督信号获取，可从 outcome reward 经 credit assignment 反推）。相关公开工作见 [Turn-Level Reward for Multi-Turn RL Agents](https://arxiv.org/abs/2505.11821)。

### 2.7 评测闭环

- **自建 benchmark 构建方法论**：从大规模合成任务中经双模型交叉过滤 + 人工审核抽出数百道，按「纯程序化验证」与「code + LLM judge 混合验证」分类维护；benchmark 上强弱模型形成合理梯度是任务集有效性的检验标准。
- **数据闭环**：从 agent 环境持续收集新交互数据，经过滤、处理、标准化管线快速回流下一轮 SFT，数据新鲜度优先于数据量（UI-TARS-2 披露的数据飞轮为公开实例）。
- **当前公认短板**：执行能力强、判断能力弱——模型难以判定自己是否做对（自降验收标准、错误假设长期存活、测试通过但输出错误），self-verification 与 persistence 是 RL 的下一个目标。

---

## 三、业界各厂实现

本节事实以官方 tech report / system card / 官方 blog 为准；经由第三方站点转述的 2026 年数字均标注来源，引用前应以官方发布页复核。

### 3.1 总览

| 厂商 | agent 主线模型 | 训练侧披露要点 | 主报 benchmark |
|---|---|---|---|
| OpenAI | o3 / GPT-5 系、ChatGPT Agent | 工具使用内嵌 RL、RL 算力扩展规律、不可完成任务的弃权奖励 | GAIA、BrowseComp、HLE、τ²-bench、GDPval、SWE-bench |
| Anthropic | Claude 4 / 4.5 / 4.6 系 | interleaved thinking、computer use、MCP 协议发起方；RL 环境与奖励设计未披露 | τ²-bench、OSWorld、SWE-bench、Terminal-Bench |
| Google DeepMind | Gemini 2.0–3 系 | 原生工具使用入训、后训练 RL 算力加码；环境细节未披露 | Terminal-Bench、Vending-Bench、WebVoyager |
| Meta | Llama 3 / 4 | Llama 3 论文含完整 tool use 训练章节（SFT + 偏好优化 + 零样本工具泛化）；Llama 4 后停更 | BFCL、Nexus、API-Bank |
| xAI | Grok 4 / 4.6 | 原生工具使用纳入 RL 回路、RL 算力扩至预训练量级 | HLE、GDPval 类 |
| 月之暗面 | Kimi K2 / K2 Thinking / K2.x | 大规模 agentic 数据合成（工具/任务/轨迹/rubric 四层）、RLVR + self-critique 联合 RL | τ²-bench、ACEBench、HLE w/ tools、BrowseComp、SWE-bench |
| DeepSeek | V3.1 / V3.2 | thinking with tools、1,827 个合成环境 + 85K prompt、增强版 GRPO、后训练算力超预训练 10% | τ²-bench、BrowseComp、Terminal-Bench、MCP-Mark、Toolathlon 类 |
| 智谱 | GLM-4.5 / 4.6 / 5 | ARC 定位、三路专家蒸馏合并、slime 异步 RL infra、GLM-5 异步 agentic RL 算法 | τ²-bench、BFCL、SWE-bench、Terminal-Bench、MCP-Atlas |
| 阿里 | Qwen3 / Qwen3.5 | agentic CPT（Tongyi DeepResearch 管线回流）、可执行环境规模化 RL、数百万 agent 环境异步 RL | BFCL、τ²-bench、SWE-bench |
| MiniMax | M2 系 | interleaved thinking 一等建模、agent 驱动数据管线、Forge agent 原生 RL 系统、20 万+ 环境 RL | SWE-bench、Multi-SWE、BrowseComp |
| 字节 Seed | Doubao / UI-TARS 系 | 通用基座披露少；GUI 线（UI-TARS-2）披露数据飞轮 + 多轮 RL + 混合环境统一沙箱 | OSWorld、AndroidWorld、Online-Mind2Web |

### 3.2 国际厂商

**OpenAI**。训练侧披露为各家中最具体。Deep Research（2025-02）披露采用跨领域困难浏览与推理任务上的端到端 RL，模型学习规划、执行多步轨迹并按实时信息回溯调整（[官方](https://openai.com/index/introducing-deep-research/)）。o3 / o4-mini（2025-04）披露通过 RL 训练工具使用——同时学习调用方式与调用时机的决策，并给出 RL 训练算力与性能呈预训练同型扩展规律的表述（[官方](https://openai.com/index/introducing-o3-and-o4-mini/)）。ChatGPT Agent（2025-07）将浏览器操作（Operator）、检索综合（Deep Research）与终端/代码执行合并为单一 agentic 模型（[官方](https://openai.com/index/introducing-chatgpt-agent/)）。GPT-5 system card（2025-08）披露一项针对性训练干预：将模型置于部分或完全不可完成的任务中，对如实承认无法完成给予奖励，并训练其对环境故障保持鲁棒——这是可验证奖励之外「反奖励欺骗」训练的公开实例（[system card](https://cdn.openai.com/gpt-5-system-card.pdf)）。评测侧：Deep Research 首发 HLE 26.6%、GAIA 榜首；ChatGPT Agent 报 HLE 41.6、BrowseComp 68.9；GPT-5.2（2025-12）报 GDPval 对 44 职业专业人员胜/平率 70.9%、τ²-bench Telecom 98.7（[官方](https://openai.com/index/introducing-gpt-5-2/)）。

**Anthropic**。对 agentic 训练方法的披露显著少于 OpenAI，system card 以评测与安全论证为主；「以大规模内部任务环境做可验证奖励 RL」属外界普遍判断而非官方口径。可确认的官方事实：2024-10 以 Claude 3.5 Sonnet 首发 computer use（截图—坐标操作范式，[官方](https://www.anthropic.com/news/3-5-models-and-computer-use)）；2024-11 发起 MCP 开放协议，后被 OpenAI、Google 采纳（[官方](https://www.anthropic.com/news/model-context-protocol)）；Claude 4（2025-05）引入 interleaved thinking（工具调用之间插入推理并按中间结果决策）、并行工具调用与基于本地文件的跨任务记忆（[官方](https://www.anthropic.com/news/claude-4)）。评测侧：Sonnet 4.5（2025-09）报 OSWorld 61.4、τ²-bench Telecom 98.0，并同步发布 Agent SDK；Opus 4.5（2025-11）报 OSWorld 66.3、SWE-bench Verified 80.9（[官方](https://www.anthropic.com/news/claude-opus-4-5)）。产品线（Claude Code / Agent SDK / Skills）构成其 agent 生态主线。

**Google DeepMind**。Gemini 2.0（2024-12）定位「agentic era 的模型」，原生工具使用进入训练目标，配套 Project Astra / Mariner / Jules 原型（[官方](https://blog.google/innovation-and-ai/models-and-research/google-deepmind/google-gemini-ai-update-december-2024/)）。Gemini 2.5 技术报告（[arXiv 2507.06261](https://arxiv.org/abs/2507.06261)）将下一代 agentic 能力列为标题级目标，披露后训练显著增加 RL 算力并以 thinking（含工具使用中的推理）为核心范式；环境与奖励细节未披露。Gemini 3（2025-11）将 agentic tool use 列为模型卡一级评测维度，报 Terminal-Bench 2.0 54.2、Vending-Bench 2 净值 $5,478（长程一致性评测）。

**Meta**。Llama 3 论文（[arXiv 2407.21783](https://arxiv.org/abs/2407.21783)）是开源阵营最完整的工具训练披露：以人工与合成工具调用轨迹做 SFT + 偏好优化，训练搜索/Python 解释器/数学引擎的多步调用，并专门训练对未见工具定义的零样本调用；评测覆盖 BFCL、Nexus、API-Bank。Llama 4（2025-04）称面向 tool-calling 与 agentic 系统优化但无方法论文，此后无实质性训练侧披露。

**其余**。xAI 披露 Grok 4 在 20 万 GPU 集群上将 RL 算力扩至预训练量级，并将原生工具使用直接纳入 RL 训练回路，采用可验证奖励与基于模型的奖励（[官方](https://x.ai/news/grok-4)）。Mistral 的 Magistral 以论文披露纯 RL 细节（GRPO 改造），通用 agent 线披露有限。Amazon Nova 技术报告披露按「工具选择—调用—结果评估」链路设计 agent 能力并以 BFCL 评测，2026 年经 Nova Forge 开放多轮 RL 定制训练路径。

### 3.3 国内厂商

**月之暗面 Kimi**。K2 tech report（[arXiv 2507.20534](https://arxiv.org/abs/2507.20534)，题为 Kimi K2: Open Agentic Intelligence）是国内首个以 agentic 能力为主线的基座报告：1T 总参 / 32B 激活 MoE。训练方法两大支柱：其一，大规模 agentic 数据合成管线，系统性生成工具、agent、任务、轨迹四层要素（3,000+ 真实 MCP 工具 + 20,000+ 合成工具），任务附带 rubric 用于轨迹质量过滤；其二，Joint RL，可验证奖励与 self-critique rubric 奖励联合训练，后者将对齐信号扩展至不可验证的开放域任务。报告分数 τ²-Bench 66.1、ACEBench (En) 76.5、SWE-bench Verified 65.8。K2 Thinking（2025-11，[HF](https://huggingface.co/moonshotai/Kimi-K2-Thinking)）端到端训练思考与工具调用交错，可连续执行 200–300 次工具调用不漂移，采用 INT4 QAT 与并行多轨迹反思聚合（Heavy Mode），报 HLE w/ tools 44.9、BrowseComp 60.2。2026 年后续版本引入多 agent 协同（数百个领域子 agent、数千步自主运行，来源为[第三方转述](https://deepinfra.com/blog/kimi-k2-6-model-overview)）。

**DeepSeek**。V3.1（2025-08）为 agent 能力转折点：混合思考模式，后训练强化工具调用与多轮搜索。V3.2（[arXiv 2512.02556](https://arxiv.org/abs/2512.02556)、[官方](https://api-docs.deepseek.com/news/news251201/)）首次将思考并入工具调用（thinking with tools），训练披露具体：冷启动统一推理与工具调用轨迹；大规模 agentic 任务合成——1,827 个环境、85,000+ 复杂 prompt（Code Agent 24,667 / Search Agent 50,275 / General Agent 4,417 / Code Interpreter 5,908）；RL 采用增强版 GRPO（无偏 KL 估计、off-policy 序列掩码、Keep Routing / Keep Sampling Mask）；后训练算力超过预训练的 10%。上下文管理策略为推理内容跨工具输出保留、新用户消息到达时丢弃。报 τ²-Bench 80.3、BrowseComp 51.4、Terminal-Bench 2.0 46.4、MCP-Mark 38.0。

**智谱 GLM**。GLM-4.5 tech report（[arXiv 2508.06471](https://arxiv.org/abs/2508.06471)）以 ARC（Agentic / Reasoning / Coding）为基座定位：355B-A32B MoE；后训练采用 agentic / reasoning / general 三路专家模型迭代蒸馏合并再 RL，RL 基础设施为自研开源 [slime](https://github.com/THUDM/slime)（异步、生成与训练解耦）。报 τ-bench 70.1、SWE-bench Verified 64.2。GLM-5（[arXiv 2602.15763](https://arxiv.org/abs/2602.15763)）提出面向复杂长程交互的异步 agentic RL 算法，报 SWE-bench Verified 77.8、Terminal-Bench 2.0 56.2，多项 agentic bench 居开源首位。

**阿里 Qwen**。Qwen3 报告（[arXiv 2505.09388](https://arxiv.org/abs/2505.09388)）中 agentic 训练披露有限（BFCL v3 70.8），方法细节的补充披露在 Tongyi DeepResearch 报告（[arXiv 2510.24701](https://arxiv.org/abs/2510.24701)：agentic CPT + 合成任务 + on-policy 异步 rollout RL），该管线经验回流基座；[Qwen-Agent](https://github.com/QwenLM/Qwen-Agent) 承载 function calling / MCP / 代码解释器封装。Qwen3-Coder 线的大规模可执行环境 RL 构成通用 agent 的共享底座。2026 年版本的官方口径为「在数百万 agent 环境上以可扩展异步 RL 框架训练规划、工具使用与自我纠错」（[第三方转述](https://www.cnbc.com/2026/02/17/china-alibaba-qwen-ai-agent-latest-model.html)）。

**MiniMax**。agentic 主线自 M2（2025-10，229.9B-A9.8B，[GitHub](https://github.com/MiniMax-AI/MiniMax-M2)）确立：interleaved thinking 作为一等 agent 建模原则，每次工具调用后保留完整推理状态跨轮前传，消融显示对深搜索与软件工程类长程任务增益最大（[官方说明](https://www.minimax.io/news/why-is-interleaved-thinking-important-for-m2)）。M2 系列报告披露三组件：agent 驱动数据管线（可执行 workspace 内生成可验证轨迹、artifact 对齐奖励）、Forge agent 原生 RL 系统（windowed-FIFO 调度、前缀树合并）、后期版本的自演化（自主修改训练与 scaffold）。2026 年版本称在 20 万+ 真实环境上大规模 RL（[官方](https://www.minimax.io/news/minimax-m25)）。

**字节跳动 Seed**。通用基座侧披露少、产品化强；agent 训练披露集中在 GUI 线：UI-TARS-2（[arXiv 2509.02544](https://arxiv.org/abs/2509.02544)）披露数据飞轮持续产轨迹、稳定化多轮 RL、GUI + 文件系统 + 终端混合环境与统一沙箱大规模 rollout，报 OSWorld 47.5、AndroidWorld 73.3。

**其他**。腾讯混元 2026 年重建预训练与 RL 基础设施并开源主打 agent 能力的 MoE 基座，无方法级 tech report；百度文心标注智能体规划与工具应用能力，agentic 后训练细节未披露；阶跃星辰面向生产级 agent 优化 API / 浏览器 / 终端 / Office 工具调用稳定性并兼容 CLI agent 框架；讯飞星火的披露以部署工程为主。

### 3.4 横向归纳

1. **方法收敛**：头部厂商配方趋同为三件套——合成环境与任务规模化（从千级环境到数十万级环境）、可验证奖励为主 + rubric / self-critique 奖励覆盖开放域、异步 agentic RL 基础设施（slime / Forge / verl 系）。与本文第二节的管线逐环节对应。
2. **披露梯度**：OpenAI 与 xAI 明确承认工具内嵌 RL 与算力扩展规律，DeepSeek 与 Kimi 披露到环境数量与奖励构成的粒度，Google 承认范式但不披露环境，Anthropic 基本只披露行为特性与评测，Meta 在 Llama 3 后停更。各家均未披露轨迹数据配比，该层依赖开源复现推断。
3. **评测收敛**：τ²-bench、BFCL、SWE-bench Verified、Terminal-Bench 2.0、BrowseComp、HLE w/ tools 与 MCP 系（MCP-Atlas / MCP-Mark / Toolathlon）构成 2025–2026 报告的公共集合；GAIA 因趋于饱和逐渐退出旗舰报告。
4. **前沿位移**：竞争点由单轨迹工具调用转向长程一致性（context compaction、记忆文件、上下文管理策略）、错误恢复与反欺骗（弃权奖励、环境故障鲁棒性）、测试时扩展（并行轨迹聚合、工具调用步数扩展）与多 agent 编排。

---

## 四、Benchmark 体系

判分方式是该方向 benchmark 的核心区分字段：程序化 verifier（状态比对 / AST / 校验脚本）可信度最高，LLM judge 覆盖不可形式化的语义维度，混合判分用规则保结构正确性、用 judge 覆盖语义等价。

### 4.1 工具调用类

| Benchmark | 出品方 / 年份 | 任务形态 | 判分方式 |
|---|---|---|---|
| [BFCL v3/v4](https://gorilla.cs.berkeley.edu/leaderboard.html) | UC Berkeley, 2024–2025 | v1 单轮、v3 多轮状态化环境、v4 扩展 agentic 场景（搜索/记忆/格式敏感性） | 程序化：AST 匹配 + 可执行校验 + 状态比对 |
| [tau-bench](https://arxiv.org/abs/2406.12045) / [tau2-bench](https://arxiv.org/abs/2506.07982) | Sierra, 2024–2025 | retail / airline / telecom 三域，agent 与 LLM 模拟用户多轮对话并操作数据库 | 程序化：终态数据库比对 + 关键信息检查，pass^k 度量一致性 |
| [ACEBench](https://arxiv.org/abs/2501.12851) | USTC + 华为, 2025 | Normal / Special（不完整或歧义指令）/ Agent 三类，8 大域中英双语 | 程序化：AST + 规则 + 沙箱模拟 |
| [API-Bank](https://arxiv.org/abs/2304.08244) | 阿里, 2023 | 314 段多轮对话、753 次调用，调用/检索/规划三级 | 混合：调用程序化判定 + 回复 ROUGE |
| [ToolBench/ToolEval](https://arxiv.org/abs/2307.16789) | 清华 OpenBMB, 2023 | 16K+ RapidAPI 真实 API 组合指令 | LLM judge（pass rate + win rate）；因真实 API 失效逐渐退役 |
| [ToolSandbox](https://arxiv.org/abs/2408.04682) | Apple, 2024 | 1,032 用例，状态化执行 + 工具间隐式状态依赖 + on-policy 模拟用户 | 程序化：Milestone / Minefield 逐里程碑轨迹相似度 |
| [ComplexFuncBench](https://arxiv.org/abs/2501.10132) | 智谱, 2025 | 1,000 条单轮多步、隐式参数推理、长参数、128K 上下文 | 混合：规则匹配 + 响应比对 + LLM 辅助参数匹配 |
| [Seal-Tools](https://arxiv.org/abs/2405.08355) / [NexusBench](https://github.com/nexusflowai/NexusBench) | 2024 | 合成 API 池 / 真实 API 的单调用、并行、嵌套 | 程序化：格式 / 工具选择 / 参数三维匹配 |

### 4.2 MCP 类（2025–2026 新增密集区）

| Benchmark | 出品方 / 年份 | 任务形态 | 判分方式 | 代表分数 |
|---|---|---|---|---|
| [MCP-Universe](https://arxiv.org/abs/2508.14704) | Salesforce, 2025 | 231 任务、6 域真实 MCP server | 程序化：format / static / dynamic 三类 execution-based evaluator | GPT-5 43.7% |
| [MCPMark](https://arxiv.org/abs/2509.24002) | EvalSys 等, 2025 | 127 任务（Notion / GitHub / Filesystem / PostgreSQL / Playwright），平均 16.2 轮 | 程序化：初始状态 + 终态断言脚本 | gpt-5-medium pass@1 52.6% |
| [LiveMCPBench](https://arxiv.org/abs/2508.01780) | 中科院系, 2025 | 95 任务、70 server、527 工具，考察大规模工具检索与路由 | LLM judge（与人工一致率约 81%） | Claude-Sonnet-4 79.0% |
| [MCP-Bench](https://arxiv.org/abs/2508.20453) | Accenture, 2025 | 28 server、250 工具跨域编排 | 混合：规则判 schema + LLM judge 判完成/规划质量 | — |
| [MCPEval](https://arxiv.org/abs/2507.12806) | Salesforce, 2025 | 自动生成任务的评测框架（非固定题库） | 混合：参考轨迹匹配 + LLM 评审 | — |
| [MCP-Atlas](https://labs.scale.com/leaderboard/mcp_atlas) | Scale AI, 2026 | 1,000 个人工编写任务、36 server、220 工具，任务不点名工具，需 3–6 步跨 server 编排 | 混合：最终答案按要点覆盖率打分（≥75% 计通过） | 榜首约 83–88%（第三方快照） |

### 4.3 综合 General Agent 类

| Benchmark | 出品方 / 年份 | 任务形态 | 判分方式 | 状态 |
|---|---|---|---|---|
| [GAIA](https://arxiv.org/abs/2311.12983) | Meta + HF, 2023 | 466 个现实问题，需浏览/多模态/工具 | 程序化：短答案 quasi-exact match | 榜首约 75%，趋于饱和 |
| [AgentBench](https://arxiv.org/abs/2308.03688) | 清华, 2023 | 8 环境（OS/DB/知识图谱/网购等） | 程序化：各环境自带 success 判定 | 评测重心已转向后继工作 |
| [AgentBoard](https://arxiv.org/abs/2401.13178) | 复旦/HKU 等, 2024 | 9 类任务、1,013 环境实例，多轮部分可观测 | 程序化：子目标标注计算 progress rate | 过程性度量的代表 |
| [Toolathlon](https://arxiv.org/abs/2510.25726) | HKUST-NLP, 2025 | 108 任务，真实软件初始状态（课程系统/邮箱/表格），平均约 20 轮调用——办公事务自动化的代表评测 | 程序化：每任务专用脚本核对环境终态 | Claude-4.5-Sonnet 38.6%，区分度高 |
| [HLE w/ tools](https://arxiv.org/abs/2501.14249) | CAIS + Scale, 2025 | 2,500 道专家级封闭题的 agent 化跑法（允许搜索/代码执行），与 closed-book 口径不可混排 | 混合：短答案 LLM 判等价 | 榜首 50%+ |
| [xbench](https://github.com/xbench-ai/xbench-evals) | 红杉中国, 2025 | evergreen 动态换题：DeepSearch 每期 100 题 + 职业域真实任务 | 混合：短答案判定 + 业务产出评估，题目加密防污染 | 分期发布 |
| [BrowseComp](https://arxiv.org/abs/2504.12516) | OpenAI, 2025 | 1,266 道难检索、易验证的搜索题 | 程序化：短答案精确匹配 | 榜首 90%+，接近饱和 |

### 4.4 环境执行类（相邻方向定位）

[OSWorld](https://arxiv.org/abs/2404.07972)（369 个真实桌面 GUI 任务，脚本校验终态）、[WebArena](https://arxiv.org/abs/2307.13854)（812 个自托管网站任务）、[AppWorld](https://arxiv.org/abs/2407.18901)（750 个跨 9 App 的 API 编排任务，单元测试式状态断言，办公/生活事务自动化的另一代表）、[Terminal-Bench](https://github.com/laude-institute/terminal-bench)（Docker 化终端长程任务）、[SpreadsheetBench](https://arxiv.org/abs/2406.14991)（912 个真实表格任务，单元格级比对）、[SWE-bench](https://arxiv.org/abs/2310.06770)（单元测试判分的可验证范式锚点）。

### 4.5 过程 / 奖励评测类

评测对象是「评估者」本身，服务于 reward 设计与 LLM judge 可靠性论证：[JudgeBench](https://arxiv.org/abs/2410.12784)（以客观正确性为标签的困难 response pair，GPT-4o 级 judge 仅略高于随机）、[RewardBench 2](https://arxiv.org/abs/2506.01937)（六子集 best-of-4 准确率，暂无独立 agent 子集）、[AgentRewardBench](https://arxiv.org/abs/2504.08942)（1,302 条 web agent 轨迹专家标注，结论为 12 个 judge 无一全面占优、规则判分系统性低估 agent 成功率）、[Agent-RewardBench](https://arxiv.org/abs/2506.21252)（多模态 agent step 级奖励，感知/规划/安全三维）。turn-level reward 方向尚无统一权威榜。

### 4.6 判分格局归纳

2025–2026 年的增量集中在两处：MCP 真实环境评测（MCPMark / MCP-Universe / MCP-Atlas）与长程办公事务执行（Toolathlon / AppWorld），共同特征是任务量小（约 100–1,000）、判分强程序化、榜首成功率低于 60%，构成当前区分度最高的评测带。早期依赖 LLM judge 的评测（ToolEval）逐渐退役，混合判分成为开放任务域的主流方案——与第二节评测闭环中程序化验证与混合验证的两类分池同构。

---

## 五、开源论文与代码仓库

按数据合成、agent RL、环境规模化、开源基座四条线组织。

### 5.1 工具调用数据合成与 SFT

| 工作 | 链接 | Star 量级 | 核心方法 |
|---|---|---|---|
| ToolLLM / ToolBench | [arXiv 2307.16789](https://arxiv.org/abs/2307.16789) · [GitHub](https://github.com/OpenBMB/ToolBench) | ~5.7k | 16K RapidAPI 真实 API 构造指令与解路径（DFSDT 决策树搜索） |
| Gorilla | [arXiv 2305.15334](https://arxiv.org/abs/2305.15334) · [GitHub](https://github.com/ShishirPatil/gorilla) | ~13k | API 文档检索增强的调用生成；仓库同时承载 BFCL |
| APIGen / xLAM | [arXiv 2406.18518](https://arxiv.org/abs/2406.18518) · [GitHub](https://github.com/SalesforceAIResearch/xLAM) | ~0.6k | 格式/执行/语义三级校验的函数调用数据管线；APIGen-MT 扩展至多轮模拟交互 |
| ToolACE | [arXiv 2409.00920](https://arxiv.org/abs/2409.00920) · [HF](https://huggingface.co/Team-ACE) | 数据集为主 | 自演化 API 池（26K+ API）+ 多 agent 对话合成 + 双层校验 |
| Hammer | [arXiv 2410.04587](https://arxiv.org/abs/2410.04587) · [GitHub](https://github.com/MadeAgents/Hammer) | ~0.1k | 函数掩码 + 不相关性增强，提升函数名扰动下的鲁棒性 |
| AgentInstruct | [arXiv 2407.03502](https://arxiv.org/abs/2407.03502) | 无官方仓库 | 原始文档为种子的多 agent 生成流（25M 样本） |
| AgentTuning / AgentLM | [arXiv 2310.12823](https://arxiv.org/abs/2310.12823) · [GitHub](https://github.com/THUDM/AgentTuning) | ~1.5k | 六环境交互轨迹 + 通用数据混合微调，早期 agent SFT 代表 |
| Agent-FLAN | [arXiv 2403.12881](https://arxiv.org/abs/2403.12881) · [GitHub](https://github.com/InternLM/Agent-FLAN) | ~0.4k | agent 语料按能力维度拆分重组 + 负样本抑制幻觉 |
| FireAct | [arXiv 2310.05915](https://arxiv.org/abs/2310.05915) · [GitHub](https://github.com/anchen1011/FireAct) | ~0.3k | 多格式轨迹（ReAct/CoT/Reflexion）蒸馏小模型 |
| persona-hub | [arXiv 2406.20094](https://arxiv.org/abs/2406.20094) · [GitHub](https://github.com/tencent-ailab/persona-hub) | ~1.6k | 十亿级人设驱动多样性合成；第二节任务合成的人设要素来源 |
| Toucan-1.5M | [arXiv 2510.01179](https://arxiv.org/abs/2510.01179) · [GitHub](https://github.com/TheAgentArk/Toucan) | ~0.3k | 495 个真实 MCP server / 2,000+ 工具真实执行合成 1.5M 轨迹，目前最接近 K2 agentic 数据管线的开源复现 |
| AgentTrek | [arXiv 2412.09605](https://arxiv.org/abs/2412.09605) · [GitHub](https://github.com/xlang-ai/AgentTrek) | <0.1k | 网页教程引导的真实环境轨迹回放合成（ICLR'25 Spotlight） |
| AgentBank | [arXiv 2410.07706](https://arxiv.org/abs/2410.07706) | — | 50K+ 跨 16 任务轨迹库微调路线 |
| Nemotron-Agentic-v1 | [HF 数据集](https://huggingface.co/datasets/nvidia/Nemotron-Agentic-v1) | — | NVIDIA 开放的 agentic 后训练数据（函数调用/终端/SWE 轨迹），CC-BY-4.0，业界数据开放度最高 |

### 5.2 Agent RL（可验证奖励 / 多轮 RL）

算法线（按信用分配粒度演进排列）：

| 工作 | 链接 | 核心方法 |
|---|---|---|
| Search-R1 | [arXiv 2503.09516](https://arxiv.org/abs/2503.09516) · [GitHub](https://github.com/PeterGriffinJin/Search-R1)（~5.4k） | 检索引擎作环境、outcome reward + retrieved-token masking 的多轮 RL 基线 |
| ToolRL | [arXiv 2504.13958](https://arxiv.org/abs/2504.13958) · [GitHub](https://github.com/qiancheng0/ToolRL)（~0.5k） | 工具调用 reward 设计的系统研究（格式分 + 名称/参数细粒度匹配分） |
| ARTIST | [arXiv 2505.01441](https://arxiv.org/abs/2505.01441) | 推理与工具调用交替的 agentic RL 统一框架（微软，未开源） |
| RAGEN / StarPO | [arXiv 2504.20073](https://arxiv.org/abs/2504.20073) · [GitHub](https://github.com/RAGEN-AI/RAGEN)（~2.5k） | 多轮轨迹级优化，分析 echo trap 等训练不稳定模式 |
| Turn-Level Reward | [arXiv 2505.11821](https://arxiv.org/abs/2505.11821) | turn 级 reward 系统对比（verifiable + LLM judge），密集信号一致优于稀疏终局信号 |
| GiGPO | [arXiv 2505.10978](https://arxiv.org/abs/2505.10978) · [GitHub](https://github.com/langfengQ/verl-agent)（~2.3k） | episode 级 + 锚点状态分组 step 级双层相对优势，critic-free 细粒度信用分配（NeurIPS'25） |
| SWEET-RL | [arXiv 2503.15478](https://arxiv.org/abs/2503.15478) · [GitHub](https://github.com/facebookresearch/sweet_rl)（~0.3k） | 训练时特权信息训练 step 级 critic，配套多轮协作基准 ColBench |

框架线（2025–2026 年的主要训练框架，共同特征为异步 rollout 与生成/训练解耦）：

- [verl](https://github.com/volcengine/verl)（~15k）：HybridFlow 架构，原生支持 multi-turn + tool calling，多数 agent RL 工作的底座；agent 定制分支 [verl-agent](https://github.com/langfengQ/verl-agent) 支持任意长 horizon 并内置多环境。
- [AReaL](https://github.com/inclusionAI/AReaL)（~3k，[arXiv 2505.24298](https://arxiv.org/abs/2505.24298)）：全异步 RL 系统（interruptible rollout + staleness 控制），2026 年演进为面向 agent 应用的 AReaL-lite。
- [AgentGym-RL](https://github.com/WooooDyy/AgentGym-RL)（~0.9k，[arXiv 2509.08755](https://arxiv.org/abs/2509.08755)）：模块化多环境 RL + ScalingInter-RL（训练早期限制交互轮数、逐步放宽 horizon 的课程调度）。
- [SkyRL](https://github.com/NovaSky-AI/SkyRL)（~2.2k）：train / gym / agent 三层，面向 SWE-Bench 级真实长程任务的开源在线 RL。
- [ART](https://github.com/OpenPipe/ART)（~11k）：GRPO 嵌入任意 Python agent 应用的工程化封装，RULER（LLM judge 相对打分）替代人工 reward 设计。
- 同类还有 ROLL（阿里）、[slime](https://github.com/THUDM/slime)（智谱，GLM 系 RL 底座）、rLLM（Agentica）。

### 5.3 环境与任务规模化

- 统一环境层：[AgentGym](https://arxiv.org/abs/2406.04151)（14 环境统一 HTTP 接口 + AgentEvol 自演化）、[AppWorld](https://github.com/stanfordnlp/appworld)（9 个模拟 App / 457 API 可执行世界，状态级单元测试）、[tau-bench 环境](https://github.com/sierra-research/tau-bench)（用户模拟器 + 领域策略约束）、[ToolSandbox](https://github.com/apple/ToolSandbox)（有状态对话式沙盒）。
- MCP 生态：[mcp-agent](https://github.com/lastmile-ai/mcp-agent)（~8.5k，MCP 之上的组合式框架）、[LiveMCPBench](https://github.com/icip-cas/LiveMCPBench)（检索错误占失败近半的诊断结论）。
- 环境合成（2026 年增量集中点）：厂商侧 DeepSeek V3.2（1,827 环境）与 Qwen（数百万环境口径）的合成管线均未开源；开源对应物为 [EnvScaler](https://arxiv.org/abs/2601.05808)（程序化合成 191 环境 / 7K 场景用于 SFT+RL）、Agent-World（[arXiv 2604.18292](https://arxiv.org/abs/2604.18292)）、Agent World Model（[arXiv 2602.10090](https://arxiv.org/abs/2602.10090)，以模型为无限合成环境）。
- 终端/CLI 线：[Terminal-Bench harness](https://github.com/laude-institute/terminal-bench)（Docker 化终端评测标准）、[TermiGen](https://arxiv.org/abs/2602.07274)（多 agent 迭代生成任务 + 容器，Generator-Critic 注错合成纠错轨迹——第二节 2.4 错误注入机制的公开来源）、SETA（[arXiv 2607.10891](https://arxiv.org/abs/2607.10891)，4,500+ 验证环境）。

### 5.4 开源 agentic 基座模型

| 模型 | 规模 / License | agent 相关卖点 |
|---|---|---|
| Kimi K2 系 | 1T-A32B MoE / Modified MIT | 大规模 agentic 数据合成 + Joint RL，开源 agentic 基座标杆（[GitHub](https://github.com/MoonshotAI/Kimi-K2) ~11k） |
| GLM-4.5 / 4.6 | 355B-A32B MoE / MIT | ARC 三合一定位，4.6 公开 CC-Bench 全部评测轨迹 |
| Qwen3 系 | 0.6B–235B（A22B MoE 旗舰）/ Apache-2.0 | 原生工具调用 + Qwen-Agent 生态，agent RL 研究默认底座 |
| MiniMax-M2 | 230B-A10B MoE / MIT | interleaved thinking + shell/browser/MCP 长链工具调用 |
| DeepSeek-V3.x | 685B-A37B MoE / MIT | 稀疏注意力支撑长程 agent 推理成本，thinking with tools |
| xLAM 系列 | 1B–8x22B / CC-BY-NC 为主 | APIGen 数据驱动的函数调用专精家族 |
| ToolACE-8B / Hammer 2.x | 8B / 0.5B–7B | 小参数量函数调用专精，BFCL 上与大模型同档 |
| Nemotron 系 | 多尺寸 / Open Model License | 配套开放 agentic 后训练数据集，数据开放度最高 |

### 5.5 推理编排框架（非训练重点）

OpenHands（~60k）、smolagents（~29k）、AgentScope（~30k）、CAMEL（~18k）及其衍生 OWL（~20k）属推理/编排层：不产训练信号、不含 RL 管线；在训练语境下的价值是充当 rollout harness（多家 SWE / agent RL 工作复用 OpenHands scaffold 采集轨迹）。

### 5.6 开源生态归纳

1. 数据合成主线的演进路径：模拟 API + 单轮（ToolBench / APIGen）、真实 MCP 执行 + 多轮轨迹（ToolACE、Toucan、Nemotron-Agentic）、合成对象从轨迹上移到环境本身（EnvScaler / TermiGen / SETA）。该路径与厂商披露（K2 四层合成、DeepSeek 环境合成）同向，开源侧滞后约半年至一年。
2. Agent RL 的算法争点集中在信用分配粒度：trajectory 级、turn 级、step 级锚点分组、特权信息 critic 四档依次细化；框架侧的共同特征是异步 rollout 与交互轮数课程。
3. 开源基座在 agentic 维度形成 1T / 355B / 235B / 230B 四档 MoE 格局，license 以 MIT / Apache 为主；训练数据与环境合成管线的开放程度成为新的差异轴——NVIDIA 数据最开放，DeepSeek 与 Qwen 保留管线。
