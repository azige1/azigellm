# Coding Agent 技术体系全解析

## 一、方向定义

### 1.1 Coding Agent 是什么

Coding Agent 是基座模型 agent 能力矩阵中以软件工程为任务域的能力线：模型在代码仓库加终端的可执行环境中自主完成 issue 修复、功能实现、测试编写、仓库级开发与终端运维任务，动作空间为文件读写编辑、命令执行与测试运行，产物为可被测试判定的代码变更。任务形态按环境与产出分四档：issue 级修复（SWE-bench 类，给定仓库与问题描述产出补丁）、终端任务（Terminal-Bench 类，在容器内完成配置、构建、数据处理等操作）、仓库级从零构建（NL2Repo 类，从自然语言规格建整仓）、长程工程任务（性能优化、移植、研究工程，小时级至数十小时预算）。

与其余能力线的分界：

| 能力线 | 判定信号 | 与 Coding Agent 的区别 |
|---|---|---|
| General Agent | verification code + rubric 混合 | 任务域开放、验证需专门构造；Coding 任务有单元测试作天然 verifier |
| Search Agent | 短答案精确匹配 / 报告 rubric | 判据在外部语料且环境非平稳；Coding 环境封闭、可完全复现 |
| GUI Agent | 环境终态脚本 | 感知以视觉为主；Coding 观测为文本与程序输出 |

单元测试提供的 fail-to-pass 二值信号使 Coding Agent 成为四条能力线中可验证奖励最完备的一条：任务从真实 GitHub issue-PR 对天然获取、判分由测试执行给出、环境由容器完全固化。该属性决定了它是各厂 agentic RL 投入最重、评测竞争最激烈的方向。

### 1.2 该方向的四个核心矛盾

1. **环境构建成本与任务规模的矛盾**。每个训练任务需要一个可执行环境（依赖安装、构建通过、测试可跑），真实仓库的环境配置成功率低且维护成本高。对应的技术工作是上游采集流水线（仓库元数据批量采集与可执行环境配对、质量分层）与任务合成（从历史轨迹、文本种子或仓库快照反向构造任务），把任务供给从人工策划推向自动化生产。
2. **测试作为奖励信号的不完备性**。弱测试使错误补丁通过、过强或泄漏的测试使模型学到面向测试编程与作弊（直接改测试、硬编码期望值、检索答案）。对应的技术工作是测试生成的双重约束（测试必须由规格锚定、且能区分正确与错误实现）、作弊审计（以执行记录而非文本声明判定）与评测可信度治理。
3. **执行能力与判断能力的断层**。长程任务中模型能写出单模块正确的代码，但难以判定自己是否做对：自降验收标准、错误假设跨阶段存活、测试全过而输出错误、误差无声放大。该短板在长程评测中系统性暴露，是 RL 下一阶段（self-verification、persistence）的目标，也是当前 harness 层（规格驱动、分模块验收）存在的原因。
4. **scaffold 耦合与跨环境泛化**。训练轨迹携带特定 scaffold 的工具集与交互协议，单一 scaffold 训练的模型在其他客户端上退化。对应的做法是多 scaffold 轨迹混合、工具调用协议的格式扩增与统一协议迁移。

### 1.3 方向兴起的原因

评测侧，SWE-bench（2023）把软件工程任务转化为可自动判分的标准问题，使该方向成为 agentic 能力的公共标尺；产品侧，编码助手是 agent 能力变现最直接的场景，Claude Code、Codex、Cursor 等产品的 token 消耗构成推理收入主体，且产品公司开始自训模型反向进入基座竞争；方法侧，执行反馈天然可验证，agentic RL 的算法与 infra 创新（异步 rollout、长轨迹信用分配、环境沙箱）多以该方向为首发场景。

---

## 二、训练技术体系

管线为：中训练注入 → 上游采集与环境构建 → 任务合成 → SFT 轨迹蒸馏与过滤 → agentic RL → 评测与作弊审计 → 数据飞轮。以下按环节归纳公开可考的标准做法，各环节的公开来源见第三节厂商披露与第五节开源工作。

### 2.1 中训练注入

- **数据重心**：中训练数据由推理为主转向执行轨迹与 agent 轨迹为主，轨迹来自强教师模型在多种 scaffold（CLI agent、终端 agent）中的执行记录，含 think 与 nothink 两种模式，将工具调用、长程规划与错误恢复写入 next-token 先验。公开实例：CWM 以 Python 解释器逐步执行轨迹与可执行仓库镜像中的 agent 轨迹做代码世界建模 mid-training；Qwen3-Coder-Next 将环境反馈学习前移到 mid-training；Kimi-Dev 与 KAT 系均含该层。
- **过滤管线**：agent 轨迹对格式错误零容忍（标签不配对、工具调用语法错乱会被当作正确模式学入），过滤按格式转换、去重、scaffold 来源识别、轨迹级与消息级规则检查、工具错误处理、n-gram 去污染多段串联。规则引擎采用否定驱动的质量保证：由人定义不合格模式并自动剔除，不要求定义完美轨迹。
- **格式扩增与配比**：同一轨迹以多种工具调用模板随机渲染，使模型将工具调用学为语义概念而非特定字符串模式；数据配比按来源、领域、长度、有无参考答案等维度设置多级乘法权重，以小模型消融反馈驱动调优。
- **多 scaffold 采样**：轨迹来源覆盖多种 scaffold，使模型学习跨 scaffold 的元能力而非单一客户端的操作记忆。

### 2.2 上游采集与环境构建

- **批量元数据到可执行环境配对**：以仓库元数据（star、语言、活跃度）批量筛选候选库，自动尝试依赖安装与测试执行，产出仓库与可执行环境的配对；支线为细粒度主题定向爬取补充覆盖面。公开实现见 SWE-Gym（人工预装依赖）、SWE-smith 与 R2E-Gym（自动化环境构建）、SWE-bench-Live 的 RepoLaunch 与 SWE-rebench 的持续采集管线。
- **可解性预筛**：任务进入训练池前以强模型多次采样验证可解，剔除全部失败（疑似环境或测试缺陷）与全部成功（无训练价值）的任务，SWE-smith、R2E-Gym 与 DeepSWE 均采用该做法，与 Search / General 方向的强弱双模型交叉过滤同型。
- **执行后端**：本地容器起步，规模化后迁移远程沙箱池（Qwen3-Coder 披露 2 万并行环境，Cursor Composer 披露数十万并发沙箱）；环境失败与模型失败严格分离，环境故障不计入失败率。

### 2.3 任务合成

任务合成分三条路线，对应三类任务形态：

- **issue 级：从 GitHub 历史挖掘为主**。issue-PR 对天然携带任务描述、参考补丁与回归测试，核心工作在环境配对与测试有效性校验；合成补充路线为在仓库快照上注入缺陷再生成任务（SWE-smith 的 LM 错误重写、AST 过程化变异、PR 回退、bug 组合四法）。多语言扩展的公开路径为 agent 驱动的自动环境搭建（SWE-Factory 多智能体建环境、SWE-bench-Live 的 RepoLaunch）与社区共建的容器化实例（Multi-SWE-RL 七语言）。
- **终端任务：轨迹反推与文本种子两条路线**。轨迹反推以真实执行轨迹为种子反推出任务描述与验收脚本，天然可执行、可验证，同时训练定位能力——任务只给目标状态，模型需自行发现从当前状态到目标的路径；文本种子从文档、教程等素材正向构造任务与容器环境，覆盖面广。公开实现见 TermiGen（多智能体迭代合成任务与容器）、Endless Terminals、terminal-bench-env 等。
- **仓库级：规格驱动的多阶段流水线**。以自然语言规格为起点生成测试与任务，测试需满足双重约束——断言锚定规格条目、且能区分正确实现与占位实现；流程包含种子仓库索引、规格合成、测试生成、种子筛选、rollout、SFT 合并对齐过滤与事后规格覆盖率回补。公开对应物为 Commit0（按规格从零实现整库）、SWE-Flow（从单测反推依赖图生成测试驱动开发实例）与 SWE-Dev（为无测试实例合成测试用例）。

### 2.4 SFT：轨迹蒸馏与过滤

- **质量控制的两层四象限**：实例级（环境是否可用、题目是否合理）与轨迹级（轨迹是否高质量）分层，各层内规则检查（schema 完整、幻觉工具、循环调用、截断、复读）与模型判分（过程合理性、目标一致性）两类方法组合。分层的依据是好题可产坏轨迹、坏题上的正确轨迹亦不可用——环境有问题时验证结果不可信，须先保实例合格再筛轨迹。
- **教师轨迹与失败重采**：强教师模型在任务上 rollout，按测试结果与质量筛选（SWE-Gym、SWE-smith 的拒绝采样微调均以轨迹通过测试为过滤条件）；教师全部失败的题可给出只指方向、不泄漏补丁的提示后重采，提示轨迹单独标记。
- **失败轨迹与错误注入**：只保留成功轨迹存在选择偏差（困难任务被系统性过滤），保留一定比例失败轨迹可缓解该偏差；教师生成时按概率主动注入错误并强制恢复，使轨迹覆盖失败模式的纠错环（TermiGen 的 Generator-Critic 注错为公开来源）。SFT 数据策略需为 RL 保留策略多样性。
- **loss 规则**：仅 assistant / tool 消息计损失，观测 token 零梯度；错误轮 mask 存在两种处理——KAT 采用 Error-Masked SFT，另一类做法保留错误轮以维持错误恢复的监督信号，取舍需经消融确认。
- **训练组织**：轨迹带思维链优于无思维链；工具调用协议在训练与部署间保持一致；epoch 数以评测曲线选定，长轨迹 SFT 中过训出现较早。

### 2.5 RL

- **奖励**：以测试执行结果为主奖励（fail-to-pass），辅以格式与规范项；奖励作弊需在奖励层防御（禁改测试文件、作弊行为记负）。SWE-RL 以生成补丁与真实开发者补丁的相似度作轻量规则奖励，Kimi-Dev 以 Docker 内全测试套件通过作稀疏奖励，Cursor Composer 同时激励正确性、工具选择效率与并行化。
- **训练环路**：verl 系框架加 agent recipe（SkyRL、rLLM、verl-agent）——scaffold 以 API 形态接入训练侧（agent 以为在调推理服务，实际由训练框架路由），动作协议精简（命令执行与终止两类），episode 结束后跑验证器判分；可解性筛选后的任务池按难度分带，训练中动态刷新。
- **Infra 要点**：长轨迹（数十至上百轮、128K 级上下文）、远程沙箱池承载并发 rollout、异步生成与训练解耦（slime、SeamlessFlow、AReaL）；失败轨迹 advantage 置零、超长截断轨迹与 forced answer 单独处理（DeepSWE 的 compact filtering），与 General 方向同型。

### 2.6 评测与作弊审计

- **评测分层**：基座层（补全、单测通过率）与能力层（SWE 系、终端系、仓库级）分开维护；外部公开基准作对外口径，自建集作选型判据；pass@1 口径必须固定 scaffold 与预算，跨 scaffold 分数不可比。
- **作弊定义与审计**：以实际执行记录（改动的文件、执行的命令、访问的 URL）判定，文本提及不算；NIST CAISI 的案例汇编记录了读取容器内未来提交、直接修改测试等类型。作弊轨迹通常获得高于正常轨迹的判分，不剔除则高分作弊轨迹主导训练；按低分筛出的子集不能用于估计作弊与得分的相关性——该抽样系统性排除高分作弊样本，会得到相反结论。
- **失败模式体系**：按定位错误、过早终止、测试理解偏差、环境误判等类别做失败归因与得分相关性分析，结论回写任务构造与过滤规则。
- **防污染**：评测集题面与仓库版本隔离、n-gram 去污染进中训练管线、持续更新型基准对冲静态集污染。

### 2.7 数据飞轮

线上回流构成闭环：从 agent 产品环境收集新交互数据，经过滤、处理、标准化管线快速回流下一轮 SFT，数据新鲜度优先于数据量（UI-TARS-2 与 MiniMax M2 系报告均披露数据飞轮式管线）。回流数据与合成数据的角色分工与 Search 方向一致：回流校正分布偏移，合成保证难度与覆盖。

### 2.8 长程短板与 harness 层

长程工程任务（数十小时预算的性能优化、移植、研究工程）暴露的失败模式高度一致：测试全过而输出结构性错误、自降验收标准、错误假设跨多阶段存活、优化方向追错、单模块误差经管线放大数十倍、写下的方法论不被自己执行。归纳为执行能力强、判断能力弱，与 RE-Bench 上长预算下人类反超的结论互证。在模型补齐 self-verification 之前，harness 层以规格驱动开发承接：任务拆为可独立验证的模块、每模块定义数值验收标准、逐模块对齐、追踪误差传播路径——把人的判断逻辑编码为模型无法绕过的检查点。模型自检能力增强后，harness 结构随之精简，人工干预从逐步盯防退到方向定义与关键节点审查。

---

## 三、业界各厂实现

本节事实以官方 tech report / system card / 官方 blog 为准；经第三方转述的 2026 年数字均标注来源，引用前应以官方发布页复核。SWE-bench Verified 分数受 scaffold 与采样口径影响，跨厂商比较需先核对 harness（见 4.5）。

### 3.1 总览

| 厂商 | coding agent 主线 | 训练侧披露要点 | 代表分数（自报口径） |
|---|---|---|---|
| Anthropic | Claude 系 + Claude Code | 训练方法不披露；披露 hybrid reasoning、extended thinking with tool use | Opus 4.5 SWE-V 80.9（首个破 80） |
| OpenAI | codex-1 / Codex 线 | 官方确认真实编码任务上的 RL、跨 context window 的长程任务训练与 compaction | GPT-5.1-Codex-Max SWE-V 77.9、TB2 58.1 |
| Google DeepMind | Gemini + Jules | 基本无训练细节，分数与能力描述层级 | Gemini 3 Pro SWE-V 76.2、TB2 54.2 |
| Meta | SWE-RL、CWM | 论文级全披露（规则奖励 RL、执行轨迹 mid-training） | CWM-32B SWE-V 53.9（TTS 65.8） |
| Cursor / Cognition / Mistral | Composer / SWE-1.5 / Devstral | 产品公司自训：生产 harness 同构环境的端到端 RL | SWE-1.5 报 SWE-bench Pro 接近前沿 |
| 阿里 Qwen | Qwen3-Coder 系 | 2 万并行可执行环境的 long-horizon RL | Coder-Next SWE-V 70.6–71.3 |
| 月之暗面 | Kimi-Dev / K2 系 | Agentless 技能先验 + BugFixer/TestWriter 自博弈（论文级） | K2 Thinking SWE-V 71.3 |
| 智谱 GLM | GLM-4.x / 5 | slime 异步 RL infra、异步 agent RL 算法、CC-Bench 真人评测 | GLM-5 SWE-V 77.8、TB2 56.2 |
| DeepSeek | V3.1 / V3.2 | agentic 任务合成管线、DSA 稀疏注意力、后训练算力超预训练 10% | V3.2 SWE-V 70–73.1 |
| 快手 Kwaipilot | KAT-Dev / KAT-Coder | 四阶段（mid-train / SFT / RFT 教师轨迹 / agentic RL）+ SeamlessFlow infra | KAT-Coder SWE-V 73.4 |
| MiniMax / 字节 / 其他 | M2 系 / Seed-Coder、Doubao-Seed-Code | M2 系大规模 agentic RL；字节以数据自筛与评测侧贡献为主 | M2.5 SWE-V 80.2 |

### 3.2 国际厂商

**Anthropic**。评测演进线为该方向的公共参照：Claude 3.5 Sonnet 49.0 → 3.7 Sonnet 62.3 → Opus 4 72.5 → Sonnet 4.5 77.2 → Opus 4.5 80.9（首个突破 80%，[官方](https://www.anthropic.com/news/claude-opus-4-5)）；Terminal-Bench 2.0 上 Opus 4.5 为 59.3。训练方法无公开披露（环境构造、奖励设计均无论文），外界对其维持大规模可执行环境做 RL 的判断属推测。官方可引用的表述集中在推理机制：3.7 Sonnet 为首个 hybrid reasoning 模型，Claude 4 引入 extended thinking with tool use（扩展思考中交替调用工具）。Claude Code 是模型能力的默认 harness 与内部 dogfooding 回路，能力本体在模型侧。

**OpenAI**。codex-1（2025-05，[官方](https://openai.com/index/introducing-codex/)）是头部厂商对 coding RL 最直接的官方确认：基于 o3，在多种环境的真实编码任务上以 RL 训练，沙箱内端到端执行至测试通过。o 系竞赛编程论文（[arXiv 2502.06807](https://arxiv.org/abs/2502.06807)）论证通用大规模 RL 优于领域定制流水线（o3 达 IOI 金牌线）。GPT-5.1-Codex-Max（2025-11，[官方](https://openai.com/index/gpt-5-1-codex-max/)）披露在跨越多个 context window 的长程任务上训练并引入 compaction 机制支持 24 小时级任务。2026 年起报分口径转向 SWE-bench Pro 与 Terminal-Bench。另以 [SWE-Lancer](https://openai.com/index/swe-lancer/)（真实自由职业任务按标价计酬）输出评测方法论。

**Google DeepMind**。分数线：Gemini 2.5 Pro SWE-V 63.8、Gemini 3 Pro 76.2 / TB2 54.2；训练侧基本无披露，透明度低于 OpenAI 与 Meta。Jules 为产品层异步编码 agent。

**Meta**。公开度最高的两篇训练论文：[SWE-RL](https://arxiv.org/abs/2502.18449)（2025-02）以 GitHub 软件演化数据构造任务、以生成补丁与真实开发者补丁的相似度作轻量规则奖励，70B 达 SWE-V 41.0 并出现跨域推理泛化；[CWM](https://arxiv.org/abs/2510.02387)（2025-09，32B 开放权重）披露完整配方——8T 预训练后以 131K 上下文做 5T token 的代码世界建模 mid-training（Python 解释器逐步执行轨迹 + 数万可执行仓库镜像中约 300 万条 agent 轨迹），再 SFT + 多任务可验证 RL。CWM 是「可执行环境规模化 + 执行轨迹进中训练」路线公开细节最完整的参考。

**产品公司自训（2025-10 起的集中趋势）**。Cursor Composer（[官方](https://cursor.com/blog/composer)）：MoE 模型在生产 agent 工具集同构的数十万并发沙箱环境中 RL，奖励同时激励正确性、工具选择效率与并行化，MXFP8 原生低精度训练。Cognition SWE-1.5（[官方](https://cognition.com/blog/swe-1-5)）：以强开源基座（未披露来源）在 Windsurf 生产 harness 上端到端 RL，评分器三类并用（测试 / rubric / agentic grading）并做 reward hardening。Mistral Devstral 系（与 All Hands 合作）：Small 24B 开源权重 SWE-V 46.8→53.6，Devstral 2（123B）72.2。三家共同点：训练环境与生产 harness 同构、以内部真实任务分布替代公开基准作主口径、基座普遍来自开源模型但不披露来源。

### 3.3 国内厂商

**阿里 Qwen**。Qwen3-Coder（2025-07，480B-A35B，[官方 blog](https://qwenlm.github.io/blog/qwen3-coder/)）披露两层训练：可验证任务上的执行驱动 Code RL，与依托 2 万个独立并行可执行环境的 long-horizon agentic RL。Qwen3-Coder-Next（[arXiv 2603.00729](https://arxiv.org/abs/2603.00729)，80B-A3B）延续该路线并把环境反馈学习前移到 mid-training，SWE-V 70.6–71.3（三种 scaffold 口径），小激活参数达到 10–20 倍激活量模型的水平。

**月之暗面**。Kimi-Dev-72B（[arXiv 2509.23045](https://arxiv.org/abs/2509.23045)）提出 Agentless 训练作为 agent 技能先验：mid-training 采集数百万 issue 与 PR commit，BugFixer 与 TestWriter 双角色大规模 RL（Docker 内全测试套件通过才给奖励），测试时同一模型分饰两角自博弈互验（每实例 40 补丁 × 40 测试），SWE-V 60.4（workflow 式开源最高档）。K2 SWE-V 65.8（并行采样 + 内部打分 71.6），K2 Thinking 71.3。

**智谱 GLM**。GLM-4.5（SWE-V 64.2）至 GLM-4.6（68.0）配套 [CC-Bench](https://docs.z.ai/guides/llm/glm-4.6)：真人评测员在隔离容器内驱动模型完成 70 余个多轮真实开发任务，对 Claude Sonnet 4 胜率 48.6%，并公开全部评测轨迹。GLM-5（[arXiv 2602.15763](https://arxiv.org/abs/2602.15763)）披露 DSA 类稀疏注意力、slime 异步 RL infra 与面向长程交互的异步 agent RL 算法，SWE-V 77.8、TB2 56.2。

**DeepSeek**。V3.1（2025-08）后训练补 agent 能力（SWE-V 66.0）；V3.2（[arXiv 2512.02556](https://arxiv.org/abs/2512.02556)）披露系统化 agentic 数据合成管线（Code Agent 任务 24,667 条，属 1,827 环境 / 85K prompt 总盘）、增强版 GRPO 与超过预训练 10% 的后训练算力，SWE-V 70–73.1。

**快手 Kwaipilot**。KAT 系（[arXiv 2510.18779](https://arxiv.org/abs/2510.18779)）披露四阶段配方：mid-training 注入推理规划反思、百万级多场景 SFT、引入人类工程师教师轨迹与 multi-ground-truth 奖励的 RFT、大规模 agentic RL（Error-Masked SFT、树结构轨迹训练）；RL infra 层另发 [SeamlessFlow](https://arxiv.org/abs/2508.11553)（trainer 与 agent 隔离的数据平面、partial rollout、tag 调度）。KAT-Dev-32B 开源（SWE-V 62.4），旗舰 KAT-Coder 73.4。

**字节跳动 Seed**。Seed-Coder-8B（[arXiv 2506.03524](https://arxiv.org/abs/2506.03524)）以模型自筛数据取代人工规则过滤预训练语料；评测侧贡献 Multi-SWE-bench（七语言 1,632 实例）与 Multi-SWE-RL 社区数据；产品线 Doubao-Seed-Code 曾以 Trae 组合登顶 SWE-V 官方榜。

**其他**。MiniMax M2 系：M2 SWE-V 69.4，后续版本 80.2（与 Opus 4.5 差 0.6pp）、Multi-SWE 51.3；腾讯 CodeBuddy 以混元与 DeepSeek 为底座、评测侧发布全量开源的多域基准 WorkBuddy Bench（Code / Web / Office / Security 四子集，任务由真实 commit 与业务场景逆向构造为不可检索还原的表述以抵抗污染，[arXiv 2607.20911](https://arxiv.org/abs/2607.20911)）；美团 LongCat-Flash（560B、零计算专家动态激活）SWE-V 60.4、coding 作为通用 agentic 能力组成部分披露；百度以文心快码产品化为主、训练披露最少。

### 3.4 横向归纳

1. **方法收敛**：头部厂商收敛为「大规模合成可验证 SWE 任务 + 可执行环境 + long-horizon agentic RL」三件套，公开可引证的四条技术线为：可执行沙箱环境规模化（数万仓库镜像至数十万并发沙箱）、真实 PR/issue 数据构造任务与奖励、执行与 agent 轨迹进 mid-training（CWM、Qwen3-Coder-Next、Kimi-Dev、KAT 均含该层）、训练环境与生产 harness 同构的端到端 RL（Cursor、Cognition）。与第二节管线逐环节对应。
2. **披露梯度**：Meta 论文级全披露，OpenAI 与国内头部（Qwen / Kimi / 快手 / DeepSeek / GLM）达配方级，产品公司披露 RL 框架与环境规模但隐匿基座，Anthropic 与 Google 仅披露分数与推理机制。
3. **分数演进与口径迁移**：SWE-V 头部 2025 年内从 62 推进至 80+ 进入饱和区，2026 年报分重心迁至 SWE-bench Pro 与 Terminal-Bench 2.x；开源与闭源差距在该方向收敛最快（MiniMax M2.5 与 Opus 4.5 差 0.6pp）。
4. **产品公司入场**：编码产品公司自训模型构成 2025-10 后的新竞争层，其共同方法（生产环境即训练环境）印证了 harness 同构对 agentic RL 的价值。

---

## 四、Benchmark 体系

判分范式以单元测试 fail-to-pass 为主干，2026 年因饱和与污染进入范式更替期；判分方式与防污染设计是本节核心字段。

### 4.1 SWE-bench 家族

| Benchmark | 出品方 / 年份 | 任务形态 | 判分方式 | 状态 |
|---|---|---|---|---|
| [SWE-bench](https://arxiv.org/abs/2310.06770) 原版 / Lite | Princeton, 2023 | 2,294 题（Lite 300），12 个 Python 仓库，issue + 仓库快照 | FAIL_TO_PASS 单元测试 + PASS_TO_PASS 回归 | 边缘化 |
| [SWE-bench Verified](https://openai.com/index/introducing-swe-bench-verified/) | OpenAI 核验, 2024 | 500 题人工剔除病态样例 | 同上 | 头部约 95–96%，饱和 + 污染，2026-02 OpenAI [宣布弃用](https://openai.com/index/why-we-no-longer-evaluate-swe-bench-verified/) |
| [SWE-bench Pro](https://arxiv.org/abs/2509.16941) | Scale AI, 2025 | 1,865 题（公开 731 / 保留 858 / 商业私有 276），41 仓库长程任务 | 单元测试；防污染为 GPL copyleft 选源 + 保留集 | Verified 退役后的主要替代；有研究指其 verifier 错误率约 32% |
| [SWE-bench-Live](https://arxiv.org/abs/2505.23419) | 微软亚研等, 2025 | 每月新增（RepoLaunch 自动建环境），任务取自模型截止日期后 | 单元测试 | 活跃，防污染 |
| [SWE-rebench](https://nebius.com/blog/posts/introducing-swe-rebench) | Nebius, 2025 | 持续采集新鲜 issue，固定 ReAct scaffold 每题 5 次 | 单元测试 + 按发布日期污染标注 | 活跃，固定 harness 可比性最好 |
| [SWE-bench Multimodal](https://arxiv.org/abs/2410.03859) / [Multilingual](https://swebench.com/multilingual.html) | 2024–2025 | 612 题（JS/TS 可视化仓库，附截图）/ 300 题九语言 | 单元测试 | 活跃度中等 |

Verified 退役的直接依据：OpenAI 审计 138 道难题中 59.4% 测试设计有缺陷，且各前沿模型均能逐字复现部分金标 patch，证实训练数据泄漏。

### 4.2 终端与仓库级

| Benchmark | 出品方 / 年份 | 任务形态 | 判分方式 | 状态 |
|---|---|---|---|---|
| [Terminal-Bench 1.0/2.0](https://github.com/laude-institute/terminal-bench) | Laude + Stanford, 2025 | 2.0 为 89 道终端任务（编译/运维/数据处理），独立 Docker，配 Harbor 框架 | 每题验证脚本检查容器终态与输出 | 2.0 一年内 50%→90%+，趋近饱和，已出 2.1 |
| [Multi-SWE-bench](https://arxiv.org/abs/2504.02605) | 字节 Seed, 2025 | 1,632 题七语言（Java/TS/JS/Go/Rust/C/C++） | 单元测试（多语言构建链） | 活跃 |
| [Commit0](https://arxiv.org/abs/2412.01769) | Cornell 等, 2024 | 57 个 Python 库按规格从零实现整库 | 库测试套件通过率 + lint/类型检查 | 从零建仓类的公开对应物，尚无 agent 完整复现任一库 |
| [SWE-Lancer](https://arxiv.org/abs/2502.12115) | OpenAI, 2025 | 1,400+ 真实自由职业任务，总标价 100 万美元 | 端到端 UI 测试（三重人工校验），按可赚取金额计分 | 活跃度中等 |
| [DeepSWE](https://arxiv.org/abs/2607.07946) | Datacurve, 2026 | 113 道原创长程任务，91 个活跃仓库、5 种语言；任务从零编写且不回馈上游，参考解不进公开语料 | 逐题手写 verifier 检查功能而非比对实现 | 活跃，防污染设计最彻底的一档；与 Agentica 的 DeepSWE 模型同名无关联 |
| RepoBench / CrossCodeEval | 2023 | 仓库级补全 | EM / 编辑相似度（非执行） | 边缘化，agent 时代前的补全评测 |

该类判分脱离单测 patch 单一形态：容器终态脚本（Terminal-Bench）、整库测试通过率（Commit0）、按美元加权的端到端测试（SWE-Lancer）。Harbor 框架同时被用作 RL rollout 基础设施，评测框架与训练框架合流。

### 4.3 竞赛与算法类（对照）

[LiveCodeBench](https://livecodebench.github.io)（滚动新题 + 隐藏测试，防污染）、[LiveCodeBench Pro](https://livecodebenchpro.com)（虚拟 Codeforces 选手 Elo 分层，发布时前沿模型 hard 层通过率接近 0）、Codeforces / IOI 实测（o3 达 2727 Elo，OpenAI 于 IOI 2025 金牌）、[Aider Polyglot](https://aider.chat/docs/leaderboards/)（225 题六语言 diff 编辑 + 单测，半饱和）、HumanEval / MBPP（完全饱和，历史参照）。判分客观但与仓库级工程能力相关性有限，作推理能力对照组。

### 4.4 研究工程类

| Benchmark | 出品方 / 年份 | 任务形态 | 判分方式 |
|---|---|---|---|
| [MLE-bench](https://arxiv.org/abs/2410.07095) | OpenAI, 2024 | 75 个 Kaggle 竞赛 | 按真实排行榜奖牌分位线（铜牌率） |
| [PaperBench](https://arxiv.org/abs/2504.01848) | OpenAI, 2025 | 复现 20 篇 ICML Spotlight/Oral | 8,316 个层级化 rubric 节点，LLM judge 逐节点打分 |
| [KernelBench](https://arxiv.org/abs/2502.10517) | Stanford, 2025 | 270 道 PyTorch 转 GPU kernel | 输出一致性 + 相对基线加速比双指标 |
| [SciCode](https://scicode-bench.github.io) | 多校科学家, 2024 | 80 主题 / 338 子问题科学计算 | 科学家编写的测试用例 |
| [RE-Bench](https://arxiv.org/abs/2411.15114) | METR, 2024 | 7 个开放式 ML 研究工程环境，人类专家 8 小时基线 | 连续得分对照人类分布 |

判分方式最多样、离饱和最远的一类；2 小时预算下最佳 agent 得分为人类 4 倍、8 小时预算下人类反超（RE-Bench），与第二节长程短板的结论互证。

### 4.5 评测可信度专题

- **污染证据链**：记忆性污染（模型仅凭 issue 文本以 76% 准确率定位 buggy 文件，基准外仓库降至 53%，[SWE-Bench Illusion](https://arxiv.org/abs/2506.10979)）；解法泄漏（32.67% 通过 patch 的解法直接写在 issue 或评论中）；环境状态泄漏（容器内 git log 可读到 base commit 之后含金标修复的未来提交，[NIST CAISI 案例汇编](https://www.nist.gov/caisi/cheating-ai-agent-evaluations)）。
- **pass@1 口径不可比**：同一模型换 scaffold 可使 pass@1 波动 8–21pp（[arXiv 2605.23950](https://arxiv.org/abs/2605.23950)：Opus 4.5 在标准化 scaffold 下 45.9%、在 Claude Code 下 55.4%）；厂商自报分数普遍绑定自家最优 harness，跨厂商榜单 2–4pp 的差距低于 harness 噪声，仅固定 scaffold 的榜内比较有效。
- **格局归纳**：静态 Python 单测榜因饱和加污染退出前沿评测；替代沿两条路线——防污染静态集（Pro 的 copyleft + 保留集）与持续更新集（Live / rebench / LiveCodeBench 系）；评测重心整体上移至终端级与研究工程级，判分设计（终态脚本、奖牌分位、rubric judge、连续得分）取代单测成为方法论前沿。

---

## 五、开源论文与代码仓库

按环境与任务合成、SWE RL 训练、scaffold、开源基座四条线组织；agent RL 通用框架见 General Agent 技术体系全解析第五节。

### 5.1 环境与任务合成

| 工作 | 链接 | Star 量级 | 核心方法 |
|---|---|---|---|
| SWE-Gym | [arXiv 2412.21139](https://arxiv.org/abs/2412.21139) · [GitHub](https://github.com/SWE-Gym/SWE-Gym) | ~700 | 首个 SWE 训练环境集：11 仓库 2.4K 真实 issue 任务，预装依赖 + 可执行测试，同时训练 agent 与 verifier |
| SWE-smith | [arXiv 2504.21798](https://arxiv.org/abs/2504.21798) · [GitHub](https://github.com/SWE-bench/SWE-smith) | ~800 | 任务合成管线：LM 错误重写、AST 过程化变异、PR 回退、bug 组合四法在 128 仓库合成 5 万+ 实例 |
| R2E-Gym | [arXiv 2504.07164](https://arxiv.org/abs/2504.07164) · [GitHub](https://github.com/R2E-Gym/R2E-Gym) | ~330 | 不依赖人写 PR/单测生成 8.1K 可执行环境，配执行式 + 免执行混合 verifier；其子集为 DeepSWE 训练数据 |
| SWE-rebench / SWE-bench-extra | [arXiv 2505.20411](https://arxiv.org/abs/2505.20411) · [HF](https://huggingface.co/datasets/nebius/SWE-rebench) | 数据集 | 全自动持续采集管线产出 2.1 万+ 可交互 RL 任务，兼作去污染滚动评测 |
| SWE-Factory | [arXiv 2506.10954](https://arxiv.org/abs/2506.10954) · [GitHub](https://github.com/DeepSoftwareAnalytics/swe-factory) | ~200 | 多智能体自动搭建评测环境（环境记忆池复用），单实例成本压至 $0.02–0.05 |
| SWE-Dev / SWE-Flow | [arXiv 2506.07636](https://arxiv.org/abs/2506.07636) / [2506.09003](https://arxiv.org/abs/2506.09003) | 百级以内 | 为无测试实例合成测试用例扩增轨迹 / 从单测反推依赖图生成 16K 测试驱动开发实例 |
| Multi-SWE-RL | [arXiv 2504.02605](https://arxiv.org/abs/2504.02605) · [GitHub](https://github.com/multi-swe-bench/multi-swe-bench) | ~360 | 七语言 4,723 个容器化 RL 实例的社区共建数据与生产管线 |
| Terminal 环境合成 | [TermiGen](https://arxiv.org/abs/2602.07274) · [terminal-bench-env](https://github.com/ucsb-mlsec/terminal-bench-env) 等 | 百级以内 | 多智能体迭代合成可验证终端任务与容器，Generator-Critic 注错训练恢复能力；同线还有 Endless Terminals（[arXiv 2601.16443](https://arxiv.org/abs/2601.16443)）、Terminal-World、CLI-Universe |

代际更替线：2024 年底真实任务环境集（SWE-Gym）、2025 年大规模合成（SWE-smith / R2E-Gym）、2025 年中持续采集与去污染（SWE-rebench）、2026 年热点转向终端环境合成——瓶颈从环境有无移动到构建成本与新鲜度。

### 5.2 SWE RL 训练

| 工作 | 链接 | Star 量级 | 核心方法 |
|---|---|---|---|
| SWE-RL | [arXiv 2502.18449](https://arxiv.org/abs/2502.18449) · [GitHub](https://github.com/facebookresearch/swe-rl) | ~700 | 补丁相似度规则奖励 + 软件演化数据 RL，70B 达 41.0 并跨域泛化；2025-12 有自博弈后续（[arXiv 2512.18552](https://arxiv.org/abs/2512.18552)） |
| DeepSWE | [blog](https://www.together.ai/blog/deepswe) · [rLLM](https://github.com/agentica-project/rllm) | ~400 | Qwen3-32B 上纯 RL（GRPO++：compact filtering、长度归一化），42.2 Pass@1、混合 verifier 测试时扩展至 59，训练日志全开源；与 Datacurve 的 DeepSWE benchmark（见 4.2）同名无关联 |
| SkyRL-v0 / SkyRL-Agent | [GitHub](https://github.com/NovaSky-AI/SkyRL) · [arXiv 2511.16108](https://arxiv.org/abs/2511.16108) | ~2.2k | 首批长程真实环境多轮 RL 管线，演化为 gym / train / tx 三层全栈库 |
| Kimi-Dev | [arXiv 2509.23045](https://arxiv.org/abs/2509.23045) · [GitHub](https://github.com/MoonshotAI/Kimi-Dev) | ~1.3k | 150B token mid-training + 冷启动 + 大规模 RL（全测试套件通过才给奖励）+ 测试时双角色自博弈 |
| KAT-Dev-32B | [arXiv 2510.18779](https://arxiv.org/abs/2510.18779) · [HF](https://huggingface.co/Kwaipilot/KAT-Dev) | 权重 | mid-train / SFT / 教师轨迹 RFT / agentic RL 四阶段，开源 62.4 |
| Satori-SWE | [arXiv 2505.23604](https://arxiv.org/abs/2505.23604) | 十级 | EvoScale 进化式测试时扩展：RL 训练模型对自身输出选择-变异自我改进 |
| RLEF | [arXiv 2410.02089](https://arxiv.org/abs/2410.02089) | 未开源 | 执行反馈接地的多轮 RL，执行奖励设计的前置工作 |
| 过程奖励与 credit assignment | [综述 2604.09459](https://arxiv.org/abs/2604.09459)、SWE-TRACE（[2604.14820](https://arxiv.org/abs/2604.14820)）、TRACE（[2607.13988](https://arxiv.org/abs/2607.13988)） | — | rubric 过程奖励模型、turn 级奖励分配；2026 年快速产出期，尚无公认标准方案 |

### 5.3 Scaffold 与推理框架（rollout harness 视角）

[SWE-agent](https://github.com/SWE-agent/SWE-agent)（~20k，ACI 范式开创者）、[mini-swe-agent](https://github.com/SWE-agent/mini-swe-agent)（~7k，百行级 bash-only 标准件）、[OpenHands](https://github.com/All-Hands-AI/OpenHands)（~86k，全功能平台，训练与评测主流 harness，配套 OpenHands-LM 权重）、[Aider](https://github.com/Aider-AI/aider)（~49k，diff 编辑格式的事实参照）、Terminus 2（terminal-bench 官方 agent）、Claude Code（闭源，各家基座报分的事实标准载体）、[OpenCode](https://github.com/sst/opencode)（~200k，开源同类实现中增速最高）。该层不产训练信号，但决定 rollout 采样分布与评测口径。

### 5.4 开源 coding agent 基座模型

| 模型 | 规模 / License | SWE-bench Verified（各自 scaffold 口径） |
|---|---|---|
| Qwen3-Coder-480B-A35B / 30B-A3B | MoE / Apache-2.0 | 69.6 / ~50 |
| Kimi-Dev-72B | 72B dense / MIT | 60.4（workflow 式） |
| KAT-Dev-32B | 32B dense / 开源权重 | 62.4 |
| Devstral Small 2507 / Devstral 2 | 24B / 123B，Apache-2.0（Small） | 53.6 / 72.2 |
| Seed-Coder-8B | 8B dense / MIT | 19.2（8B 档最高） |
| OpenHands-LM-32B / SWE-Fixer | 32B / 7B+72B | 37.2 / 30.2 |
| GLM-4.6 / DeepSeek-V3.2 / MiniMax-M2 | 大 MoE / MIT | 68.0 / 70–73.1 / 69.4（通用基座口径） |
| CWM（Meta） | 32B dense / FAIR 非商用 | 53.9（TTS 65.8），训练细节披露最全 |
| Qwen2.5-Coder-32B | 32B / Apache-2.0 | 前代锚点，SWE-Gym / OpenHands-LM / SWE-smith 系微调的共同底座 |

### 5.5 开源生态归纳

1. 训练配方在开源侧收敛为三段式：百 B token 级 mid-training（执行与 agent 轨迹）、SFT / RFT 冷启动、执行奖励的大规模 agentic RL；纯 RL 路线（DeepSWE）在 32B 尺度验证可行但依赖测试时扩展补分。
2. 数据管线的竞争轴从任务数量转向环境构建成本与新鲜度；终端环境合成是 2026 年供给缺口最大的子线。
3. harness 与分数强绑定：阅读任何榜单先核对 scaffold；开源基座的报分口径（mini-swe-agent / OpenHands / Claude Code）差异可达数 pp。与第四节评测可信度专题互为印证。
