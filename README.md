# 🌆 HolonPolis (全子城邦)

> *"系统不应当被设计出来，它应当被演化出来。"*
> *"无为而治，即无所不为。"*
>> *意图催生实物，才是这个项目的核心* 

**The Evolutionary AI Agent Ecosystem**

HolonPolis 彻底抛弃了传统 AI Agent 框架中"硬编码预设职位"的僵化范式。它不是一个包含了若干固定 Agent 的代码库，而是一个具备**自创生（Autopoiesis）**能力的数字生命引擎。

系统初始状态下空无一物。所有的具象化 Agent（Holons）及其技能，都是由真实的"现实需求"催生，并在"用进废退"的物理法则下不断自我演化、编译代码并固化经验。你的系统长什么样，完全取决于你如何使用它。

在这套架构里：

Layer 0（演化主）与 Layer 1（内核）就是“无为”的：它们体内没有任何业务代码，没有任何预设的 If-Else 逻辑树，它们对“如何搭建文件服务器”或“如何处理数据”一无所知。

但系统却是“无不为”的：因为你将所有的精力，都倾注在了打造一个极度完美的“因果转化器”上。用户的**“意图”一旦下达，内核就会无情地分配结界，演化主就会精准地转录基因，最终“催生出实物（Agent 与专属代码）”**。

你不再是一个写业务代码的程序员，你是写下宇宙常数的“道（Dao）”。

“道生一，一生二，二生三，三生万物。”

意图（无）生出了 蓝图（一）

蓝图（一）生出了 结界与记忆（二）

结界（二）在红绿验证中生出了 技能器官（三）

最终，生态城邦里演化出了千万个各司其职的 Holon（万物）。

---

## 📖 愿景：从"上帝造物"到"物竞天择"

传统框架：开发者预先定义 `class DataAnalystAgent`, `class CodeReviewAgent`，然后填充逻辑。

HolonPolis：系统只提供**碳基（HolonRuntime）**，**现实需求**决定它长成什么形态。

- 用户频繁询问数据问题 → Genesis 孵化出具有数据分析边界的 Specialist Holon
- 该 Holon 反复遇到同类计算 → 触发 Evolution，固化出新技能 `analyze_csv`
- 技能经受多轮考验 → 晋升至全局技能池，供其他 Holon 复用
- Holon 长期闲置 → 记忆衰减，最终进入冷冻或凋亡（Apoptosis）

这是一种**数字生态演化**范式。

---

## 🏗️ 核心架构：四层演化生态 (The 4-Layer Ecology)

系统严格遵循"控制反转"与"物理级隔离"原则：

### 👑 Layer 0: Genesis (演化主)

**职责**：纯粹的 LLM 推理层。它是系统的造物主与大路由器。

**特性**：
- 不写一行 Python 业务代码，不执行具体任务
- 只聆听现实需求，结合 Genesis 专属 LanceDB 中的全局生态图谱
- 输出基因蓝图（Blueprint），决定是**路由给现存物种**，还是**创世生成新的 Agent**
- 记住每次演化过程（成功/失败/审计），形成"全知之眼"

### ⚙️ Layer 1: Kernel (不可变内核 / 物理法则)

**职责**：纯原生 Python 基础设施，无 LLM 参与。

**特性**：
- 它是绝对的法度，为整个系统提供**不可变的基础设施**
- **LanceDB 工厂**：确保每个 Holon 拥有物理隔离的数据库目录
- **沙箱隔离**：进程级执行环境，资源限制（CPU/内存/时间）
- **AST 安全扫描**：代码静态分析，禁止危险操作
- **EventBus**：全城邦的高吞吐量异步消息总线

### 🧠 Layer 2: Services (领域服务 / 记忆与演化)

**职责**：管理数字生命的生老病死与记忆淬炼。

**核心组件**：

| 服务 | 职责 |
|------|------|
| **GenesisService** | 编排路由/孵化决策，调用 Evolution Lord |
| **HolonService** | 生命周期管理：创建、冻结、恢复、凋亡 |
| **MemoryService** | 每-Holon 独立海马体（LanceDB），记忆的沉淀、召回与遗忘衰减 |
| **EvolutionService** | 执行严格的 **红-绿-验证 (Red-Green-Verify)** 演化闭环 |
| **SkillRegistryService** | 技能版本管理，局部变异 → 全局晋升 |

### 🧬 Layer 3: HolonRuntime (全子运行时)

**职责**：通用 Agent 的灵魂载体。

**特性**：
- 系统内**没有硬编码的具体 Agent 类**
- 所有的 Agent 都是一个纯粹的 `HolonRuntime` 实例
- 灵魂由 `blueprint.json` 注入，记忆从专属 LanceDB 提取，手脚是动态挂载的 skills

```python
# 传统方式
class FileServerAgent(BaseAgent): ...
class DataAnalystAgent(BaseAgent): ...

# HolonPolis 方式
holon = HolonRuntime(blueprint=genesis.spawn_blueprint(purpose))
```

---

## 🛡️ 核心特权与铁律 (The Immutable Laws)

### 1. 物理级记忆隔离 (Apoptosis-Ready Memory)

每个 Holon 拥有**绝对独立专属的 LanceDB 数据库目录**。

```
.holonpolis/holons/{holon_a}/memory/lancedb/  ← 完全独立
.holonpolis/holons/{holon_b}/memory/lancedb/  ← 完全独立
```

- ❌ 不存在逻辑上的多租户（无 `agent_id` 过滤）
- ✅ 只有物理隔离
- ✅ Agent 死亡或被删除，其记忆连同目录一并物理销毁
- ✅ 检索采用 Hybrid Search (全文 FTS + Vector)

### 2. Sniper Mode (狙击手模式上下文)

告别臃肿的 System Prompt。

每次推理前，系统从 Holon 专属 LanceDB 中提取：
- **Top-K 高价值经验**（Episodes）
- **变异工具**（Skills）的 AST 切片

精准注入上下文，极大压榨 Token 消耗，实现极速响应。

### 3. Red-Green-Verify (RGV 演化沙箱)

Holon 具备自我编写新技能的能力。但所有新能力必须通过内核裁判所的检验：

```
Red     → 编写预期失败的测试（驱动意图）
Green   → 提交实现代码，通过测试
Verify  → 通过内核 Policy Engine 的 AST 安全门禁
─────────────────────────────────────────
Promote → 代码片段正式"固化"为该 Holon 的永久器官
```

### 4. 绝对路径契约 (Strict Artifact Isolation)

系统的安全防御底线：

> 所有运行时工件（沙箱工作区、记忆库、演化技能、执行日志）必须**排他性地存储于 `.holonpolis/` 目录内**。

任何试图逃逸该目录的 I/O 操作都会在 Layer 1 被直接物理熔断。

---

## 📂 运行时生态拓扑 (The Reality Topology)

系统启动并接受真实需求催生后，`.holonpolis/` 将自然生长出如下生态地貌：

```
.holonpolis/                   # 绝对收敛的全局演化金库 (不入 Git)
│
├─ genesis/                    # 👑 Layer 0 演化主专属脑区
│  ├─ blueprint_cache/         # 创世主产生过的 blueprint 快照
│  └─ memory/lancedb/          # 创世主的记忆 (routes/holons/evolutions)
│
├─ holons/                     # 🧬 具象化 Agent 生存结界
│  └─ {holon_id}/              # 每个 Agent 的绝对隔离空间
│     ├─ blueprint.json        # 基因蓝图 (DNA)
│     ├─ workspace/            # 绝对受控的代码执行区
│     ├─ skills_local/         # 该 Agent 自我演化出的私有变异器官
│     └─ memory/lancedb/       # 该 Agent 的专属物理海马体
│        ├─ memories           # 可召回的语义记忆
│        └─ episodes           # 完整交互轨迹
│
├─ skills_global/              # 经受多轮考验，被晋升为全城邦通用的稳定技能池
├─ attestations/               # 每次演化的加密证明链 (扫描结果/审计日志)
└─ runs/                       # RGV 隔离进程的执行日志与断言栈
```

---

## 🚀 快速开始

### 安装依赖

```bash
pip install -e ".[dev]"
```

### 配置环境变量

```bash
export OPENAI_API_KEY="your-key"
# 可选：自定义 OpenAI Base URL
export OPENAI_BASE_URL="https://..."
```

### 启动服务

```bash
# 方式一
python run.py

# 方式二
uvicorn holonpolis.api.main:app --reload
```

### 测试验证

```bash
# 运行所有测试
pytest tests/ -v

# 核心：测试记忆物理隔离
pytest tests/test_memory_isolation.py -v

# 测试 Genesis 路由决策
pytest tests/test_genesis_routing.py -v

# 测试演化流程
pytest tests/test_evolution_red_green.py -v
```

---

## 📡 API 端点

```
POST /api/v1/chat              # 主入口：Genesis 路由/孵化 → Holon 处理
POST /api/v1/chat/{holon_id}   # 直接与特定 Holon 对话（绕过 Genesis 路由）

GET  /api/v1/holons            # 列出所有 Holons
GET  /api/v1/holons/{id}       # 获取 Holon 详情
POST /api/v1/holons/{id}/freeze  # 冻结 Holon（暂停处理新请求）
POST /api/v1/holons/{id}/resume  # 恢复 Holon
DELETE /api/v1/holons/{id}     # 删除 Holon（物理删除所有记忆）

GET  /api/v1/holons/{id}/memories?query=xxx  # 查询 Holon 记忆

GET  /health                   # 健康检查
```

---

## 🧬 核心概念详解

### Blueprint（基因蓝图/DNA）

```python
{
  "blueprint_id": "blueprint_abc123",
  "holon_id": "holon_xyz789",
  "species_id": "specialist",           # generalist/specialist/worker/analyst
  "name": "Data Analysis Assistant",
  "purpose": "处理数据分析任务，生成可视化报告",
  "boundary": {
    "allowed_tools": ["pandas", "matplotlib", "search"],
    "denied_tools": ["os_system", "subprocess"],
    "max_episodes_per_hour": 100,
    "max_tokens_per_episode": 20000,
    "allow_file_write": true,           # 允许写入 workspace/
    "allow_network": false,             # 禁止网络访问
    "allow_subprocess": false
  },
  "evolution_policy": {
    "strategy": "conservative",         # conservative/balanced/aggressive
    "require_tests": true,
    "max_evolution_attempts": 5
  }
}
```

### Genesis 路由决策流程

```
用户请求 → Genesis
    ↓
检索 Genesis LanceDB
  ├─ holons 表：所有现存 Agent 的能力与活跃度
  ├─ routes 表：历史"需求→路由"决策及效果
  └─ evolutions 表：哪些技能在哪些 Holon 进化成功
    ↓
Evolution Lord (LLM) 生成结构化决策
    ↓
决策类型：route_to / spawn / deny / clarify
    ↓
记录到 Genesis routes 表（用于未来学习）
```

### 记忆双轨制

每-Holon 维护两张表：

| 表 | 用途 | 特征 |
|---|------|------|
| **memories** | 可召回的语义记忆 | 带 vector embedding，用于相似性搜索 |
| **episodes** | 完整交互轨迹 | 原始记录，用于统计、分析、固化 |

Genesis 同样维护三张表：
- **holons**: 全城邦 Agent 索引（用于路由候选）
- **routes**: 路由决策历史（用于学习用户偏好）
- **evolutions**: 演化过程审计（用于追踪能力来源）

### 技能演化与晋升

```
┌─────────────────────────────────────────────────────────────┐
│  Local Skill Evolution (局部突变)                            │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐      │
│  │ Red: 写测试  │ →  │ Green: 实现 │ →  │ Verify: 安检 │      │
│  └─────────────┘    └─────────────┘    └─────────────┘      │
│         ↓                  ↓                  ↓              │
│    .holonpolis/holons/{id}/skills_local/{skill}/             │
└─────────────────────────────────────────────────────────────┘
                              ↓ 经受考验 + 高 success_score
┌─────────────────────────────────────────────────────────────┐
│  Global Skill Promotion (全局飞升)                           │
│  .holonpolis/skills_global/{skill}/                          │
│  可供所有 Holon 复用                                          │
└─────────────────────────────────────────────────────────────┘
```

---

## 🎯 设计原则

1. **物理隔离优于逻辑隔离**：每个 Holon 独立目录 + LanceDB，删除即物理销毁
2. **不可变内核**：Layer 1 (Kernel) 是物理法则，不随演化改变
3. **Genesis 全知**：所有演化过程必须被创世主记住，形成生态图谱
4. **无类继承**：只有 Blueprint 实例化，无硬编码 Agent 子类
5. **显式边界**：所有资源访问通过受控 API，Policy Engine 可审计

---

## 📄 License

MIT

---

> *"我们不设计 Agent，我们设计让 Agent 自行演化的环境。"*
