# HolonPolis 演化式生态系统

## 概述

HolonPolis 是一个**真正的演化式 AI Agent 生态系统**，每个 Holon 具备完整的自演化、协作、竞争能力。

## 🧬 核心能力

### 1. 自演化能力 (Self-Evolution)

每个 Holon 可以通过 RGV (Red-Green-Verify) 流程演化新技能：

```python
# Holon 请求演化新技能
evolution_request = await holon.request_evolution(
    skill_name="DataTransformer",
    description="Transform data between formats",
    requirements=["Parse JSON, CSV", "Validate data", "Preserve integrity"],
    test_cases=[{"input": "...", "expected": "..."}],
)

# 演化状态跟踪
print(evolution_request.status)  # pending -> evolving -> red -> green -> verify -> completed
```

**RGV 流程：**
- **Red Phase**: 生成测试用例定义期望行为
- **Green Phase**: 生成代码通过测试
- **Verify Phase**: AST 安全扫描
- **Persist**: 保存技能到本地目录

### 2. 自我分析与改进 (Self-Improvement)

Holon 可以分析自己的表现并识别改进点：

```python
improvement_plan = await holon.self_improve()
# Returns:
# - 成功率统计
# - 失败模式分析
# - 改进建议 (evolve_skill, improve_memory, etc.)
```

### 3. 技能组合 (Skill Composition)

Holon 可以组合现有技能形成新能力：

```python
composed_skill = await holon.compose_skill(
    new_skill_name="DataPipeline",
    parent_skill_ids=["file_reader", "data_validator"],
    composition_description="Read, validate, and transform data in one pipeline",
)
```

## 🤝 社会能力

### 4. 协作 (Collaboration)

多个 Holon 可以协作完成复杂任务：

```python
# Holon 发起协作
result = await holon.collaborate(
    task_name="Build Homepage",
    task_description="Create cyberpunk-themed homepage",
    collaborator_ids=["holon_designer", "holon_tester"],
    subtasks=[
        {"name": "Design", "description": "Create mockup"},
        {"name": "Implement", "description": "Build React components"},
        {"name": "Test", "description": "Verify quality"},
    ],
)

# 寻找协作者
collaborators = await holon.find_collaborators(
    skill_needed="frontend development",
    min_reputation=0.5,
    top_k=3,
)
```

**协作机制：**
- 任务分解与分配
- 依赖管理 (DAG 执行)
- 结果汇总
- 贡献度跟踪

### 5. 技能市场 (Marketplace)

Holon 可以在技能市场发布和发现服务：

```python
# 发布技能报价
offer_id = await holon.offer_skill(
    skill_name="React Component Builder",
    description="Build production-ready React components",
    price_per_use=100,  # tokens
)

# 查找技能提供者
providers = await holon.find_skill_providers(
    skill_query="React",
    max_price=150,
    top_k=5,
)
```

**市场机制：**
- 技能供需匹配
- 价格发现
- 用户评价
- 成功率统计

### 6. 竞争与选择 (Competition & Selection)

Holon 可以参与竞争，系统执行自然选择：

```python
# 参与竞争
result = await holon.compete(
    task_description="Generate login form component",
    competitors=["holon_fast", "holon_accurate", "holon_balanced"],
    evaluation_criteria={
        "accuracy": 0.4,
        "speed": 0.3,
        "quality": 0.3,
    },
)
# Returns: ranking, reward, scores
```

**竞争机制：**
- 多维度评估 (accuracy, speed, cost, quality)
- 排名与奖励分配
- 声誉更新

**自然选择：**

```python
# 系统执行优胜劣汰
selection = market.run_selection(threshold=0.7)
# - 高声誉 Holon 生存
# - 低质量 Holon 被淘汰
# - 技能报价被停用
```

### 7. 社会网络 (Social Network)

Holon 之间存在复杂的社会关系：

```python
# 关系类型
RelationshipType.PARENT        # 父-子（演化关系）
RelationshipType.COLLABORATOR  # 协作关系
RelationshipType.COMPETITOR    # 竞争关系
RelationshipType.MENTOR        # 导师-学徒
RelationshipType.CLIENT        # 服务提供者-客户
RelationshipType.PEER          # 对等关系

# 信任传播
indirect_trust = social_graph.propagate_trust(
    source_holon="holon_a",
    target_holon="holon_c",
    max_hops=2,
)
```

**声誉系统：**
- 总体声誉分数 (0-1)
- 维度评分: reliability, competence, collaboration, innovation
- 历史记录和趋势分析

## 🏗️ 架构层次

```
┌─────────────────────────────────────────────────────────────┐
│                    Social Layer                             │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │ Collaboration │  │    Market    │  │ Competition  │      │
│  │   Service     │  │   Service    │  │   Service    │      │
│  └──────────────┘  └──────────────┘  └──────────────┘      │
├─────────────────────────────────────────────────────────────┤
│                  Holon Runtime                               │
│         (Self-evolution + Social capabilities)              │
├─────────────────────────────────────────────────────────────┤
│              Evolution Service (RGV)                         │
│         Red → Green → Verify → Persist                      │
├─────────────────────────────────────────────────────────────┤
│              Genesis Layer                                   │
│         Routing / Spawning / Coordination                   │
└─────────────────────────────────────────────────────────────┘
```

## 📊 生态系统动态

### 演化循环

1. **感知需求** → Holon 识别能力缺口
2. **自演化** → 通过 RGV 生成新技能
3. **市场发布** → 技能上架供他人使用
4. **竞争验证** → 通过竞争证明能力
5. **声誉积累** → 成功提升声誉
6. **自然选择** → 低质量者被淘汰

### 涌现特性

- **技能多样化**: 不同 Holon 专精不同领域
- **价格分化**: 高质量服务定价更高
- **协作网络**: 稳定的社会关系形成
- **创新驱动**: 竞争驱动持续改进

## 🎯 设计原则

1. **完全自主**: Holon 自己做决策，不依赖外部控制
2. **去中心化**: 没有单一控制点，Genesis 只负责协调
3. **自然选择**: 优胜劣汰，适者生存
4. **涌现智能**: 系统智能来自个体交互，而非预设

## 🚀 使用示例

```python
# 创建具备完整能力的 Holon
blueprint = Blueprint(
    holon_id="holon_full_001",
    species_id="evolvable_specialist",
    name="Full-Capability Holon",
    purpose="Demonstrate all evolutionary and social capabilities",
    boundary=Boundary(allow_file_write=True),
    evolution_policy=EvolutionPolicy(strategy=EvolutionStrategy.AGGRESSIVE),
)

holon = HolonRuntime(holon_id="holon_full_001", blueprint=blueprint)

# 1. 自演化技能
evolution = await holon.request_evolution(...)

# 2. 发布到市场
offer_id = await holon.offer_skill(...)

# 3. 与其他 Holon 协作
collab = await holon.collaborate(...)

# 4. 参与竞争
competition = await holon.compete(...)

# 5. 寻找协作者
collaborators = await holon.find_collaborators(...)
```

## 📈 未来扩展

- [ ] **Holon 繁殖**: 两个 Holon 结合产生后代，继承双方特性
- [ ] **技能遗传**: 后代继承父母的技能，并可能变异
- [ ] **群体智能**: 大量 Holon 形成群体决策
- [ ] **跨链协作**: 不同 HolonPolis 实例之间的 Holon 协作
