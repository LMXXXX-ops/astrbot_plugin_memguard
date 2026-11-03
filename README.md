# 记忆守卫 MemGuard

一个 AstrBot 插件，针对「记性时好时坏」的三个根因各下一刀：**召回不设门、语料不设防、效果不设尺**。

| 你截图里的短板 | 这里的做法 | 落在哪 |
|---|---|---|
| ① 命中率 13.6%，「提高命中率」被误命中到闲聊 | **召回门控**：先判断这句话该不该查记忆，元话语/调试/闲聊直接跳过；再用 BM25+意图+实体三重打分，阈值与间隔双门限 | `memguard/gate.py`、`retrieve.py` |
| ② chatlog 混着调试、测试、接口讨论，被当成「你的事」 | **污染隔离**：写入前算污染分，代码块/报错/技术词命中即隔离，只留日志不进召回池；`/mem purge` 可回溯清洗 | `memguard/extract.py` |
| ③ 没有评估集，改检索只能靠感觉 | **金标准 + 回归门禁**：自动生成正例、内置负例，`/mem eval` 出 P@1/FPR，`/mem tune` 改参数前先跑一遍，退化就拒绝 | `memguard/evaluate.py` |

## 一轮对话发生了什么

```
用户消息
   │
   ├─ 门控 decide()
   │    元话语？调试？闲聊？── 是 ──▶ 不召回（记一次 skip 决策）
   │    显式回忆 / 第一人称自述 ──▶ 继续
   │
   ├─ 检索 select()
   │    BM25(字符二元组) + 意图谓词加成 + 实体出现加成
   │    绝对阈值 + 第一二名间隔门限 + 条数上限
   │
   ├─ 注入 system_prompt（带 #编号 与日期，便于追溯）
   │
   └─ 采集 capture()
        污染分 ≥ 阈值 ──▶ 只写 turns 日志（可在 /mem audit 看到）
        否则 ──▶ 抽成结构化事实写库；功能性谓词（居住/工作/称呼）新值覆盖旧值
```

## 安装

**方式一：从 URL 安装（推荐）**

AstrBot WebUI → 插件管理 → 从 URL 安装，填：

```
https://github.com/LMXXXX-ops/astrbot_plugin_memguard
```

**方式二：手动放目录**

1. 把 `astrbot_plugin_memguard/` 整个目录放进 AstrBot 的 `data/plugins/`。
2. 在 WebUI 的「插件」页重载插件（或重启 AstrBot）。
3. 私聊机器人发 `/mem help` 确认加载。

依赖：无。只用标准库（sqlite3、re、json），因此在任何 Python 3.10+ 环境都能跑。

数据落在 `data/plugin_data/astrbot_plugin_memguard/`：

- `memguard.db` —— 记忆、轮次、召回日志、评估记录
- `golden.jsonl` —— 金标准测试集
- `settings_override.json` —— `/mem tune apply` 后的参数覆盖

## 指令

| 指令 | 作用 |
|---|---|
| `/mem status` | 记忆规模、门控分布、当前参数 |
| `/mem why` | 上一条消息为什么召回 / 为什么跳过（可解释性） |
| `/mem search <查询>` | 查看候选与分数，调阈值前先看这个 |
| `/mem add <内容>` | 手动写入一条记忆（最高优先级） |
| `/mem forget <编号>` | 删除指定记忆 |
| `/mem audit` | 查看被隔离的高污染文本 |
| `/mem purge` | 重扫已有记忆，把污染条目移出召回池 |
| `/mem eval [seed]` | 跑金标准；`seed` 用现有记忆重建测试集 |
| `/mem tune k=v [apply]` | 带回归门禁地改参数 |

## 评估怎么用

```text
/mem eval seed          # 用现有记忆生成正例，加上内置负例
/mem eval               # 看当前 P@1 / P@3 / R@3 / MRR / 误召回率
/mem tune min_score=0.28 margin=0.03     # 只看不应用
/mem tune min_score=0.28 apply           # 通过门禁才写入覆盖配置
```

门禁规则写死在 `evaluate.py`：P@1 或 R@3 掉超过 0.02、误召回率涨超过 0.03，直接拒绝应用。这样「优化」不会悄悄把系统改坏——这正是第三个短板要的解药。

金标准是 `golden.jsonl`，一行一条：

```json
{"query": "我住在哪里？", "expected": [3], "expect_none": false, "note": "auto:live:杭州"}
{"query": "那怎么提高命中率", "expected": [], "expect_none": true, "note": "negative"}
```

可以手工补充真实用户的问法，尤其是**本不该召回**的句子——负例比正例更能暴露精度问题。

## 配置

WebUI 插件配置页可直接改（`_conf_schema.json`）：

| 键 | 默认 | 说明 |
|---|---|---|
| `enabled` | true | 总开关 |
| `auto_capture` | true | 是否自动抽取事实 |
| `scope` | global | global 跨会话 / session 仅当前会话 |
| `max_items` | 4 | 单次注入上限 |
| `min_score` | 0.34 | 召回最低分 |
| `margin` | 0.04 | 第一二名最小分差 |
| `contamination_threshold` | 0.55 | 污染隔离阈值 |
| `recency_half_life_days` | 120 | 新鲜度半衰期 |
| `debug` | false | 输出门控日志 |

## 设计取舍

**为什么先门控再检索。** 13.6% 的命中率意味着绝大多数召回是噪声。与其把排序做得更花，不如先承认「这句话不需要记忆」。门控用规则而不是模型：可解释、零延迟、不会把「提高命中率」当成用户偏好。

**为什么不用纯向量。** 个人记忆是短句、专名、数字，向量对「猫」和「狗」的区分不如字面匹配稳定；而 BM25 能直接说清命中了哪几个词。真要用 embedding，门控和阈值仍然需要，向量只是换掉打分函数。

**为什么污染在写入时拦，而不是检索时拦。** 检索时拦截等于让脏数据一直躺在库里参与打分。写入时隔离，召回池永远是干净的；隔离区仍可审计，不会「悄悄丢掉用户说过的话」。

**为什么评估要内置负例。** 只测「该召回的召回了吗」，会把阈值越调越低，最后退化成什么都召回。误召回率必须和召回率一起看。

## 已知边界

- 事实抽取是规则式的：覆盖自述、偏好、居住、工作、宠物、生日、过敏、习惯、计划。复杂长句会漏，宁可漏也不乱记。
- 污染判定基于文本特征，不区分「用户本人在聊技术」和「用户在贴日志」。前者会被误隔离，可把 `contamination_threshold` 调高到 0.7 缓解。
- 单用户单库：没有做多租户隔离；群聊场景建议设 `scope=session`。
- 门控对反讽、省略主语的长句仍可能误判，`/mem why` 是排查入口。

## 离线测试

不需要装 AstrBot：

```bash
cd astrbot_plugin_memguard
python -m unittest discover -s tests -v
```

覆盖 14 个用例：元话语抑制、调试语境抑制、闲聊跳过、自述召回、「提高命中率」不召回、代码块隔离、事实抽取、功能性事实覆盖旧值、脏数据不落库、评估指标、门禁拒绝退化配置。
