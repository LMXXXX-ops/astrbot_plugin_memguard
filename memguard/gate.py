"""召回门控：先判断「这句话到底需不需要查长期记忆」。

13.6% 命中率的根因通常不是排序差，而是**不该召回的句子也去召回了**：
闲聊、调试、讨论记忆系统本身，都会被向量相似度硬凑出一个结果。
门控的目标是：宁可这一次不召回，也不要把不相干的记忆塞进上下文。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from .text import META_TERMS, TECH_TERMS, contains_any, normalize

# 强召回线索：用户在明确要求「回忆」。
STRONG_CUES = (
    r"你还记得", r"记得我", r"记不记得", r"还记得",
    r"我(?:之前|上次|以前|刚才)(?:说|提|讲|问)过",
    r"(?:之前|上次|以前)(?:说|提|讲|问)过",
    r"我说过的", r"我告诉过你", r"我跟你说过",
    r"我们(?:之前|上次|以前)",
)

# 自述线索：第一人称 + 稳定属性/偏好/经历。
SELF_PATTERNS = (
    r"我(?:的)?名字", r"我叫", r"我是", r"我在.{0,12}(?:工作|上班|读书|上学)",
    r"我(?:住|住在|搬到了?|来自)", r"我(?:养|有(?:一|两|三)?(?:只|个|条))",
    r"我(?:喜欢|爱|讨厌|不喜欢|害怕|习惯|擅长|不会|会)",
    r"我(?:的)?(?:生日|年龄|专业|职业|工作|公司|学校|爱好|宠物|老家|家乡)",
    r"我(?:最近|正在|打算|计划|准备|想要|想)",
    r"我(?:吃素|过敏|戒了|在减肥|在健身)",
    r"对.{0,6}过敏",
)

# 询问自身信息的疑问句：没有「我」也常见，例如「我喜欢什么来着」有；「我养了什么」有。
QUESTION_SELF = (
    r"我(?:喜欢|爱|讨厌|养|住|叫|是|在做|在做什么|的)",
    r"(?:我|俺)(?:的)?[^？?]{0,10}(?:是什么|是谁|有哪些|在哪|几点|多少|怎么样)",
    r"我(?:平时|一般|都)?(?:用|做|干|养|吃|喜欢|住|叫)[^？?]{0,8}(?:什么|哪|谁|多少)",
    r"我[^？?]{0,8}(?:习惯|平时|一般)[^？?]{0,8}(?:什么|做|干)",
)

# 明确的技术/调试语境：命中两条以上时判定为开发对话。
DEBUG_PATTERNS = (
    r"```", r"\b(?:traceback|exception|error)\b", r"报错", r"堆栈",
    r"(?:get|post|put|delete)\s+/", r"\b\d{3}\b\s*(?:错误|状态码)",
    r"接口|请求体|响应体|返回(?:值|结果)", r"单元测试|测试用例|跑一下|复现",
)

# 查询意图 → 谓词。命中后给同谓词记忆加权，是提升精度的关键结构信息。
INTENTS: tuple[tuple[str, str], ...] = (
    ("name", r"名字|我叫什么|怎么称呼|我是谁"),
    ("live", r"住在|住哪|老家|来自|搬到|城市"),
    ("job", r"工作|职业|做什么的|上班|公司|职位"),
    ("pet", r"宠物|养了|养什么|猫|狗"),
    ("birthday", r"生日|出生|几岁|年龄"),
    ("allergy", r"过敏"),
    ("dislike", r"讨厌|不喜欢|受不了|害怕"),
    ("like", r"喜欢|爱好|爱吃什么|爱玩"),
    ("habit", r"习惯|平时|一般会"),
    ("plan", r"计划|打算|准备|最近在"),
    ("tool_pref", r"包管理|依赖管理|用什么(?:工具|包)|pnpm|npm"),
)


@dataclass
class GateDecision:
    """一次召回判定的完整依据，便于 /mem why 解释。"""

    should_recall: bool
    reason: str
    query: str
    meta_hits: list[str] = field(default_factory=list)
    tech_hits: list[str] = field(default_factory=list)
    cues: list[str] = field(default_factory=list)
    confidence: float = 0.0
    intent: str | None = None

    def explain(self) -> str:
        parts = [f"结论={'召回' if self.should_recall else '跳过'}", f"原因={self.reason}"]
        if self.intent:
            parts.append(f"意图={self.intent}")
        if self.cues:
            parts.append("线索=" + "、".join(self.cues[:4]))
        if self.meta_hits:
            parts.append("元话语=" + "、".join(self.meta_hits[:4]))
        if self.tech_hits:
            parts.append("技术词=" + "、".join(self.tech_hits[:4]))
        return "；".join(parts)


def detect_intent(text: str) -> str | None:
    # 「我养的狗叫什么名字」里有「名字」，但它问的是宠物，不是本人姓名。
    if re.search(r"宠物|狗|猫", text):
        return "pet"
    for predicate, pattern in INTENTS:
        if re.search(pattern, text):
            return predicate
    return None


def _matches(patterns: tuple[str, ...], text: str) -> list[str]:
    hits: list[str] = []
    for pattern in patterns:
        found = re.search(pattern, text)
        if found:
            hits.append(found.group(0))
    return hits


def decide(message: str) -> GateDecision:
    """判定是否召回，并给出可直接用于检索的清洗后查询串。"""
    text = normalize(message)
    if not text:
        return GateDecision(False, "空消息", "", confidence=0.0)

    # 极短消息（表情、单字）几乎不携带可检索信息。
    if len(text) <= 2 and not re.search(r"[我你]", text):
        return GateDecision(False, "消息过短", text, confidence=0.0)

    cues = _matches(STRONG_CUES, text)
    self_hits = _matches(SELF_PATTERNS, text) + _matches(QUESTION_SELF, text)
    meta_hits = contains_any(text, META_TERMS)
    tech_hits = contains_any(text, TECH_TERMS)
    debug_hits = _matches(DEBUG_PATTERNS, text)

    # 1) 用户明确要求回忆：最高优先级，即使句子里混着技术词。
    if cues:
        query = _clean_query(text)
        return GateDecision(
            True, "显式回忆请求", query, meta_hits, tech_hits, cues, 0.95, intent=detect_intent(query)
        )

    # 2) 讨论记忆/检索系统本身，且没有任何自述：这是元话语，不是用户事实。
    if meta_hits and not self_hits:
        return GateDecision(
            False,
            "元话语（在讨论系统而不是陈述自己）",
            text,
            meta_hits,
            tech_hits,
            cues,
            0.9,
        )

    # 3) 开发/调试语境：出现代码块或两条以上技术词，判定为工作对话。
    if (debug_hits and not self_hits) or len(tech_hits) >= 2 and not self_hits:
        return GateDecision(
            False,
            "技术语境（开发/调试对话不应写入或触发个人记忆）",
            text,
            meta_hits,
            tech_hits + debug_hits,
            cues,
            0.8,
        )

    # 4) 第一人称自述或询问自身信息：召回。
    if self_hits:
        query = _clean_query(text)
        return GateDecision(
            True, "第一人称自述/询问", query, meta_hits, tech_hits, self_hits, 0.75,
            intent=detect_intent(query),
        )

    # 5) 其余情况：闲聊、命令、泛问，默认不召回。
    return GateDecision(False, "无个人线索的普通消息", text, meta_hits, tech_hits, cues, 0.2)


def _clean_query(text: str) -> str:
    """去掉元话语与技术噪声，留下更接近「用户事实」的检索词。"""
    cleaned = text
    for pattern in (
        r"你还记得[^，。？?]*[吗么]?", r"记不记得", r"还记得", r"记得我(?:说|讲|提)过",
        r"我(?:之前|上次|以前)(?:说|提|讲|问)过",
    ):
        cleaned = re.sub(pattern, " ", cleaned)
    return normalize(cleaned) or text


