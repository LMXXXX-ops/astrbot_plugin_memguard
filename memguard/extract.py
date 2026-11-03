"""污染识别与事实抽取。

chatlog 里混着调试、测试、接口讨论。若不区分，回顾时会把它们当成「用户的事」。
这里的做法是：写入前先算污染分，超阈值直接进隔离区，永不参与召回。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from .text import TECH_TERMS, contains_any, normalize

CODE_FENCE = re.compile(r"```|\bdef\s+\w+\(|\bclass\s+\w+|[{};]\s*$", re.M)
URL = re.compile(r"https?://\S+")
JSONISH = re.compile(r'^\s*[\[{].*[\]}]\s*$', re.S)
ERROR_LINE = re.compile(r"(?:Traceback|Error|Exception|错误|失败|超时|500|404|403)", re.I)
BOT_META = re.compile(r"(?:你刚才|你上次|你的回答|你回复|你说的|你生成|作为(?:一个)?(?:AI|助手|模型))")

# 事实模板：predicate -> (正则, 重要度)
FACT_PATTERNS: list[tuple[str, re.Pattern, float]] = [
    ("name", re.compile(r"我(?:的名字)?(?:叫|是)\s*([\u4e00-\u9fffA-Za-z0-9_·]{1,12})"), 0.9),
    ("like", re.compile(r"我(?:很|超|最)?(?:喜欢|爱|迷上了|粉)\s*([^，。！？,.!?]{1,24})"), 0.7),
    ("dislike", re.compile(r"我(?:很|超|最)?(?:讨厌|不喜欢|受不了|害怕)\s*([^，。！？,.!?]{1,24})"), 0.7),
    ("live", re.compile(r"(?:我)?(?:现在)?(?:住在|住|搬到了?|来自|老家在)\s*([^，。！？,.!?]{1,20})"), 0.85),
    ("job", re.compile(r"我(?:在|是|做)\s*([^，。！？,.!?]{2,20}?)(?:工作|上班|工程师|程序员|老师|学生|医生|设计师|运营|产品经理)"), 0.8),
    ("pet", re.compile(r"(?:我)?养了?\s*(?:一|两|三|四|五|\d+)?\s*(?:只|条|个)?\s*([^，。！？,.!?]{1,16}?)(?:，|。|$|很|叫)"), 0.75),
    ("birthday", re.compile(r"我(?:的)?生日(?:是|在)?\s*([0-9]{1,2}\s*月\s*[0-9]{1,2}\s*[日号]|[0-9]{4}-[0-9]{1,2}-[0-9]{1,2})"), 0.9),
    ("allergy", re.compile(r"我(?:对)?\s*([^，。！？,.!?]{1,12}?)\s*过敏"), 0.85),
    ("habit", re.compile(r"我(?:通常|一般|习惯|总是)\s*([^，。！？,.!?]{2,28})"), 0.6),
    ("plan", re.compile(r"我(?:最近|正在|打算|计划|准备)\s*([^，。！？,.!?]{2,28})"), 0.55),
    ("tool_pref", re.compile(r"我(?:一般|通常|都)?(?:用|使用|不用|不用再)\s*([A-Za-z][A-Za-z0-9_+.#-]{1,20})"), 0.5),
]

# 允许省略主语的谓词：只有整句话里出现过「我」时才启用，避免把「他养了只猫」记到用户头上。
OPTIONAL_SUBJECT = {"live", "pet"}

# 显式记忆指令：以「记住」开头的整句都是用户主动要求保存的内容。
REMEMBER_CMD = re.compile(r"^(?:请)?记住[：:，,\s]*(.{2,120})$")


@dataclass
class Contamination:
    score: float
    labels: list[str] = field(default_factory=list)
    quarantined: bool = False

    def explain(self) -> str:
        if not self.labels:
            return f"污染分 {self.score:.2f}（干净）"
        return f"污染分 {self.score:.2f}（{'、'.join(self.labels)}）"


@dataclass
class Fact:
    predicate: str
    obj: str
    text: str
    importance: float
    source: str = "auto"

    def key(self) -> str:
        return f"{self.predicate}:{self.obj}"


def classify_contamination(text: str, *, threshold: float = 0.55) -> Contamination:
    """给一段文本打污染分。分数越高，越不可能是「用户本人的稳定事实」。"""
    raw = text or ""
    norm = normalize(raw)
    score = 0.0
    labels: list[str] = []

    if CODE_FENCE.search(raw):
        score += 0.45
        labels.append("代码块")
    if URL.search(raw):
        score += 0.12
        labels.append("链接")
    if JSONISH.match(raw.strip()):
        score += 0.2
        labels.append("结构化数据")
    if ERROR_LINE.search(raw):
        score += 0.12
        labels.append("报错信息")
    if BOT_META.search(norm):
        score += 0.15
        labels.append("对机器人说话")

    tech_hits = contains_any(norm, TECH_TERMS)
    if tech_hits:
        score += min(0.4, 0.12 * len(tech_hits))
        labels.append(f"技术词×{len(tech_hits)}")

    # 长文本且没有第一人称，通常是从别处贴过来的资料。
    if len(norm) > 260 and "我" not in norm:
        score += 0.2
        labels.append("长文本无自述")

    # 第一人称自述会显著降低污染概率。
    if re.search(r"我(?:的)?(?:叫|是|住|喜欢|讨厌|养|在|打算|习惯)", norm):
        score -= 0.25

    score = max(0.0, min(1.0, score))
    return Contamination(round(score, 3), labels, score >= threshold)


def extract_facts(text: str, *, threshold: float = 0.55) -> tuple[list[Fact], Contamination]:
    """从一句话里抽取可长期保存的事实。污染超阈值时返回空列表。"""
    norm = normalize(text)
    contamination = classify_contamination(norm, threshold=threshold)
    if contamination.quarantined:
        return [], contamination

    facts: list[Fact] = []

    explicit = REMEMBER_CMD.search(norm)
    if explicit:
        content = explicit.group(1).strip(" 。.!！")
        if content:
            facts.append(Fact("note", content, content, 0.95, source="explicit"))
        return facts, contamination

    for predicate, pattern, importance in FACT_PATTERNS:
        if predicate in OPTIONAL_SUBJECT and "我" not in norm:
            continue
        match = pattern.search(norm)
        if not match:
            continue
        obj = match.group(1).strip(" 的了是在")
        if not obj or len(obj) > 40:
            continue
        if re.fullmatch(r"[你我他她它]", obj):
            continue
        facts.append(
            Fact(
                predicate=predicate,
                obj=obj,
                text=_phrase(predicate, obj),
                importance=importance,
            )
        )

    # 同一句里同一 predicate 只保留第一个，避免「我喜欢猫也喜欢狗」被拆碎。
    seen: set[str] = set()
    unique: list[Fact] = []
    for fact in facts:
        if fact.predicate in seen:
            continue
        seen.add(fact.predicate)
        unique.append(fact)

    return unique, contamination


def _phrase(predicate: str, obj: str) -> str:
    """把三元组写成一句自然中文，让检索能命中用户的真实问法。"""
    return {
        "name": f"用户的名字叫{obj}",
        "like": f"用户喜欢{obj}",
        "dislike": f"用户不喜欢{obj}",
        "live": f"用户住在{obj}",
        "job": f"用户的工作是{obj}",
        "pet": f"用户养了{obj}作为宠物",
        "birthday": f"用户的生日是{obj}",
        "allergy": f"用户对{obj}过敏",
        "habit": f"用户习惯{obj}",
        "plan": f"用户最近计划{obj}",
        "tool_pref": f"用户通常使用{obj}",
        "note": obj,
    }.get(predicate, f"{predicate}：{obj}")
