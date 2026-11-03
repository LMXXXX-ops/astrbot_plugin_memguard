"""检索：BM25 + 重要度/新鲜度加成 + 绝对阈值 + 间隔门限。

刻意不用纯向量相似度：
  - 13.6% 的场景里，向量会把「语义接近但事实上无关」的句子排到第一；
  - BM25 对专名、数字、短偏好更稳，且可解释（能说清命中了哪些词）。
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime, timezone

from .config import Settings
from .store import Memory
from .text import tokenize


@dataclass
class Hit:
    memory: Memory
    score: float
    matched: list[str]

    @property
    def id(self) -> int:
        return self.memory.id


def _age_days(iso_ts: str) -> float:
    try:
        then = datetime.fromisoformat(iso_ts)
        if then.tzinfo is None:
            then = then.replace(tzinfo=timezone.utc)
        return max(0.0, (datetime.now(timezone.utc) - then).total_seconds() / 86400)
    except (ValueError, TypeError):
        return 365.0


class Retriever:
    """一次检索面对一组固定的记忆，构造后不可变，便于离线评估复现。"""

    K1 = 1.4
    B = 0.72

    def __init__(self, memories: list[Memory], settings: Settings):
        self.settings = settings
        self.memories = [m for m in memories if m.status == "active"]
        self.docs: list[list[str]] = [tokenize(m.text) for m in self.memories]
        self.lengths = [len(d) for d in self.docs]
        self.avg_len = (sum(self.lengths) / len(self.lengths)) if self.lengths else 0.0
        self.df: dict[str, int] = {}
        for doc in self.docs:
            for term in set(doc):
                self.df[term] = self.df.get(term, 0) + 1
        self.n = len(self.memories)

    def _idf(self, term: str) -> float:
        df = self.df.get(term, 0)
        return math.log(1 + (self.n - df + 0.5) / (df + 0.5))

    def _bm25(self, index: int, query_terms: list[str]) -> tuple[float, list[str]]:
        doc = self.docs[index]
        if not doc:
            return 0.0, []
        tf: dict[str, int] = {}
        for term in doc:
            tf[term] = tf.get(term, 0) + 1
        length = self.lengths[index] or 1
        score = 0.0
        matched: list[str] = []
        for term in set(query_terms):
            if term not in tf:
                continue
            idf = self._idf(term)
            freq = tf[term]
            denom = freq + self.K1 * (1 - self.B + self.B * length / (self.avg_len or 1))
            score += idf * (freq * (self.K1 + 1)) / denom
            matched.append(term)
        return score, matched

    def rank(self, query: str, session: str = "", intent: str | None = None) -> list[Hit]:
        """返回全部候选（已过滤作用域），按分数降序，不做阈值裁剪。"""
        query_terms = tokenize(query)
        if not query_terms or not self.memories:
            return []

        unique_terms = set(query_terms)
        hits: list[Hit] = []
        normalized_query = query.lower()

        for index, memory in enumerate(self.memories):
            if self.settings.scope == "session" and memory.scope == "session" and memory.session != session:
                continue
            raw, matched = self._bm25(index, query_terms)
            object_hit = 1.0 if memory.object and len(memory.object) >= 2 and memory.object.lower() in normalized_query else 0.0
            intent_match = 1.0 if intent and memory.predicate == intent else 0.0
            # 意图匹配的记忆即使没有字面重合也要进入排序：
            # 「我讨厌吃什么」和「用户不喜欢香菜」可以完全没有共同词。
            if raw <= 0 and object_hit == 0 and intent_match == 0:
                continue

            # 覆盖率：命中的查询词占比。BM25 的饱和值只作次要信号。
            coverage = len(set(matched)) / len(unique_terms)
            saturation = raw / (raw + 4.0) if raw > 0 else 0.0

            score = 0.55 * coverage + 0.30 * saturation + 0.35 * intent_match + 0.30 * object_hit
            score = min(1.0, score)

            importance = 0.5 if memory.importance is None else memory.importance
            score *= 1 + self.settings.importance_weight * (importance - 0.5)

            half_life = max(1.0, self.settings.recency_half_life_days)
            freshness = 0.85 + 0.15 * math.exp(-_age_days(memory.last_seen) / half_life)
            score *= freshness

            if memory.hits:
                score *= 1 + min(0.08, 0.01 * math.log1p(memory.hits))

            hits.append(Hit(memory, min(1.0, score), matched))

        hits.sort(key=lambda h: (h.score, h.memory.importance), reverse=True)
        return hits

    def select(self, query: str, session: str = "", intent: str | None = None) -> tuple[list[Hit], str]:
        """按阈值与间隔门限挑出最终注入的记忆，并返回决策原因。"""
        ranked = self.rank(query, session, intent)
        if not ranked:
            return [], "无候选"

        top = ranked[0]
        if top.score < self.settings.min_score:
            return [], f"最高分 {top.score:.2f} 低于阈值 {self.settings.min_score:.2f}"

        chosen = [top]
        for hit in ranked[1:]:
            if len(chosen) >= self.settings.max_items:
                break
            if hit.score < self.settings.min_score:
                break
            # 与上一条过于接近时只保留更靠前的一条，避免同义记忆重复占位。
            if chosen and abs(chosen[-1].score - hit.score) < 0.01 and hit.memory.key == chosen[-1].memory.key:
                continue
            chosen.append(hit)

        # 间隔门限：第一名与第二名咬得太紧，说明证据不充分，只保留第一名。
        if len(ranked) >= 2 and (top.score - ranked[1].score) < self.settings.margin:
            chosen = [top]
            return chosen, f"命中但区分度低（{top.score:.2f} vs {ranked[1].score:.2f}），仅取最佳一条"

        return chosen, f"命中 {len(chosen)} 条，最高分 {top.score:.2f}"

    def format_context(self, hits: list[Hit]) -> str:
        if not hits:
            return ""
        lines = [self.settings.inject_header, ""]
        for hit in hits:
            date = (hit.memory.last_seen or "")[:10]
            lines.append(f"- [#{hit.memory.id} {date}] {hit.memory.text}")
        lines.append("")
        lines.append("若这些信息与用户当前说法冲突，以用户当前说法为准，并说明你依据的是较早的记录。")
        return "\n".join(lines)

