"""离线评估：金标准集、指标、A/B 对比与回归门禁。

第三个短板是「没有评估集」。这里把评估集变成插件的一部分：
  - 有正例（该召回什么）也有负例（不该召回什么）；
  - 每次改检索参数前后各跑一次，用数字判断而不是凭感觉；
  - 指标退化时拒绝应用新配置。
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field

from .config import Settings
from .gate import decide
from .retrieve import Retriever
from .store import Memory

# 负例：闲聊、元话语、调试对话。命中任何一条都算误召回。
NEGATIVE_QUERIES = [
    "今天天气怎么样",
    "那怎么提高命中率",
    "接口 500 报错怎么回事",
    "帮我看看这段代码有没有问题",
    "晚上吃什么好呢",
    "这个插件的配置项在哪",
    "哈哈哈哈",
]

QUERY_TEMPLATES = {
    "name": "我叫什么名字？",
    "like": "我喜欢什么？",
    "dislike": "我讨厌什么？",
    "live": "我住在哪里？",
    "job": "我是做什么工作的？",
    "pet": "我养了什么宠物？",
    "birthday": "我的生日是哪天？",
    "allergy": "我对什么过敏？",
    "habit": "我平时有什么习惯？",
    "plan": "我最近在计划什么？",
    "note": "我之前让你记住过什么？",
}


@dataclass
class Case:
    query: str
    expected: list[int] = field(default_factory=list)
    expect_none: bool = False
    note: str = ""

    def to_dict(self) -> dict:
        return {"query": self.query, "expected": self.expected, "expect_none": self.expect_none, "note": self.note}

    @classmethod
    def from_dict(cls, data: dict) -> "Case":
        return cls(
            query=str(data.get("query", "")),
            expected=[int(x) for x in data.get("expected", [])],
            expect_none=bool(data.get("expect_none", False)),
            note=str(data.get("note", "")),
        )


@dataclass
class Metrics:
    cases: int
    precision_at_1: float
    precision_at_3: float
    recall_at_3: float
    mrr: float
    no_answer_fpr: float
    gate_pass_rate: float
    latency_ms: float
    failures: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "cases": self.cases,
            "p@1": round(self.precision_at_1, 4),
            "p@3": round(self.precision_at_3, 4),
            "r@3": round(self.recall_at_3, 4),
            "mrr": round(self.mrr, 4),
            "no_answer_fpr": round(self.no_answer_fpr, 4),
            "gate_pass_rate": round(self.gate_pass_rate, 4),
            "latency_ms": round(self.latency_ms, 2),
        }


def seed_cases(memories: list[Memory], *, limit: int = 40) -> list[Case]:
    """从现有记忆自动生成正例，再补一组固定负例。"""
    cases: list[Case] = []
    for memory in memories:
        template = QUERY_TEMPLATES.get(memory.predicate)
        if not template:
            continue
        cases.append(Case(template, [memory.id], False, f"auto:{memory.key}"))
        if len(cases) >= limit:
            break
    cases.extend(Case(query, [], True, "negative") for query in NEGATIVE_QUERIES)
    return cases


def run_eval(cases: list[Case], memories: list[Memory], settings: Settings) -> Metrics:
    retriever = Retriever(memories, settings)
    by_id = {m.id: m for m in memories}

    pos_total = pos_hit1 = pos_hit3 = 0
    recall_sum = 0.0
    rr_sum = 0.0
    neg_total = neg_fp = 0
    gate_pass = 0
    gate_checked = 0
    latencies: list[float] = []
    failures: list[dict] = []

    for case in cases:
        started = time.perf_counter()
        decision = decide(case.query)
        gate_checked += 1
        if decision.should_recall:
            gate_pass += 1
        hits = retriever.select(decision.query, "", decision.intent)[0] if decision.should_recall else []
        latencies.append((time.perf_counter() - started) * 1000)

        if case.expect_none:
            neg_total += 1
            if hits:
                neg_fp += 1
                failures.append(
                    {"query": case.query, "kind": "false_positive", "got": [h.id for h in hits]}
                )
            continue

        pos_total += 1
        expected = {i for i in case.expected if i in by_id}
        if not expected:
            continue
        ranked = [h.id for h in hits]
        if ranked and ranked[0] in expected:
            pos_hit1 += 1
        top3 = ranked[:3]
        if any(i in expected for i in top3):
            pos_hit3 += 1
        recall_sum += len([i for i in top3 if i in expected]) / len(expected)
        rank = next((idx + 1 for idx, i in enumerate(ranked) if i in expected), None)
        rr_sum += (1.0 / rank) if rank else 0.0
        if rank is None or rank > 1:
            failures.append(
                {"query": case.query, "kind": "miss" if rank is None else "rank", "expected": sorted(expected), "got": ranked[:3]}
            )

    latencies.sort()

    def pct(values: list[float], q: float) -> float:
        if not values:
            return 0.0
        idx = min(len(values) - 1, int(len(values) * q))
        return values[idx]

    return Metrics(
        cases=len(cases),
        precision_at_1=(pos_hit1 / pos_total) if pos_total else 0.0,
        precision_at_3=(pos_hit3 / pos_total) if pos_total else 0.0,
        recall_at_3=(recall_sum / pos_total) if pos_total else 0.0,
        mrr=(rr_sum / pos_total) if pos_total else 0.0,
        no_answer_fpr=(neg_fp / neg_total) if neg_total else 0.0,
        gate_pass_rate=(gate_pass / gate_checked) if gate_checked else 0.0,
        latency_ms=pct(latencies, 0.5),
        failures=failures[:20],
    )


@dataclass
class Comparison:
    base: Metrics
    candidate: Metrics
    accepted: bool
    reasons: list[str]

    def to_dict(self) -> dict:
        return {
            "base": self.base.to_dict(),
            "candidate": self.candidate.to_dict(),
            "accepted": self.accepted,
            "reasons": self.reasons,
        }


# 回归门禁：任何一项恶化超过容差就拒绝。
TOLERANCE = {
    "precision_at_1": 0.02,
    "recall_at_3": 0.02,
    "no_answer_fpr": 0.03,
}


def compare(base: Metrics, candidate: Metrics) -> Comparison:
    reasons: list[str] = []
    if candidate.precision_at_1 < base.precision_at_1 - TOLERANCE["precision_at_1"]:
        reasons.append(f"P@1 退化 {base.precision_at_1:.3f} → {candidate.precision_at_1:.3f}")
    if candidate.recall_at_3 < base.recall_at_3 - TOLERANCE["recall_at_3"]:
        reasons.append(f"R@3 退化 {base.recall_at_3:.3f} → {candidate.recall_at_3:.3f}")
    if candidate.no_answer_fpr > base.no_answer_fpr + TOLERANCE["no_answer_fpr"]:
        reasons.append(
            f"误召回率上升 {base.no_answer_fpr:.3f} → {candidate.no_answer_fpr:.3f}"
        )
    return Comparison(base, candidate, accepted=not reasons, reasons=reasons)


def load_cases(path: str) -> list[Case]:
    cases: list[Case] = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            cases.append(Case.from_dict(json.loads(line)))
    return cases


def dump_cases(cases: list[Case], path: str) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        for case in cases:
            handle.write(json.dumps(case.to_dict(), ensure_ascii=False) + "\n")
