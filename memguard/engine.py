"""引擎：把门控、抽取、存储、检索、评估串成一条可测试的流水线。

main.py 只是 AstrBot 的适配层；所有可测试逻辑都在这里，便于离线跑单元测试。
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, replace
from pathlib import Path

from .config import Settings
from .evaluate import Case, Comparison, Metrics, compare, load_cases, run_eval, seed_cases, dump_cases
from .extract import classify_contamination, extract_facts
from .gate import GateDecision, decide
from .retrieve import Hit, Retriever
from .store import Memory, Store

# 这些谓词在同一时间只可能有一个真值，新值覆盖旧值。
FUNCTIONAL = {"name", "live", "job", "birthday"}


@dataclass
class RecallResult:
    decision: GateDecision
    hits: list[Hit]
    context: str
    reason: str = ""

    @property
    def injected(self) -> bool:
        return bool(self.hits)


class Engine:
    def __init__(self, store: Store, settings: Settings, data_dir: str | Path):
        self.store = store
        self.data_dir = Path(data_dir)
        self.settings = self._load_override(settings)
        self.last: dict[str, RecallResult] = {}

    # ------------------------------------------------------------ settings

    def _override_path(self) -> Path:
        return self.data_dir / "settings_override.json"

    def _load_override(self, settings: Settings) -> Settings:
        path = self._override_path()
        if not path.exists():
            return settings
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            return replace(settings, **{k: v for k, v in data.items() if hasattr(settings, k)})
        except (OSError, ValueError, TypeError):
            return settings

    def save_override(self) -> None:
        payload = {
            "min_score": self.settings.min_score,
            "margin": self.settings.margin,
            "max_items": self.settings.max_items,
            "scope": self.settings.scope,
            "recency_half_life_days": self.settings.recency_half_life_days,
        }
        self._override_path().write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    # ------------------------------------------------------------ capture

    def capture(self, session: str, role: str, text: str) -> dict:
        """记录一轮对话；只把「干净且像用户事实」的内容写进记忆。"""
        text = (text or "").strip()
        if not text:
            return {"facts": 0, "quarantined": False}

        if role != "user":
            self.store.log_turn(session, role, text, 0.0, [])
            return {"facts": 0, "quarantined": False}

        facts, contamination = extract_facts(text, threshold=self.settings.contamination_threshold)
        self.store.log_turn(session, role, text, contamination.score, contamination.labels)

        if contamination.quarantined:
            return {"facts": 0, "quarantined": True, "score": contamination.score}

        written = 0
        for fact in facts:
            if not self.settings.auto_capture and fact.source != "explicit":
                continue
            memory_id, action = self.store.add_fact(
                predicate=fact.predicate,
                obj=fact.obj,
                text=fact.text,
                importance=fact.importance,
                scope=self.settings.scope if fact.source != "explicit" else "global",
                session=session if self.settings.scope == "session" else "",
                source=fact.source,
            )
            written += 1
            if fact.predicate in FUNCTIONAL:
                self._archive_siblings(memory_id, fact.predicate)

        return {"facts": written, "quarantined": False, "score": contamination.score}

    def _archive_siblings(self, keep_id: int, predicate: str) -> None:
        for memory in self.store.active_memories():
            if memory.predicate == predicate and memory.id != keep_id:
                self.store.set_status(memory.id, "archived")

    # ------------------------------------------------------------ recall

    def recall(self, session: str, message: str) -> RecallResult:
        decision = decide(message)
        if not decision.should_recall:
            result = RecallResult(decision, [], "", decision.reason)
            self.last[session] = result
            self.store.log_retrieval(session, message[:200], "skip:" + decision.reason, [], [])
            return result

        memories = self.store.active_memories(session)
        retriever = Retriever(memories, self.settings)
        hits, reason = retriever.select(decision.query, session, decision.intent)
        context = retriever.format_context(hits)
        result = RecallResult(decision, hits, context, reason)
        self.last[session] = result

        self.store.log_retrieval(
            session,
            decision.query[:200],
            ("hit:" + reason) if hits else ("miss:" + reason),
            [h.id for h in hits],
            [h.score for h in hits],
        )
        self.store.bump_hit([h.id for h in hits])
        return result

    def why(self, session: str) -> str:
        result = self.last.get(session)
        if not result:
            return "本会话还没有发生过召回判定。"
        lines = [
            f"门控：{result.decision.explain()}",
            f"检索：{result.reason or '未执行'}",
        ]
        for hit in result.hits:
            lines.append(f"  #{hit.id} {hit.score:.3f} {hit.memory.text}")
        return "\n".join(lines)

    # ------------------------------------------------------------ management

    def stats(self) -> dict:
        data = self.store.stats()
        decisions = self.store.retrieval_report()
        data["decisions"] = decisions
        data["settings"] = asdict(self.settings)
        return data

    def search(self, query: str, session: str = "") -> list[Hit]:
        memories = self.store.active_memories(session or None)
        return Retriever(memories, self.settings).rank(query, session)[:10]

    def add_manual(self, text: str, session: str) -> int:
        memory_id, _ = self.store.add_fact(
            predicate="note",
            obj=text[:60],
            text=text,
            importance=0.95,
            scope="global",
            session=session,
            source="explicit",
        )
        return memory_id

    def audit(self, limit: int = 10) -> list[dict]:
        """列出被隔离的高污染文本，方便人工复查。"""
        rows = self.store.all_memories("quarantine", limit)
        return [{"id": m.id, "text": m.text, "score": m.confidence} for m in rows]

    def purge(self) -> int:
        """重扫现有记忆，把事后看出的污染条目摘出召回池。"""
        moved = 0
        for memory in self.store.all_memories(None, 2000):
            if memory.status != "active" or memory.source == "explicit":
                continue
            contamination = classify_contamination(
                memory.text, threshold=self.settings.contamination_threshold
            )
            if contamination.quarantined:
                self.store.set_status(memory.id, "quarantine")
                moved += 1
        return moved

    # ------------------------------------------------------------ eval

    def golden_path(self) -> Path:
        return self.data_dir / "golden.jsonl"

    def cases(self, refresh: bool = False) -> list[Case]:
        path = self.golden_path()
        if path.exists() and not refresh:
            try:
                return load_cases(str(path))
            except (OSError, ValueError):
                pass
        memories = self.store.active_memories()
        cases = seed_cases(memories)
        dump_cases(cases, str(path))
        return cases

    def evaluate(self, refresh: bool = False) -> tuple[Metrics, list[Case]]:
        cases = self.cases(refresh)
        metrics = run_eval(cases, self.store.active_memories(), self.settings)
        self.store.save_eval("current", metrics.to_dict())
        return metrics, cases

    def tune(self, patch: dict, apply_change: bool = False) -> Comparison:
        """用同一套金标准比较新旧配置；未通过门禁则拒绝应用。"""
        cases = self.cases()
        memories = self.store.active_memories()
        base = run_eval(cases, memories, self.settings)

        candidate_settings = replace(
            self.settings,
            **{k: v for k, v in patch.items() if hasattr(self.settings, k)},
        )
        candidate = run_eval(cases, memories, candidate_settings)
        comparison = compare(base, candidate)

        if comparison.accepted and apply_change:
            self.settings = candidate_settings
            self.save_override()
        return comparison
