"""离线测试：不依赖 AstrBot，直接验证三个短板的修复逻辑。

运行：cd astrbot_plugin_memguard && python -m unittest discover -s tests -v
"""

from __future__ import annotations

import shutil
import sys
import unittest
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from memguard.config import Settings  # noqa: E402
from memguard.engine import Engine  # noqa: E402
from memguard.extract import classify_contamination, extract_facts  # noqa: E402
from memguard.gate import decide  # noqa: E402
from memguard.store import Store  # noqa: E402


class GateTests(unittest.TestCase):
    def test_meta_discussion_is_suppressed(self):
        decision = decide("那怎么提高命中率")
        self.assertFalse(decision.should_recall)
        self.assertIn("元话语", decision.reason)

    def test_debug_talk_is_suppressed(self):
        decision = decide("接口 500 报错，看下日志和请求体")
        self.assertFalse(decision.should_recall)

    def test_chitchat_is_suppressed(self):
        self.assertFalse(decide("今天天气不错").should_recall)

    def test_personal_question_recalls(self):
        decision = decide("我住在哪里？")
        self.assertTrue(decision.should_recall)
        self.assertEqual(decision.intent, "live")

    def test_explicit_recall_request(self):
        decision = decide("你还记得我喜欢什么吗")
        self.assertTrue(decision.should_recall)
        self.assertIn("显式", decision.reason)


class ExtractionTests(unittest.TestCase):
    def test_code_block_is_quarantined(self):
        text = "```python\nprint('hello')\n```\n接口返回 500"
        result = classify_contamination(text)
        self.assertTrue(result.quarantined)

    def test_personal_sentence_is_clean(self):
        facts, contamination = extract_facts("我叫小林，住在杭州，养了一只猫叫豆豆")
        self.assertFalse(contamination.quarantined)
        predicates = {f.predicate for f in facts}
        self.assertIn("name", predicates)
        self.assertIn("live", predicates)
        self.assertIn("pet", predicates)
        name = next(f for f in facts if f.predicate == "name")
        self.assertEqual(name.obj, "小林")

    def test_quarantined_yields_no_facts(self):
        facts, contamination = extract_facts("你的回答里 traceback 显示接口 500 报错")
        self.assertEqual(facts, [])
        self.assertTrue(contamination.quarantined)


class EngineTests(unittest.TestCase):
    def setUp(self):
        # 用工作区内的独立目录：系统临时目录在受限环境下可能不可写。
        self.tmp = Path(__file__).resolve().parent / "_tmp" / uuid.uuid4().hex[:8]
        self.tmp.mkdir(parents=True)
        self.addCleanup(shutil.rmtree, self.tmp, True)
        self.store = Store(self.tmp / "mem.db")
        self.addCleanup(self.store.close)
        self.engine = Engine(self.store, Settings(), self.tmp)

    def _seed(self):
        for text in (
            "我叫小林，住在杭州",
            "我喜欢猫",
            "我对花生过敏",
        ):
            self.engine.capture("s1", "user", text)

    def test_recall_hits_the_right_memory(self):
        self._seed()
        result = self.engine.recall("s1", "我住在哪里？")
        self.assertTrue(result.hits)
        self.assertIn("杭州", result.hits[0].memory.text)

    def test_recall_does_not_fire_on_meta_talk(self):
        self._seed()
        result = self.engine.recall("s1", "那怎么提高命中率")
        self.assertFalse(result.hits)
        self.assertEqual(result.context, "")

    def test_functional_fact_supersedes_old_value(self):
        self._seed()
        self.engine.capture("s1", "user", "我搬到深圳了")
        live = [m for m in self.store.active_memories() if m.predicate == "live"]
        self.assertEqual(len(live), 1)
        self.assertIn("深圳", live[0].text)

    def test_dirty_turn_never_becomes_memory(self):
        before = self.store.stats()["active"]
        self.engine.capture("s1", "user", "```python\nTraceback: 500 接口报错\n```")
        self.assertEqual(self.store.stats()["active"], before)

    def test_eval_reports_quality(self):
        self._seed()
        metrics, cases = self.engine.evaluate(refresh=True)
        self.assertGreaterEqual(metrics.precision_at_1, 0.6)
        self.assertLessEqual(metrics.no_answer_fpr, 0.2)
        self.assertTrue(any(c.expect_none for c in cases))

    def test_tune_rejects_regression(self):
        self._seed()
        comparison = self.engine.tune({"min_score": 0.99})
        self.assertFalse(comparison.accepted)
        self.assertTrue(comparison.reasons)


if __name__ == "__main__":
    unittest.main()
