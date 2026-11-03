"""AstrBot 适配层：把 MemGuard 引擎接到 LLM 请求、响应与指令上。

三个短板的对应关系：
  1) 召回精度 —— on_llm_request 先过门控，再按阈值+间隔取记忆；
  2) 数据污染 —— 每轮先算污染分，超阈值只进 turn 日志，绝不进记忆库；
  3) 没有评估集 —— /mem eval 跑金标准，/mem tune 带回归门禁改参数。
"""

from __future__ import annotations

import hashlib
import sys
import time
from pathlib import Path

_PLUGIN_DIR = Path(__file__).resolve().parent
if str(_PLUGIN_DIR) not in sys.path:
    sys.path.insert(0, str(_PLUGIN_DIR))

from astrbot.api import logger  # noqa: E402
from astrbot.api.event import AstrMessageEvent, filter  # noqa: E402
from astrbot.api.star import Context, Star  # noqa: E402

try:  # 4.5.7+ 的请求对象
    from astrbot.api.provider import ProviderRequest  # noqa: E402
except Exception:  # pragma: no cover - 旧版本兜底
    ProviderRequest = object  # type: ignore

from memguard.config import Settings  # noqa: E402
from memguard.engine import Engine  # noqa: E402
from memguard.store import Store  # noqa: E402


def _plugin_data_dir() -> Path:
    """AstrBot 数据目录；拿不到时退回插件目录下的 data（仅兜底）。"""
    try:
        from astrbot.core.utils.astrbot_path import get_astrbot_data_path

        return Path(get_astrbot_data_path()) / "plugin_data" / "astrbot_plugin_memguard"
    except Exception:
        path = _PLUGIN_DIR / "data"
        return path


class Main(Star):
    def __init__(self, context: Context, config: dict | None = None):
        super().__init__(context)
        # AstrBot 传入的可能是 AstrBotConfig（dict 子类）或普通 dict，两种都接受。
        if isinstance(config, dict):
            self.conf = dict(config)
        else:
            try:
                self.conf = {k: v for k, v in config.items()}  # type: ignore[union-attr]
            except Exception:
                self.conf = {}
        self.data_dir = _plugin_data_dir()
        self.data_dir.mkdir(parents=True, exist_ok=True)

        self.store = Store(self.data_dir / "memguard.db")
        self.engine = Engine(self.store, Settings.from_conf(self.conf), self.data_dir)
        self._recent: dict[str, tuple[str, float]] = {}

    # ------------------------------------------------------------ hooks

    @filter.on_llm_request()
    async def inject_memory(self, event: AstrMessageEvent, req: ProviderRequest):
        """请求前：采集这一轮的用户事实，并按门控结果注入记忆。"""
        try:
            if not self.engine.settings.enabled:
                return
            session = self._session(event)
            message = getattr(event, "message_str", "") or ""

            self._capture_once(session, message)
            result = self.engine.recall(session, message)
            if result.injected:
                block = result.context
                req.system_prompt = (req.system_prompt or "") + "\n\n" + block
                if self.engine.settings.debug:
                    logger.info("[memguard] %s | %s", result.decision.explain(), result.reason)
        except Exception as exc:  # 绝不影响正常对话
            logger.warning("[memguard] 注入失败：%s", exc)

    @filter.on_llm_response()
    async def observe_response(self, event: AstrMessageEvent, resp):
        """响应后：只记录助手文本，不从中抽取事实（避免把模型的话当用户事实）。"""
        try:
            text = getattr(resp, "completion_text", "") or ""
            if not text:
                return
            session = self._session(event)
            self.store.log_turn(session, "assistant", text, 0.0, [])
        except Exception as exc:
            logger.debug("[memguard] 记录回复失败：%s", exc)

    # ------------------------------------------------------------ commands

    @filter.command_group("mem")
    def mem(self):
        """记忆守卫：/mem status | why | search | add | forget | audit | purge | eval | seed | tune"""
        pass

    @mem.command("status")
    async def mem_status(self, event: AstrMessageEvent):
        """查看记忆库规模、门控分布与当前参数。"""
        stats = self.engine.stats()
        decisions = "，".join(f"{d['decision'][:12]}×{d['count']}" for d in stats["decisions"]) or "暂无"
        settings = stats["settings"]
        text = (
            "🧠 MemGuard\n"
            f"记忆：生效 {stats['active']}，隔离 {stats['quarantine']}，归档 {stats['archived']}\n"
            f"轮次 {stats['turns']}｜召回 {stats['retrievals']}｜评估 {stats['evals']}\n"
            f"判定：{decisions}\n"
            f"参数：阈值 {settings['min_score']:.2f}｜间隔 {settings['margin']:.2f}｜"
            f"上限 {settings['max_items']}｜污染阈值 {settings['contamination_threshold']:.2f}"
        )
        yield event.plain_result(text)

    @mem.command("why")
    async def mem_why(self, event: AstrMessageEvent):
        """解释上一条消息为什么召回/没召回。"""
        yield event.plain_result(self.engine.why(self._session(event)))

    @mem.command("search")
    async def mem_search(self, event: AstrMessageEvent, query: str = ""):
        """调试检索：/mem search 我喜欢什么"""
        if not query.strip():
            yield event.plain_result("用法：/mem search <查询>")
            return
        hits = self.engine.search(query.strip(), self._session(event))
        if not hits:
            yield event.plain_result("没有候选命中。")
            return
        lines = [f"{i + 1}. {h.score:.3f} [#{h.id}] {h.memory.text}" for i, h in enumerate(hits)]
        yield event.plain_result("🔎 候选：\n" + "\n".join(lines))

    @mem.command("add")
    async def mem_add(self, event: AstrMessageEvent, text: str = ""):
        """手动写入一条记忆：/mem add 我对花生过敏"""
        if not text.strip():
            yield event.plain_result("用法：/mem add <内容>")
            return
        memory_id = self.engine.add_manual(text.strip(), self._session(event))
        yield event.plain_result(f"已记住 #{memory_id}。")

    @mem.command("forget")
    async def mem_forget(self, event: AstrMessageEvent, memory_id: str = ""):
        """删除一条记忆：/mem forget 12"""
        try:
            ident = int(memory_id.strip())
        except ValueError:
            yield event.plain_result("用法：/mem forget <记忆编号>")
            return
        ok = self.store.delete(ident)
        yield event.plain_result("已删除。" if ok else "没有这条记忆。")

    @mem.command("audit")
    async def mem_audit(self, event: AstrMessageEvent):
        """查看被污染过滤器拦下的内容。"""
        rows = self.engine.audit(10)
        if not rows:
            yield event.plain_result("隔离区是空的。")
            return
        lines = [f"#{r['id']} {r['text'][:60]}" for r in rows]
        yield event.plain_result("🛡 隔离区：\n" + "\n".join(lines))

    @mem.command("purge")
    async def mem_purge(self, event: AstrMessageEvent):
        """重扫已有记忆，把事后看出的污染条目移出召回池。"""
        moved = self.engine.purge()
        yield event.plain_result(f"已隔离 {moved} 条污染记忆。")

    @mem.command("eval")
    async def mem_eval(self, event: AstrMessageEvent, mode: str = ""):
        """跑金标准：/mem eval 或 /mem eval seed 重建测试集"""
        metrics, _ = self.engine.evaluate(refresh=(mode.strip() == "seed"))
        data = metrics.to_dict()
        text = (
            "📏 评估（{} 条）\n"
            "P@1 {:.3f}｜P@3 {:.3f}｜R@3 {:.3f}｜MRR {:.3f}\n"
            "误召回率 {:.3f}｜门控通过率 {:.3f}｜P50 {:.1f}ms"
        ).format(
            data["cases"],
            data["p@1"],
            data["p@3"],
            data["r@3"],
            data["mrr"],
            data["no_answer_fpr"],
            data["gate_pass_rate"],
            data["latency_ms"],
        )
        if metrics.failures:
            worst = metrics.failures[:3]
            text += "\n样例失败：" + "；".join(f"{f['query'][:16]}({f['kind']})" for f in worst)
        yield event.plain_result(text)

    @mem.command("tune")
    async def mem_tune(self, event: AstrMessageEvent, rest: str = ""):
        """带门禁的调参：/mem tune min_score=0.3 margin=0.02 [apply]"""
        parts = rest.split()
        patch: dict[str, float | int | str] = {}
        apply_change = False
        for part in parts:
            if part == "apply":
                apply_change = True
                continue
            if "=" not in part:
                continue
            key, value = part.split("=", 1)
            if key not in {"min_score", "margin", "max_items", "scope", "recency_half_life_days"}:
                continue
            try:
                patch[key] = float(value) if key != "scope" else value
            except ValueError:
                continue
        if not patch:
            yield event.plain_result("用法：/mem tune min_score=0.30 margin=0.03 [apply]")
            return

        comparison = self.engine.tune(patch, apply_change=apply_change)
        delta = comparison.to_dict()
        lines = [
            f"基准 P@1 {delta['base']['p@1']:.3f} → 候选 {delta['candidate']['p@1']:.3f}",
            f"基准 FPR {delta['base']['no_answer_fpr']:.3f} → 候选 {delta['candidate']['no_answer_fpr']:.3f}",
        ]
        if comparison.accepted:
            lines.append("✅ 通过回归门禁" + ("，已应用。" if apply_change else "（未应用，加 apply 生效）"))
        else:
            lines.append("❌ 拒绝应用：" + "；".join(comparison.reasons))
        yield event.plain_result("\n".join(lines))

    @mem.command("help")
    async def mem_help(self, event: AstrMessageEvent):
        """列出所有子指令。"""
        yield event.plain_result(
            "MemGuard 指令：\n"
            "/mem status 规模与参数\n"
            "/mem why 上一次为什么召回\n"
            "/mem search <查询> 调试检索\n"
            "/mem add <内容> 手动记忆\n"
            "/mem forget <编号> 删除记忆\n"
            "/mem audit 查看隔离区\n"
            "/mem purge 重扫污染\n"
            "/mem eval [seed] 跑评估\n"
            "/mem tune key=value [apply] 带门禁调参"
        )

    # ------------------------------------------------------------ utils

    def _session(self, event: AstrMessageEvent) -> str:
        return (
            getattr(event, "unified_msg_origin", None)
            or getattr(event, "session_id", None)
            or "default"
        )

    def _capture_once(self, session: str, message: str) -> None:
        """同一会话同一句话只采集一次，避免多步请求重复写入。"""
        if not message:
            return
        digest = hashlib.sha1(message.encode("utf-8")).hexdigest()[:16]
        last = self._recent.get(session)
        now = time.time()
        if last and last[0] == digest and now - last[1] < 120:
            return
        self._recent[session] = (digest, now)
        result = self.engine.capture(session, "user", message)
        if result.get("quarantined") and self.engine.settings.debug:
            logger.info("[memguard] 已隔离污染文本（score=%.2f）", result.get("score", 0.0))

    async def terminate(self):
        """插件停用时关闭数据库。"""
        try:
            self.store.close()
        except Exception:
            pass
