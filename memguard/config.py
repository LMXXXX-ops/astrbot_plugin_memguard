"""插件配置：从 AstrBot 的 _conf_schema 字典安全装配，缺字段用默认值。"""

from __future__ import annotations

from dataclasses import dataclass, fields


@dataclass
class Settings:
    enabled: bool = True
    auto_capture: bool = True
    max_items: int = 4
    min_score: float = 0.34
    margin: float = 0.04
    contamination_threshold: float = 0.55
    scope: str = "global"  # global=跨会话共享；session=只在当前会话生效
    recency_half_life_days: float = 120.0
    importance_weight: float = 0.15
    inject_header: str = "以下是与当前对话相关的长期记忆。只在确实相关时使用；不确定就不要提及。"
    debug: bool = False

    @classmethod
    def from_conf(cls, conf: dict | None) -> "Settings":
        conf = conf or {}
        allowed = {f.name: f.type for f in fields(cls)}
        kwargs = {}
        for key, value in conf.items():
            if key not in allowed:
                continue
            default = getattr(cls(), key)
            try:
                if isinstance(default, bool):
                    kwargs[key] = bool(value)
                elif isinstance(default, int):
                    kwargs[key] = int(value)
                elif isinstance(default, float):
                    kwargs[key] = float(value)
                else:
                    kwargs[key] = str(value)
            except (TypeError, ValueError):
                continue

        settings = cls(**kwargs)
        settings.max_items = max(1, min(8, settings.max_items))
        settings.min_score = max(0.0, min(1.0, settings.min_score))
        settings.margin = max(0.0, min(0.5, settings.margin))
        settings.contamination_threshold = max(0.1, min(1.0, settings.contamination_threshold))
        if settings.scope not in ("global", "session"):
            settings.scope = "global"
        return settings
