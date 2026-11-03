"""文本处理：CJK 分词、停用词、BM25 所需的词项统计。

不依赖任何第三方库，便于在 AstrBot 之外单独测试。
"""

from __future__ import annotations

import re
from collections import Counter
from typing import Iterable

_CJK = r"\u4e00-\u9fff\u3400-\u4dbf"
_CJK_RUN = re.compile(f"[{_CJK}]+")
_LATIN = re.compile(r"[a-zA-Z][a-zA-Z0-9_'-]*")
_NUM = re.compile(r"\d{2,}")

STOPWORDS = {
    "的", "了", "是", "在", "我", "你", "他", "她", "它", "们", "这", "那", "有", "和",
    "就", "都", "也", "还", "吗", "呢", "吧", "啊", "呀", "哦", "嗯", "什么", "怎么",
    "一个", "一下", "可以", "没有", "不是", "这个", "那个", "现在", "然后", "因为",
    "所以", "但是", "如果", "已经", "自己", "觉得", "知道", "时候",
}

# 元话语：当用户谈论「记忆/检索/提示词/插件」本身时，这些词说明他在讨论系统，
# 而不是在陈述自己的事实。用于抑制「提高命中率」被当成用户偏好召回。
META_TERMS = {
    "记忆", "召回", "命中率", "检索", "向量", "嵌入", "embedding", "prompt", "提示词",
    "系统提示", "上下文", "插件", "接口", "api", "日志", "调试", "测试", "评估",
    "评测", "准确率", "精确率", "算法", "模型", "参数", "配置", "部署", "报错",
    "异常", "函数", "代码", "脚本", "数据库", "sql", "json", "token", "上下文窗口",
}

# 技术/调试语境词：出现在消息里时，整段更可能是开发对话而非生活自述。
TECH_TERMS = {
    "报错", "异常", "堆栈", "traceback", "error", "exception", "接口", "请求", "响应",
    "状态码", "超时", "部署", "容器", "docker", "日志", "调试", "单元测试", "用例",
    "函数", "变量", "类", "方法", "返回值", "依赖", "版本", "commit", "分支", "合并",
    "构建", "编译", "重启", "配置项", "字段", "表结构", "迁移",
}


def normalize(text: str) -> str:
    """统一空白与全角标点，降低匹配噪声。"""
    text = (text or "").replace("\u3000", " ").replace("\r", " ").replace("\n", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def tokenize(text: str) -> list[str]:
    """CJK 用二元组、拉丁用整词，数字保留。

    单个汉字成段时保留单字，避免「猫」这类短词被整体丢掉。
    """
    text = normalize(text).lower()
    tokens: list[str] = []

    for run in _CJK_RUN.findall(text):
        if len(run) == 1:
            tokens.append(run)
        else:
            tokens.extend(run[i : i + 2] for i in range(len(run) - 1))

    tokens.extend(_LATIN.findall(text))
    tokens.extend(_NUM.findall(text))

    return [t for t in tokens if t not in STOPWORDS and len(t) <= 24]


def term_freq(tokens: Iterable[str]) -> Counter:
    return Counter(tokens)


def contains_any(text: str, terms: Iterable[str]) -> list[str]:
    """返回命中的词，便于解释判定原因。"""
    low = normalize(text).lower()
    return [t for t in terms if t.lower() in low]


def jaccard(a: str, b: str) -> float:
    sa, sb = set(tokenize(a)), set(tokenize(b))
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)
