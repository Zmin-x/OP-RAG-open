from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PUBLIC_DATA_DIR = PROJECT_ROOT / "data" / "kb"
# The public demo is the default. The paper audit used a separate private,
# case-informed knowledge-base extension and must be selected explicitly.
DATA_DIR = Path(os.getenv("OP_RAG_DATA_DIR", str(PUBLIC_DATA_DIR))).resolve()
ENV_PATH = PROJECT_ROOT / ".env"

load_dotenv(ENV_PATH, override=True)


QWEN_API_KEY = os.getenv("QWEN_API_KEY", os.getenv("DASHSCOPE_API_KEY", "")).strip()
QWEN_BASE_URL = os.getenv(
    "QWEN_BASE_URL",
    "https://dashscope.aliyuncs.com/compatible-mode/v1",
).rstrip("/")
QWEN_MODEL = os.getenv("QWEN_MODEL", "qwen-plus")

DEFAULT_TOP_K_SYNDROMES = int(os.getenv("TOP_K_SYNDROMES", "2"))
DEFAULT_TOP_K_FORMULAS = int(os.getenv("TOP_K_FORMULAS", "2"))

# 方名别名映射：临床简写、加减写法 → formulas.json 标准方名
# 用于医生方剂映射时做名称归一，不改知识库原始内容
FORMULA_ALIAS_MAP: dict[str, str] = {
    "桃红四物汤": "桃红四物汤 / 桃红四物汤加减",
    "桃红四物汤加减": "桃红四物汤 / 桃红四物汤加减",
    "桃红四物加减": "桃红四物汤 / 桃红四物汤加减",
    "桃红四物": "桃红四物汤 / 桃红四物汤加减",
    "桃红四物方": "桃红四物汤 / 桃红四物汤加减",
    "桃红四物汤加味": "桃红四物汤 / 桃红四物汤加减",
    "加味桃红四物汤": "桃红四物汤 / 桃红四物汤加减",
    "桃红四物汤化裁": "桃红四物汤 / 桃红四物汤加减",
    "桃红四物汤加减方": "桃红四物汤 / 桃红四物汤加减",
    "桃红四物汤合方": "桃红四物汤 / 桃红四物汤加减",
    "四物汤合桃红": "桃红四物汤 / 桃红四物汤加减",
    "四物汤加桃仁红花": "桃红四物汤 / 桃红四物汤加减",
}

# 药名别名映射：炮制名、产地名 → herbs.json 标准名
# 用于机制层查找时做名称归一，不改知识库原始内容
HERB_ALIAS_MAP: dict[str, str] = {
    "川断": "续断",
    "熟地": "熟地黄",
    "生地": "生地黄",
    "生黄芪": "黄芪",
    "川杜仲": "杜仲",
    "广陈皮": "陈皮",
    "广皮": "陈皮",
    "云苓": "茯苓",
    "白茯苓": "茯苓",
    "仙灵脾": "淫羊藿",
    "山萸肉": "山茱萸",
    "狗嵴": "狗脊",
    "续断": "续断",
    "怀牛膝": "牛膝",
    "川牛膝": "牛膝",
    "炙甘草": "甘草",
    "麸炒甘草": "甘草",
    "制附子": "附子",
    "淡附子": "附子",
    "醋没药": "没药",
    "炒没药": "没药",
    "醋乳香": "乳香",
    "炒乳香": "乳香",
    "麸炒白术": "白术",
    "炒白术": "白术",
    "酒当归": "当归",
    "全当归": "当归",
    "酒川芎": "川芎",
    "盐杜仲": "杜仲",
    "炒杜仲": "杜仲",
    "盐续断": "续断",
    "酒续断": "续断",
    "盐菟丝子": "菟丝子",
    "炒菟丝子": "菟丝子",
    "炒山药": "山药",
    "麸炒山药": "山药",
    "酒萸肉": "山茱萸",
    "萸肉": "山茱萸",
    "盐骨碎补": "骨碎补",
    "烫骨碎补": "骨碎补",
    "蒸淫羊藿": "淫羊藿",
    "炙淫羊藿": "淫羊藿",
    "盐知母": "知母",
    "炙黄芪": "黄芪",
    "生黄芪": "黄芪",
    "酒丹参": "丹参",
    "炒丹参": "丹参",
    "炒薏苡仁": "薏苡仁",
    "麸炒薏苡仁": "薏苡仁",
    "熟地黄": "熟地黄",
    "生地黄": "地黄",
    "酒白芍": "白芍",
    "炒白芍": "白芍",
    "炒枳壳": "枳壳",
    "麸炒枳壳": "枳壳",
    "云木香": "木香",
    "广木香": "木香",
    "土木香": "木香",
    "元胡": "延胡索",
    "玄胡": "延胡索",
    "延胡": "延胡索",
    "川柏": "黄柏",
    "生白术": "白术",
    "生白芍": "白芍",
    "生牡蛎": "牡蛎",
    "生龙骨": "龙骨",
    "生龙齿": "龙齿",
    "炙自然铜": "自然铜",
    "生地黄": "生地黄",
    "熟地黄": "熟地黄",
    "川续断": "续断",
    "寄生": "桑寄生",
    "全当归": "当归",
}
