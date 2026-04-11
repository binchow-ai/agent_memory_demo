# -*- coding: utf-8 -*-
"""固定场景 NPC：稳定 agent_id、展示信息与系统提示。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set

from config import GENERIC_NPC_SYSTEM_PROMPT_TEMPLATE, KATHERYNE_SYSTEM_PROMPT


@dataclass(frozen=True)
class NpcPersona:
    id: str
    name_zh: str
    name_en: str
    mbti: str
    traits: str

    def label_display(self) -> str:
        return f"{self.name_zh} ({self.name_en})"


# 顺序即前端下拉默认顺序（首项为默认 NPC）
NPC_PERSONAS: tuple[NpcPersona, ...] = (
    NpcPersona(
        id="katheryne",
        name_zh="凯瑟琳",
        name_en="Katheryne",
        mbti="ESTP",
        traits="冒险精神、领导力、现实主义",
    ),
    NpcPersona(
        id="beidou",
        name_zh="北斗",
        name_en="Beidou",
        mbti="ESTP",
        traits="豪爽直率、无畏勇气、义气为先",
    ),
    NpcPersona(
        id="albedo",
        name_zh="阿贝多",
        name_en="Albedo",
        mbti="INTP",
        traits="冷静分析、探究本质、内省",
    ),
    NpcPersona(
        id="zhongli",
        name_zh="钟离",
        name_en="Zhongli",
        mbti="ISTJ",
        traits="重视契约、知识渊博、传统感",
    ),
    NpcPersona(
        id="hu_tao",
        name_zh="胡桃",
        name_en="Hu Tao",
        mbti="ENFP",
        traits="古灵精怪、情感导向、不可预测",
    ),
    NpcPersona(
        id="ayaka",
        name_zh="神里绫华",
        name_en="Ayaka",
        mbti="INFJ",
        traits="理想主义、完美倾向、温婉",
    ),
)

FIXED_NPC_IDS: Set[str] = {p.id for p in NPC_PERSONAS}
DEFAULT_NPC_ID: str = NPC_PERSONAS[0].id

_BY_ID: Dict[str, NpcPersona] = {p.id: p for p in NPC_PERSONAS}


def normalize_npc_id(npc_id: Optional[str]) -> str:
    """仅将空值转为默认 NPC；非空时调用方须已通过 validate_npc_id_for_chat。"""
    s = (npc_id or "").strip()
    return DEFAULT_NPC_ID if not s else s


def validate_npc_id_for_chat(npc_id: Optional[str]) -> None:
    """非空且不在白名单时抛出 ValueError（由 main 转为 HTTP 400）。"""
    s = (npc_id or "").strip()
    if s and s not in FIXED_NPC_IDS:
        raise ValueError(
            "npc_id 必须是以下之一: " + ", ".join(sorted(FIXED_NPC_IDS))
        )


def get_npc_persona(npc_id: str) -> NpcPersona:
    return _BY_ID.get(npc_id) or _BY_ID[DEFAULT_NPC_ID]


def get_npc_system_prompt(npc_id: str) -> str:
    p = get_npc_persona(npc_id)
    if p.id == "katheryne":
        return KATHERYNE_SYSTEM_PROMPT
    return GENERIC_NPC_SYSTEM_PROMPT_TEMPLATE.format(
        name_zh=p.name_zh, name_en=p.name_en, traits=p.traits
    )


def list_npc_public_info() -> List[Dict[str, Any]]:
    """供 GET /npcs 与前端展示。"""
    return [
        {
            "id": p.id,
            "name_zh": p.name_zh,
            "name_en": p.name_en,
            "label": p.label_display(),
            "mbti": p.mbti,
            "traits": p.traits,
        }
        for p in NPC_PERSONAS
    ]
