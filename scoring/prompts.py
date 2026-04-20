"""Prompt loading and construction for P1–P7 pipelines."""

import os
import yaml

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

_CACHE: dict | None = None


def _load() -> dict:
    global _CACHE
    if _CACHE is None:
        path = os.path.join(BASE_DIR, "prompts", "bcs_prompts.yaml")
        with open(path, "r", encoding="utf-8") as f:
            _CACHE = yaml.safe_load(f)
    return _CACHE


# ── P1: Direct Integer ──────────────────────────────────────────────

def p1_prompts() -> tuple[str, str]:
    p = _load()
    return p["system_prompt_integer"].strip(), p["user_prompt_integer"].strip()


# ── P2: JSON Mode (full BCS scale + breed list) ─────────────────────

def p2_prompts() -> tuple[str, str]:
    p = _load()
    system = "\n\n".join([
        p["role"].strip(),
        p["bcs_scale"].strip(),
        p["confidence_guide"].strip(),
        p["breed_ids"].strip(),
        p["p2_json_instruction"].strip(),
    ])
    user = p["user_msg_finetune"].strip()
    return system, user


# ── P3: Reasoning Mode ──────────────────────────────────────────────

def p3_prompts() -> tuple[str, str]:
    p = _load()
    return (p["system_prompt_reasoning"].strip(),
            p["user_prompt_reasoning"].strip())


# ── P5: AAV1 – Verifier ─────────────────────────────────────────────

def p5_verifier_prompts(scorer_bcs: int, scorer_reasoning: str) -> tuple[str, str]:
    p = _load()
    sys = p["p5_verifier_system"].strip()
    usr = p["p5_verifier_user"].strip().format(
        scorer_bcs=scorer_bcs, scorer_reasoning=scorer_reasoning)
    return sys, usr


# ── P6: Visual Few-Shot ──────────────────────────────────────────────

def p6_vfewshot_prompts() -> tuple[str, str]:
    p = _load()
    return (p["p6_vfewshot_system"].strip(),
            p["p6_vfewshot_user"].strip())


# ── P7: Debate ───────────────────────────────────────────────────────

def p7_debate_prompts(my_bcs: int, my_reasoning: str,
                      other_bcs: int, other_reasoning: str) -> tuple[str, str]:
    p = _load()
    sys = p["p7_debate_system"].strip()
    usr = p["p7_debate_user"].strip().format(
        my_bcs=my_bcs, my_reasoning=my_reasoning,
        other_bcs=other_bcs, other_reasoning=other_reasoning)
    return sys, usr


# ── P8: Visual Few-Shot (cat + dog reference, reasoning-first JSON) ──

def p8_vfewshot_prompts() -> tuple[str, str]:
    p = _load()
    return (p["p8_vfewshot_system"].strip(),
            p["p8_vfewshot_user"].strip())
