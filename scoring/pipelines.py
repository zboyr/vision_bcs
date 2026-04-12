"""Pipeline implementations P1–P7.

Each pipeline function has the signature::

    fn(client, model, image_path, *, max_retries=3, delay=1.0) -> dict

The returned dict always contains ``bcs`` (int | None) and ``error`` (str | None).
Additional keys are pipeline-specific and will be persisted in the JSONL log.
"""

import os
import time
from collections import Counter
from typing import Any, Dict

from .checkpoint import get_log_dir, get_run_path, load_completed
from .client import call_llm, call_llm_raw, build_image_part
from .parsers import parse_integer, parse_json_bcs, extract_bcs
from .prompts import (
    p1_prompts, p2_prompts, p3_prompts,
    p5_verifier_prompts, p6_vfewshot_prompts,
    p7_debate_prompts,
)

# Set by run_experiment.py before scoring starts; used by P5/P7 to reuse data.
_LOG_DIR: str | None = None
_CURRENT_MODEL_ID: str | None = None


def set_reuse_context(log_dir: str, model_id: str) -> None:
    global _LOG_DIR, _CURRENT_MODEL_ID
    _LOG_DIR, _CURRENT_MODEL_ID = log_dir, model_id


def _load_existing_results(prompt_id: str) -> Dict[int, Dict[str, Any]]:
    """Load completed results for the current model + given prompt from logs."""
    if not _LOG_DIR or not _CURRENT_MODEL_ID:
        return {}
    cell_dir = get_log_dir(_LOG_DIR, _CURRENT_MODEL_ID, prompt_id)
    run_path = get_run_path(cell_dir, 1)
    return load_completed(run_path)

# ── Registry ─────────────────────────────────────────────────────────

PIPELINES: Dict[str, Any] = {}

PIPELINE_LABELS: Dict[str, str] = {
    "P1": "P1_direct",
    "P2": "P2_json",
    "P3": "P3_reasoning",
    "P4": "P4_bo5",
    "P5": "P5_aav1",
    "P6": "P6_vfewshot",
    "P7": "P7_debate",
}


def _register(name: str):
    def decorator(fn):
        PIPELINES[name] = fn
        return fn
    return decorator


def get_pipeline(name: str):
    if name not in PIPELINES:
        raise ValueError(
            f"Unknown pipeline '{name}'. Available: {list(PIPELINES.keys())}")
    return PIPELINES[name]


# ── Helpers ──────────────────────────────────────────────────────────

def _fail(error: str, **extra) -> Dict[str, Any]:
    return {"bcs": None, "error": error, **extra}


def _ok(bcs: int, **extra) -> Dict[str, Any]:
    return {"bcs": bcs, "error": None, **extra}


# ── P1: Direct Integer ──────────────────────────────────────────────

@_register("P1")
def p1_direct(client, model, image_path, *, max_retries=3, delay=1.0, **_kw):
    sys_p, usr_p = p1_prompts()
    content, err = call_llm(client, model, sys_p, usr_p, image_path,
                            max_retries=max_retries, temperature=0.1)
    if err:
        return _fail(err, raw="")
    bcs = parse_integer(content)
    if bcs is None:
        return _fail(f"parse_fail: {content[:100]}", raw=content)
    return _ok(bcs, raw=content)


# ── P2: JSON Mode ───────────────────────────────────────────────────

@_register("P2")
def p2_json(client, model, image_path, *, max_retries=3, delay=1.0, **_kw):
    sys_p, usr_p = p2_prompts()
    content, err = call_llm(client, model, sys_p, usr_p, image_path,
                            max_retries=max_retries, temperature=0.1)
    if err:
        return _fail(err, raw="")
    result = parse_json_bcs(content)
    if result:
        return _ok(result["bcs"], reasoning=result.get("reasoning", ""),
                   confidence=result.get("confidence"), raw=content)
    bcs = parse_integer(content)
    if bcs is not None:
        return _ok(bcs, raw=content)
    return _fail(f"parse_fail: {content[:100]}", raw=content)


# ── P3: Reasoning Mode ──────────────────────────────────────────────

@_register("P3")
def p3_reasoning(client, model, image_path, *, max_retries=3, delay=1.0, **_kw):
    sys_p, usr_p = p3_prompts()
    content, err = call_llm(client, model, sys_p, usr_p, image_path,
                            max_retries=max_retries, temperature=0.1)
    if err:
        return _fail(err, raw="")
    result = parse_json_bcs(content)
    if result:
        return _ok(result["bcs"], reasoning=result.get("reasoning", ""),
                   raw=content)
    bcs = parse_integer(content)
    if bcs is not None:
        return _ok(bcs, raw=content)
    return _fail(f"parse_fail: {content[:100]}", raw=content)


# ── P4: Best-of-5 Majority Vote ─────────────────────────────────────

@_register("P4")
def p4_bo5(client, model, image_path, *, max_retries=3, delay=1.0, **_kw):
    sys_p, usr_p = p1_prompts()
    votes, raws, errors = [], [], []

    for i in range(5):
        if i > 0:
            time.sleep(delay * 0.3)
        content, err = call_llm(client, model, sys_p, usr_p, image_path,
                                max_retries=max_retries, temperature=0.7)
        if err:
            errors.append(f"call{i+1}: {err}")
            continue
        raws.append(content)
        bcs = parse_integer(content)
        if bcs is not None:
            votes.append(bcs)
        else:
            errors.append(f"parse{i+1}: {content[:50]}")

    if not votes:
        return _fail("; ".join(errors), votes=[], raws=raws, vote_errors=errors)

    majority = Counter(votes).most_common(1)[0][0]
    return _ok(majority, votes=votes, raws=raws,
               vote_errors=errors if errors else None)


# ── P5: Agent-as-a-Verifier v1 (Scorer → Verifier) ──────────────────

@_register("P5")
def p5_aav1(client, model, image_path, *, max_retries=3, delay=1.0,
            _image_id: int | None = None):
    # Stage 1: Try reusing P2 result for scorer
    scorer = None
    if _image_id is not None:
        p2_data = _load_existing_results("P2")
        cached = p2_data.get(_image_id)
        if cached and cached.get("bcs") is not None:
            scorer = {"bcs": int(cached["bcs"]),
                      "reasoning": cached.get("reasoning", "N/A"),
                      "raw": cached.get("raw", "")}

    if scorer is None:
        scorer = p2_json(client, model, image_path,
                         max_retries=max_retries, delay=delay)

    if scorer["bcs"] is None:
        return _fail(f"scorer: {scorer.get('error')}", stage="scorer",
                     scorer_raw=scorer.get("raw", ""))

    scorer_bcs = scorer["bcs"]
    scorer_reasoning = scorer.get("reasoning", "N/A")

    # Stage 2: Verifier
    sys_p, usr_p = p5_verifier_prompts(scorer_bcs, scorer_reasoning)
    content, err = call_llm(client, model, sys_p, usr_p, image_path,
                            max_retries=max_retries, temperature=0.1)
    if err:
        # Verifier failed → fall back to scorer
        return _ok(scorer_bcs, scorer_bcs=scorer_bcs, verifier_bcs=None,
                   verifier_error=err, raw=scorer.get("raw", ""))

    verifier_bcs = extract_bcs(content)
    if verifier_bcs is None:
        verifier_bcs = scorer_bcs

    return _ok(verifier_bcs, scorer_bcs=scorer_bcs,
               scorer_reasoning=scorer_reasoning,
               verifier_bcs=verifier_bcs, raw=content)


# ── P6: Visual Few-Shot (reference chart + target image) ─────────────

@_register("P6")
def p6_vfewshot(client, model, image_path, *, max_retries=3, delay=1.0,
                reference_images=None, reference_image=None, **kwargs):
    """P6: Send BCS reference chart(s) alongside the target image.

    Accepts either ``reference_images`` (list) or ``reference_image`` (str).
    """
    sys_p, usr_p = p6_vfewshot_prompts()

    # Normalize reference image list
    if reference_images is None:
        if reference_image is None:
            reference_images = ["prompts/cat_bcs.jpg"]
        elif isinstance(reference_image, str):
            reference_images = [reference_image]
        else:
            reference_images = list(reference_image)

    content_parts = [{"type": "text", "text": usr_p}]
    for ref_path in reference_images:
        ref_part, ref_err = build_image_part(ref_path)
        if ref_err:
            return _fail(f"reference_image '{ref_path}': {ref_err}", raw="")
        content_parts.append(ref_part)

    tgt_part, tgt_err = build_image_part(image_path)
    if tgt_err:
        return _fail(f"target_image: {tgt_err}", raw="")
    content_parts.append(tgt_part)

    messages = [
        {"role": "system", "content": sys_p},
        {"role": "user", "content": content_parts},
    ]

    content, err = call_llm_raw(client, model, messages,
                                max_retries=max_retries, temperature=0.1)
    if err:
        return _fail(err, raw="")

    result = parse_json_bcs(content)
    if result:
        return _ok(result["bcs"], reasoning=result.get("reasoning", ""),
                   raw=content)
    bcs = parse_integer(content)
    if bcs is not None:
        return _ok(bcs, raw=content)
    return _fail(f"parse_fail: {content[:100]}", raw=content)


# ── P7: Debate v1 (A + B → debate round if disagree → average) ──────

@_register("P7")
def p7_debate(client, model, image_path, *, max_retries=3, delay=1.0,
              _image_id: int | None = None):
    # Stage 1: Agent A — reuse P3 result if available
    agent_a = None
    if _image_id is not None:
        p3_data = _load_existing_results("P3")
        cached = p3_data.get(_image_id)
        if cached and cached.get("bcs") is not None:
            agent_a = {"bcs": int(cached["bcs"]),
                       "reasoning": cached.get("reasoning", ""),
                       "raw": cached.get("raw", "")}

    if agent_a is None:
        agent_a = p3_reasoning(client, model, image_path,
                               max_retries=max_retries, delay=delay)

    if agent_a["bcs"] is None:
        return _fail(f"agent_a: {agent_a.get('error')}", stage="agent_a",
                     raw=agent_a.get("raw", ""))

    time.sleep(delay * 0.5)

    # Stage 2: Agent B (P3, higher temperature for diversity)
    sys_p, usr_p = p3_prompts()
    content_b, err_b = call_llm(client, model, sys_p, usr_p, image_path,
                                max_retries=max_retries, temperature=0.5)
    if err_b:
        return _ok(agent_a["bcs"], agent_a_bcs=agent_a["bcs"],
                   agent_b_error=err_b, debate=False,
                   raw=agent_a.get("raw", ""))

    result_b = parse_json_bcs(content_b)
    if result_b:
        b_bcs, b_reason = result_b["bcs"], result_b.get("reasoning", "")
    else:
        b_bcs = parse_integer(content_b)
        b_reason = content_b[:200]

    if b_bcs is None:
        return _ok(agent_a["bcs"], agent_a_bcs=agent_a["bcs"],
                   agent_b_parse_fail=content_b[:100], debate=False,
                   raw=content_b)

    a_bcs = agent_a["bcs"]
    a_reason = agent_a.get("reasoning", "")

    # If they agree, done
    if a_bcs == b_bcs:
        return _ok(a_bcs, agent_a_bcs=a_bcs, agent_b_bcs=b_bcs,
                   debate=False, raw=content_b)

    time.sleep(delay * 0.5)

    # Debate: A reconsiders
    sys_a, usr_a = p7_debate_prompts(a_bcs, a_reason, b_bcs, b_reason)
    c_a, e_a = call_llm(client, model, sys_a, usr_a, image_path,
                        max_retries=max_retries, temperature=0.1)

    time.sleep(delay * 0.3)

    # Debate: B reconsiders
    sys_b, usr_b = p7_debate_prompts(b_bcs, b_reason, a_bcs, a_reason)
    c_b, e_b = call_llm(client, model, sys_b, usr_b, image_path,
                        max_retries=max_retries, temperature=0.1)

    final_a = (extract_bcs(c_a) if c_a and not e_a else None) or a_bcs
    final_b = (extract_bcs(c_b) if c_b and not e_b else None) or b_bcs

    final_bcs = round((final_a + final_b) / 2)

    return _ok(final_bcs,
               agent_a_bcs=a_bcs, agent_b_bcs=b_bcs,
               debate_a_bcs=final_a, debate_b_bcs=final_b,
               debate=True, raw=f"A:{c_a}\nB:{c_b}")
