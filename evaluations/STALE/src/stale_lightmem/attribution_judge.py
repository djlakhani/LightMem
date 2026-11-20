from __future__ import annotations

import json
import os
import time
from typing import Any

from openai import OpenAI

from .attribution import DIMENSIONS, parse_json_object
from .config import JudgeConfig

LABEL_SYSTEM_PROMPT = """
You are labeling ground-truth evidence for a memory attribution experiment.

For M_old and M_new, you receive memory chunks attributed to the source message
that contained that state. Decide which chunks actually preserve the material
state expressed by M_old or M_new.

Select a chunk only when its POST-RECONCILIATION text explicitly states, clearly
paraphrases, or necessarily entails the target state. Exclude chunks that merely
ask a related question, give incidental context, discuss a downstream task, or
share vocabulary without preserving the state. Do not infer missing facts.

Classify every candidate exactly once. Return JSON:
{
  "old": {
    "decisions": [
      {"id": "...", "matches": true, "reasoning": "..."}
    ]
  },
  "new": {
    "decisions": [
      {"id": "...", "matches": true, "reasoning": "..."}
    ]
  }
}
"""


EVALUATION_SYSTEM_PROMPT = """
You are a strict evaluator for the STALE long-term-memory benchmark and its
evidence-attribution extension.

Grade answer correctness using these rules:
- SR / dim1: pass only if the answer recognizes that M_old may no longer be valid
  and respects M_new.
- PR / dim2: pass only if the answer resists the false M_old premise or explicitly
  accounts for M_new.
- IPA / dim3: pass only if the recommendation safely adheres to M_new rather than
  the obsolete constraints in M_old; generic or vague answers fail.

For evidence use, the input includes a mechanically verified UUID comparison.
You must preserve its `used_old_evidence` and `used_new_evidence` booleans exactly.
Explain what the cited old/new chunks imply, but do not reinterpret UUID equality.

Return JSON with dim1_eval, dim2_eval, and dim3_eval. Each must contain:
{
  "answer_pass": true,
  "answer_reasoning": "...",
  "old_evidence": {"used": true, "reasoning": "..."},
  "new_evidence": {"used": true, "reasoning": "..."},
  "citation_assessment": "..."
}
"""


def _usage(response: Any) -> dict[str, int]:
    usage = getattr(response, "usage", None)
    return {
        "prompt_tokens": int(getattr(usage, "prompt_tokens", 0) or 0),
        "completion_tokens": int(getattr(usage, "completion_tokens", 0) or 0),
        "total_tokens": int(getattr(usage, "total_tokens", 0) or 0),
    }


def _unique_candidates(discovery: dict[str, Any]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    seen: set[str] = set()
    for item in discovery.get("eligible_candidates", []):
        memory_id = str(item["id"])
        if memory_id not in seen:
            seen.add(memory_id)
            result.append(item)
    return result


def _label_user_prompt(
    scenario: dict[str, Any], discoveries: dict[str, dict[str, Any]]
) -> str:
    sections: list[str] = [
        "[Ground truth]",
        f"M_old: {scenario['M_old']}",
        f"M_new: {scenario['M_new']}",
        f"Hidden logic: {scenario['explanation']}",
    ]
    for key, title in (("old", "M_old candidates"), ("new", "M_new candidates")):
        sections.extend(["", f"[{title}]"])
        candidates = _unique_candidates(discoveries[key])
        if not candidates:
            sections.append("(No retrievable source-attributed candidates.)")
        for item in candidates:
            sections.extend(
                [
                    f"ID: {item['id']}",
                    f"Post-reconciliation memory: {item['post_memory']}",
                    f"Original extracted fact: {item['extracted_fact']}",
                    "",
                ]
            )
    return "\n".join(sections).rstrip()


def _parse_labels(
    content: str, discoveries: dict[str, dict[str, Any]]
) -> dict[str, Any]:
    value = parse_json_object(content)
    result: dict[str, Any] = {}
    for key in ("old", "new"):
        expected = {item["id"] for item in _unique_candidates(discoveries[key])}
        section = value.get(key)
        if not isinstance(section, dict) or not isinstance(
            section.get("decisions"), list
        ):
            raise ValueError(f"Evidence labeler is missing {key}.decisions")
        decisions = section["decisions"]
        observed: set[str] = set()
        normalized: list[dict[str, Any]] = []
        for decision in decisions:
            if not isinstance(decision, dict):
                raise ValueError(f"{key} decision must be an object")
            memory_id = decision.get("id")
            matches = decision.get("matches")
            reasoning = decision.get("reasoning")
            if memory_id not in expected or memory_id in observed:
                raise ValueError(
                    f"{key} decision has invalid/duplicate ID: {memory_id}"
                )
            if not isinstance(matches, bool) or not isinstance(reasoning, str):
                raise ValueError(f"{key} decision has invalid fields for {memory_id}")
            observed.add(memory_id)
            normalized.append(
                {"id": memory_id, "matches": matches, "reasoning": reasoning}
            )
        if observed != expected:
            raise ValueError(
                f"{key} labeler did not classify every candidate; "
                f"missing={sorted(expected - observed)}"
            )
        result[key] = {
            "decisions": normalized,
            "matching_ids": sorted(
                item["id"] for item in normalized if item["matches"]
            ),
            "excluded_ids": sorted(
                item["id"] for item in normalized if not item["matches"]
            ),
        }
    return result


def _evaluation_user_prompt(
    scenario: dict[str, Any],
    answers: dict[str, Any],
    ground_truth: dict[str, Any],
) -> str:
    sections = [
        "[Ground truth state]",
        f"M_old: {scenario['M_old']}",
        f"M_new: {scenario['M_new']}",
        f"Hidden logic: {scenario['explanation']}",
        f"Verified M_old IDs: {json.dumps(ground_truth['old']['matching_ids'])}",
        f"Verified M_new IDs: {json.dumps(ground_truth['new']['matching_ids'])}",
    ]
    for dim_key, short_name, _ in DIMENSIONS:
        answer = answers[dim_key]
        metrics = answer["citation_metrics"]
        selected_text = {
            item["id"]: item.get("payload", {}).get("memory", "")
            for item in answer["retrieval"]
            if item["id"] in set(answer["memory_ids_used"])
        }
        sections.extend(
            [
                "",
                f"[{dim_key} / {short_name}]",
                f"Question: {answer['query']}",
                f"Target answer: {answer['answer']}",
                f"Cited IDs in importance order: {json.dumps(answer['memory_ids_used'])}",
                f"Cited memory text: {json.dumps(selected_text, ensure_ascii=False)}",
                "Mechanically verified citation comparison:",
                json.dumps(metrics, ensure_ascii=False, sort_keys=True),
            ]
        )
    return "\n".join(sections)


def _parse_evaluation(content: str, answers: dict[str, Any]) -> dict[str, Any]:
    value = parse_json_object(content)
    result: dict[str, Any] = {}
    for dim_key, _, _ in DIMENSIONS:
        item = value.get(f"{dim_key}_eval")
        if not isinstance(item, dict):
            raise ValueError(f"Judge response is missing {dim_key}_eval")
        if not isinstance(item.get("answer_pass"), bool) or not isinstance(
            item.get("answer_reasoning"), str
        ):
            raise ValueError(f"Judge response has invalid answer fields for {dim_key}")
        for evidence_key, metric_key in (
            ("old_evidence", "used_old_evidence"),
            ("new_evidence", "used_new_evidence"),
        ):
            evidence = item.get(evidence_key)
            if not isinstance(evidence, dict):
                raise ValueError(f"Judge response is missing {dim_key}.{evidence_key}")
            expected = answers[dim_key]["citation_metrics"][metric_key]
            if evidence.get("used") is not expected:
                raise ValueError(
                    f"Judge changed verified {metric_key} for {dim_key}: "
                    f"expected {expected}, got {evidence.get('used')}"
                )
            if not isinstance(evidence.get("reasoning"), str):
                raise ValueError(
                    f"Judge response needs reasoning for {dim_key}.{evidence_key}"
                )
        if not isinstance(item.get("citation_assessment"), str):
            raise ValueError(f"Judge response needs citation_assessment for {dim_key}")
        result[f"{dim_key}_eval"] = item
    return result


class AttributionJudge:
    def __init__(self, config: JudgeConfig):
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY is required for gpt-6-luna judging")
        self.config = config
        self.client = OpenAI(api_key=api_key, base_url=os.getenv("OPENAI_BASE_URL"))

    def _call(
        self,
        system: str,
        user: str,
        parser: Any,
    ) -> dict[str, Any]:
        last_error: Exception | None = None
        for attempt in range(1, self.config.max_retries + 1):
            started = time.perf_counter()
            try:
                response = self.client.chat.completions.create(
                    model=self.config.model,
                    messages=[
                        {"role": "system", "content": system},
                        {"role": "user", "content": user},
                    ],
                    response_format={"type": "json_object"},
                )
                content = response.choices[0].message.content or ""
                parsed = parser(content)
                return {
                    "result": parsed,
                    "raw_response": content,
                    "elapsed_seconds": time.perf_counter() - started,
                    "usage": _usage(response),
                    "attempts": attempt,
                    "model": self.config.model,
                    "reasoning_enabled": self.config.reasoning_enabled,
                    "prompt": {"system": system, "user": user},
                }
            except Exception as exc:
                last_error = exc
                if attempt < self.config.max_retries:
                    time.sleep(2 ** (attempt - 1))
        raise RuntimeError(
            f"Attribution judge failed after {self.config.max_retries} attempts"
        ) from last_error

    def label(
        self,
        scenario: dict[str, Any],
        discoveries: dict[str, dict[str, Any]],
    ) -> dict[str, Any]:
        user = _label_user_prompt(scenario, discoveries)
        return self._call(
            LABEL_SYSTEM_PROMPT,
            user,
            lambda content: _parse_labels(content, discoveries),
        )

    def evaluate(
        self,
        scenario: dict[str, Any],
        answers: dict[str, Any],
        ground_truth: dict[str, Any],
    ) -> dict[str, Any]:
        user = _evaluation_user_prompt(scenario, answers, ground_truth)
        return self._call(
            EVALUATION_SYSTEM_PROMPT,
            user,
            lambda content: _parse_evaluation(content, answers),
        )
