from __future__ import annotations

import json
import re
from typing import Any

import requests

from .config import QWEN_API_KEY, QWEN_BASE_URL, QWEN_MODEL
from .prompt import PILOT_SECTION_KEYS, build_system_prompt, build_user_prompt


class QwenRemoteError(RuntimeError):
    """Raised when an explicit remote Qwen request does not succeed."""


def _strip_code_fence(text: str) -> str:
    stripped = text.strip()
    if stripped.startswith("```") and stripped.endswith("```"):
        stripped = re.sub(r"^```(?:json)?\s*", "", stripped, flags=re.IGNORECASE)
        stripped = re.sub(r"\s*```$", "", stripped)
    return stripped.strip()


def validate_pilot_response(
    response: Any,
    *,
    expected_level: int | None,
    allowed_source_ids: set[str],
    required_unretrieved_herbs: set[str] | None = None,
    formula_has_indication_overlap: bool | None = None,
) -> dict[str, Any]:
    errors: list[str] = []
    warnings: list[str] = []
    level_alignment = "not_applicable" if expected_level is None else "not_evaluable"
    if not isinstance(response, dict):
        return {"valid": False, "errors": ["response is not a JSON object"]}
    required_top_level = {"sections", "cited_source_ids", "missing_evidence", "boundary_statement"}
    if expected_level is not None:
        required_top_level.add("assessment_level")
    missing_top_level = sorted(required_top_level - set(response))
    if missing_top_level:
        errors.append(f"required top-level JSON keys are missing: {missing_top_level}")
    unexpected_top_level = sorted(set(response) - required_top_level)
    if unexpected_top_level:
        warnings.append(f"unexpected top-level JSON keys: {unexpected_top_level}")
    sections = response.get("sections")
    if not isinstance(sections, dict):
        errors.append("sections must be a JSON object")
    else:
        missing_sections = sorted(set(PILOT_SECTION_KEYS) - set(sections))
        if missing_sections:
            errors.append(f"required sections are missing: {missing_sections}")
        extra_sections = sorted(set(sections) - set(PILOT_SECTION_KEYS))
        if extra_sections:
            warnings.append(f"unexpected sections: {extra_sections}")
        blank_sections = [key for key in PILOT_SECTION_KEYS if key in sections and (not isinstance(sections[key], str) or not sections[key].strip())]
        if blank_sections:
            errors.append(f"required sections must contain non-empty strings: {blank_sections}")
    level = response.get("assessment_level")
    if expected_level is None:
        if "assessment_level" in response:
            warnings.append("assessment_level was ignored because this mode has no four-level rule evaluation")
    elif not isinstance(level, int) or level not in {1, 2, 3, 4}:
        errors.append("assessment_level is unavailable or invalid for g4")
    else:
        level_alignment = "matches" if level == expected_level else "disagrees"
        if level_alignment == "disagrees":
            errors.append(
                f"assessment_level disagrees with the deterministic g4 rule: expected {expected_level}, received {level}"
            )
    if level == 4 and formula_has_indication_overlap is True:
        errors.append("level 4 conflicts with the retrieved formula-level syndrome overlap")
    cited_ids = response.get("cited_source_ids")
    if not isinstance(cited_ids, list) or any(not isinstance(item, str) for item in cited_ids):
        errors.append("cited_source_ids must be a list of strings")
    else:
        unknown_ids = sorted(set(cited_ids) - allowed_source_ids)
        if unknown_ids:
            errors.append(f"cited_source_ids include identifiers outside retrieved evidence: {unknown_ids}")
    missing_evidence = response.get("missing_evidence")
    if not isinstance(missing_evidence, list) or any(not isinstance(item, str) for item in missing_evidence):
        errors.append("missing_evidence must be a list of strings")
    elif required_unretrieved_herbs:
        section_values = [str(value) for value in sections.values()] if isinstance(sections, dict) else []
        disclosure_text = "\n".join([*section_values, *missing_evidence])
        undisclosed = sorted(herb for herb in required_unretrieved_herbs if herb not in disclosure_text)
        if undisclosed:
            warnings.append(f"unretrieved herb evidence is not disclosed: {undisclosed}")
    boundary_statement = response.get("boundary_statement")
    if not isinstance(boundary_statement, str) or not boundary_statement.strip():
        errors.append("boundary_statement must be a non-empty string")
    prohibited_patterns = (
        r"proves? clinical efficacy",
        r"treatment is effective",
        r"diagnos(?:e|ed|is) .* patient",
        r"recommend(?:s|ed)? (?:a |the )?(?:treatment|prescription|medication)",
        r"确诊",
        r"诊断准确",
        r"临床疗效显著",
        r"治疗有效",
        r"推荐(?:用药|处方|治疗)",
        r"建议(?:用药|处方|治疗)",
    )
    serialized = json.dumps(response, ensure_ascii=False)
    matched = [pattern for pattern in prohibited_patterns if re.search(pattern, serialized, flags=re.IGNORECASE)]
    if matched:
        errors.append(f"prohibited clinical claim pattern(s): {matched}")
    return {
        "valid": not errors,
        "errors": errors,
        "warnings": warnings,
        "expected_assessment_level": expected_level,
        "reported_assessment_level": level,
        "assessment_level_alignment": level_alignment,
        "allowed_source_id_count": len(allowed_source_ids),
        "cited_source_id_count": len(cited_ids) if isinstance(cited_ids, list) else None,
        "required_unretrieved_herb_count": len(required_unretrieved_herbs or set()),
        "formula_has_indication_overlap": formula_has_indication_overlap,
    }


class QwenClient:
    def __init__(self, api_key: str | None = None, base_url: str | None = None, model: str | None = None) -> None:
        self.api_key = (api_key or QWEN_API_KEY).strip()
        self.base_url = (base_url or QWEN_BASE_URL).rstrip("/")
        self.model = model or QWEN_MODEL

    @property
    def is_configured(self) -> bool:
        return bool(self.api_key)

    @property
    def chat_completions_url(self) -> str:
        return f"{self.base_url}/chat/completions"

    def generate(self, context: dict[str, Any]) -> str:
        if not self.api_key:
            return build_local_report(context, reason="qwen_api_key_missing")
        try:
            return self._request_remote_text(context)["content"]
        except QwenRemoteError:
            return build_local_report(context, reason="qwen_request_failed")

    def request_json_once(self, context: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
        """Make exactly one HTTP request and parse its JSON content; never fall back or retry."""
        request_result = self._request_remote_text(context)
        try:
            parsed = json.loads(_strip_code_fence(request_result["content"]))
        except json.JSONDecodeError as exc:
            raise QwenRemoteError(f"remote response is not valid JSON: {exc.msg}") from exc
        return parsed, request_result

    def _request_remote_text(self, context: dict[str, Any]) -> dict[str, Any]:
        if not self.api_key:
            raise QwenRemoteError("QWEN_API_KEY is not configured")
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": build_system_prompt(context.get("mode"))},
                {"role": "user", "content": build_user_prompt(context)},
            ],
            "temperature": 0.0,
        }
        try:
            response = requests.post(
                self.chat_completions_url,
                headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
                json=payload,
                timeout=120,
            )
            response.raise_for_status()
            data = response.json()
            content = data["choices"][0]["message"]["content"]
        except (requests.RequestException, KeyError, IndexError, TypeError, ValueError) as exc:
            raise QwenRemoteError(f"remote Qwen request failed: {type(exc).__name__}") from exc
        if not isinstance(content, str) or not content.strip():
            raise QwenRemoteError("remote Qwen response contained no text content")
        return {
            "content": content,
            "model": data.get("model", self.model),
            "http_status": response.status_code,
            "request_id": response.headers.get("x-request-id") or data.get("id"),
        }


def build_local_report(context: dict[str, Any], reason: str) -> str:
    plan = context.get("physician_plan", {})
    evidence = context.get("rag_evidence", {})
    primary = plan.get("primary_syndrome", {})
    formula = plan.get("formula", {})
    return "\n".join(
        [
            "Local deterministic evidence report",
            f"Reason: {reason}",
            f"Physician primary syndrome: {primary.get('std_name') or primary.get('raw_name') or 'not provided'}",
            f"Physician formula: {formula.get('std_name') or formula.get('raw_name') or 'not provided'}",
            f"Syndrome evidence records: {len(evidence.get('syndrome_evidence', []))}",
            f"Formula evidence records: {len(evidence.get('formula_evidence', []))}",
            f"Herb evidence records: {len(evidence.get('herb_mechanism_evidence', []))}",
            "Research evidence-traceability summary only.",
        ]
    )
