from __future__ import annotations

import json
import hashlib
import re
import time
from copy import deepcopy
from typing import Any

import requests

from protocol import ASSESSMENT_LABELS, parse_json_response
from src.config import QWEN_API_KEY, QWEN_BASE_URL, QWEN_MODEL


BOUNDARY_STATEMENT = (
    "This report audits support within the supplied evidence and does not establish diagnosis, "
    "treatment efficacy, prescription appropriateness, or clinical decision benefit."
)

GENERATION_ROLES = {
    "structured_fields": "deterministic_python",
    "assessment_summary": "qwen_verbalization_of_structured_audit",
}

SYSTEM_PROMPT = """You verbalize a structured evidence audit that has already been calculated by code.

Return exactly one JSON object with one field:
{
  "assessment_summary": "one concise English sentence"
}

Rules:
1. Use only structured_audit.narrative_facts. Do not inspect, infer, recalculate, or modify any audit field.
2. Begin the sentence with the exact assessment_label phrase supplied in narrative_facts, followed by a colon.
3. Do not write any number, percentage, fraction, count, identifier, herb name, formula name, syndrome name, source ID, target, or pathway.
4. Include every supplied required_clause exactly as written. You may only add basic connecting words and punctuation.
5. Do not add medical knowledge, recommendations, diagnostic claims, efficacy claims, or clinical judgments.
"""


def build_user_prompt(context: dict[str, Any]) -> str:
    audit = context.get("structured_audit") or {}
    payload = {
        "structured_audit": {
            "narrative_facts": audit.get("narrative_facts"),
        },
    }
    return json.dumps(payload, ensure_ascii=False, indent=2)


def assemble_response(model_response: dict[str, Any], context: dict[str, Any]) -> dict[str, Any]:
    audit = deepcopy(context.get("structured_audit") or {})
    return {
        "audit_scope": audit.get("audit_scope"),
        "consistency_audit_applicable": audit.get("consistency_audit_applicable"),
        "evidence_claims": audit.get("evidence_claims", []),
        "unverified_parametric_claims": [],
        "missing_evidence_items": audit.get("missing_evidence_items", []),
        "coverage_metrics": audit.get("coverage_metrics", {}),
        "assessment_level": audit.get("assessment_level"),
        "assessment_label": audit.get("assessment_label"),
        "assessment_rule_trace": audit.get("assessment_rule_trace"),
        "formula_syndrome_relation": audit.get("formula_syndrome_relation"),
        "assessment_summary": model_response.get("assessment_summary"),
        "boundary_statement": BOUNDARY_STATEMENT,
        "generation_roles": deepcopy(GENERATION_ROLES),
    }


class FairQwenClient:
    def __init__(self) -> None:
        self.api_key = QWEN_API_KEY.strip()
        self.base_url = QWEN_BASE_URL.rstrip("/")
        self.model = QWEN_MODEL

    def request_once(self, context: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
        if not self.api_key:
            raise RuntimeError("QWEN_API_KEY is not configured in the project .env")
        user_prompt = build_user_prompt(context)
        response = requests.post(
            f"{self.base_url}/chat/completions",
            headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
            json={
                "model": self.model,
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                "temperature": 0.0,
                "response_format": {"type": "json_object"},
            },
            timeout=180,
        )
        response.raise_for_status()
        payload = response.json()
        content = payload["choices"][0]["message"]["content"]
        parsed = parse_json_response(content)
        return parsed, {
            "model": payload.get("model", self.model),
            "request_id": response.headers.get("x-request-id") or payload.get("id"),
            "http_status": response.status_code,
            "usage": payload.get("usage"),
            "system_prompt_sha256": hashlib.sha256(SYSTEM_PROMPT.encode("utf-8")).hexdigest(),
            "user_prompt_sha256": hashlib.sha256(user_prompt.encode("utf-8")).hexdigest(),
        }

    def request_validated(
        self, context: dict[str, Any], *, max_attempts: int = 3
    ) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
        failures: list[str] = []
        for attempt in range(1, max_attempts + 1):
            try:
                raw_response, metadata = self.request_once(context)
                response = assemble_response(raw_response, context)
                validation = validate_response(
                    response, context, raw_model_response=raw_response
                )
                if not validation["valid"]:
                    raise ValueError(
                        "generated narrative failed validation: "
                        + "; ".join(validation["errors"] + validation["provenance_violations"])
                    )
                metadata["attempt"] = attempt
                metadata["prior_failures"] = failures
                metadata["model_output_fields"] = sorted(raw_response)
                return response, metadata, validation
            except (requests.RequestException, KeyError, IndexError, TypeError, ValueError, RuntimeError) as exc:
                failures.append(f"{type(exc).__name__}: {exc}")
                if attempt == max_attempts:
                    raise RuntimeError(
                        f"Qwen request failed validation after {max_attempts} attempts: {failures[-1]}"
                    ) from exc
                time.sleep(2 ** (attempt - 1))
        raise AssertionError("unreachable")


def allowed_source_ids(context: dict[str, Any]) -> set[str]:
    return {
        str(source_id)
        for record in context.get("evidence_context", [])
        for source_id in (record.get("source_ids") or [])
        if source_id
    }


def validate_response(
    response: dict[str, Any],
    context: dict[str, Any],
    *,
    raw_model_response: dict[str, Any] | None = None,
) -> dict[str, Any]:
    errors: list[str] = []
    warnings: list[str] = []
    provenance_violations: list[str] = []
    required = {
        "audit_scope",
        "consistency_audit_applicable",
        "evidence_claims",
        "unverified_parametric_claims",
        "missing_evidence_items",
        "coverage_metrics",
        "assessment_level",
        "assessment_label",
        "assessment_rule_trace",
        "formula_syndrome_relation",
        "assessment_summary",
        "boundary_statement",
        "generation_roles",
    }
    missing = sorted(required - set(response))
    if missing:
        errors.append(f"missing required fields: {missing}")

    if raw_model_response is not None and set(raw_model_response) != {"assessment_summary"}:
        errors.append("model may output only assessment_summary")

    audit = context.get("structured_audit")
    if not isinstance(audit, dict):
        errors.append("structured_audit is missing from context")
        audit = {}
    exact_fields = (
        "audit_scope",
        "consistency_audit_applicable",
        "evidence_claims",
        "missing_evidence_items",
        "coverage_metrics",
        "assessment_level",
        "assessment_label",
        "assessment_rule_trace",
        "formula_syndrome_relation",
    )
    for field in exact_fields:
        if response.get(field) != audit.get(field):
            errors.append(f"{field} differs from deterministic structured_audit")
    if response.get("unverified_parametric_claims") != []:
        errors.append("unverified_parametric_claims must be empty in narration-only mode")
    if response.get("generation_roles") != GENERATION_ROLES:
        errors.append("generation_roles does not match the protocol")

    claims = response.get("evidence_claims")
    if not isinstance(claims, list):
        errors.append("evidence_claims must be a list")
        claims = []
    records_by_id = {
        str(record.get("item_id")): record
        for record in context.get("evidence_context", [])
        if record.get("item_id")
    }
    visible_items = set(records_by_id)
    valid_sources = allowed_source_ids(context)
    for index, claim in enumerate(claims):
        if not isinstance(claim, dict):
            errors.append(f"evidence_claims[{index}] is not an object")
            continue
        item_id = str(claim.get("item_id") or "")
        if item_id not in visible_items and not item_id.startswith("relation:"):
            provenance_violations.append(
                f"claim item_id was not visible in evidence_context: {item_id}"
            )
        source_ids = claim.get("source_ids")
        if not isinstance(source_ids, list) or not source_ids:
            provenance_violations.append(f"claim {item_id} has no source identifier")
            continue
        unknown = sorted(set(source_ids) - valid_sources)
        if unknown:
            provenance_violations.append(f"claim {item_id} cites unavailable sources: {unknown}")
        claim_record = records_by_id.get(item_id)
        if claim_record is None and item_id.startswith("relation:"):
            parts = item_id.split(":")
            formula_id = parts[1] if len(parts) >= 3 else ""
            claim_record = records_by_id.get(f"formula:{formula_id}")
        record_sources = set(str(value) for value in ((claim_record or {}).get("source_ids") or []))
        mismatched = sorted(set(source_ids) - record_sources)
        if mismatched:
            provenance_violations.append(
                f"claim {item_id} cites sources from another evidence record: {mismatched}"
            )

    summary = response.get("assessment_summary")
    if not isinstance(summary, str) or not summary.strip():
        errors.append("assessment_summary must be a non-empty string")
    else:
        if any(character.isnumeric() for character in summary) or any(
            marker in summary for marker in ("%", "％")
        ):
            errors.append("assessment_summary must not contain model-generated numbers")
        number_words = (
            "zero",
            "one",
            "two",
            "three",
            "four",
            "five",
            "six",
            "seven",
            "eight",
            "nine",
            "ten",
            "eleven",
            "twelve",
            "thirteen",
            "fourteen",
            "fifteen",
            "sixteen",
            "seventeen",
            "eighteen",
            "nineteen",
            "twenty",
            "hundred",
        )
        if re.search(r"\b(?:" + "|".join(number_words) + r")\b", summary.casefold()):
            errors.append("assessment_summary must not spell out model-generated numbers")
        expected_phrase = str(
            (audit.get("narrative_facts") or {}).get("assessment_label") or ""
        ).replace("_", " ")
        if expected_phrase and not summary.strip().casefold().startswith(
            f"{expected_phrase.casefold()}:"
        ):
            errors.append("assessment_summary does not begin with the deterministic assessment label")
        for label in ASSESSMENT_LABELS.values():
            phrase = label.replace("_", " ").casefold()
            if label != audit.get("assessment_label") and phrase in summary.casefold():
                errors.append("assessment_summary contains a conflicting assessment label")
                break
        required_clauses = (audit.get("narrative_facts") or {}).get("required_clauses") or []
        for clause in required_clauses:
            if str(clause).casefold() not in summary.casefold():
                errors.append(f"assessment_summary omits required clause: {clause}")
        remainder = summary.casefold()
        if expected_phrase:
            remainder = remainder.replace(expected_phrase.casefold(), " ", 1)
        for clause in required_clauses:
            remainder = remainder.replace(str(clause).casefold(), " ", 1)
        remaining_words = set(re.findall(r"[a-z]+", remainder))
        allowed_connectors = {"and", "but", "while", "however"}
        unexpected_words = sorted(remaining_words - allowed_connectors)
        if unexpected_words:
            errors.append(
                "assessment_summary adds text outside deterministic clauses: "
                + ", ".join(unexpected_words)
            )
        forbidden_claim_terms = (
            "diagnos",
            "efficacy",
            "effective",
            "appropriate",
            "recommend",
            "clinical benefit",
            "treatment decision",
        )
        if any(term in summary.casefold() for term in forbidden_claim_terms):
            errors.append("assessment_summary adds a prohibited clinical claim")

    if response.get("boundary_statement") != BOUNDARY_STATEMENT:
        errors.append("boundary_statement does not match the protocol")

    semantic_error_markers = (
        "differs from deterministic",
        "assessment_summary",
        "prohibited clinical claim",
    )
    return {
        "valid": not errors and not provenance_violations,
        "schema_valid": not errors,
        "structured_audit_consistent": not any("differs from deterministic" in value for value in errors),
        "semantic_consistent": not any(
            marker in value for value in errors for marker in semantic_error_markers
        ),
        "narrative_numbers_absent": not any("model-generated numbers" in value for value in errors),
        "errors": errors,
        "warnings": warnings,
        "provenance_violations": provenance_violations,
        "provenance_clean": not provenance_violations,
        "visible_item_count": len(visible_items),
        "allowed_source_id_count": len(valid_sources),
    }
