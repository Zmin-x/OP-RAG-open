from __future__ import annotations

import hashlib
import re
from pathlib import Path


HERE = Path(__file__).resolve().parent
SOURCE = (
    HERE
    / "outputs"
    / "api_spot_check_gate_20260810"
    / "post_formal_run_audit"
    / "FORMAL_CASE_018_WALKTHROUGH.md"
)
OUTPUT = SOURCE.with_name("FORMAL_CASE_018_WALKTHROUGH_中文版.md")
SOURCE_SHA256 = "249b575b611efff7d16b3d2093c4480cd58d30d0924ad27a75861ad740fb221c"


KEYS = {
    "case_id": "病例编号",
    "source_group": "病例来源组",
    "source_cluster_id": "来源文献簇编号",
    "excluded_source_ids": "本病例排除的来源编号",
    "expected_retrieval_item_ids": "预期应检索到的项目编号",
    "expected_claims": "预期证据声明",
    "item_id": "项目编号",
    "layer": "所属层级",
    "expected_status": "预期证据状态",
    "source_ids": "来源编号",
    "core_herbs": "核心药材",
    "formula_composition_herbs": "标准方剂组成药材",
    "primary_syndrome_name": "主要证型名称",
    "secondary_syndrome_names": "次要证型名称",
    "formula_name": "方剂名称",
    "herbs": "医师处方药材",
    "name": "名称",
    "retrieval_score": "检索相似度",
    "structured_audit": "结构化审计",
    "narrative_facts": "供千问转述的定性事实",
    "assessment_label": "审计结论标签",
    "supported_layers": "已有来源证据的层级",
    "missing_evidence_present": "是否存在缺失证据",
    "explicit_contradiction_present": "是否存在明确矛盾",
    "required_clauses": "说明中必须包含的含义",
    "assessment_summary": "审计结论说明",
    "remote_metadata": "远程调用元数据",
    "model": "模型",
    "request_id": "请求编号",
    "http_status": "接口状态码",
    "usage": "用量",
    "total_tokens": "总词元数",
    "completion_tokens": "输出词元数",
    "prompt_tokens": "输入词元数",
    "prompt_tokens_details": "输入词元明细",
    "cached_tokens": "缓存词元数",
    "system_prompt_sha256": "系统提示词哈希",
    "user_prompt_sha256": "用户提示词哈希",
    "attempt": "本次尝试序号",
    "prior_failures": "此前失败记录",
    "model_output_fields": "模型实际输出字段",
    "validation": "校验结果",
    "valid": "总体是否有效",
    "schema_valid": "数据结构是否有效",
    "structured_audit_consistent": "结构化字段是否一致",
    "semantic_consistent": "语义是否一致",
    "narrative_numbers_absent": "模型说明是否不含数字",
    "errors": "错误",
    "warnings": "警告",
    "provenance_violations": "来源违规项",
    "provenance_clean": "来源是否合法",
    "visible_item_count": "模型上下文中可见项目数",
    "allowed_source_id_count": "允许使用的来源编号数",
    "context_sha256": "本配置上下文哈希",
    "evidence_claims": "证据声明",
    "unverified_parametric_claims": "未核验的参数性声明",
    "missing_evidence_items": "缺失证据项目",
    "coverage_metrics": "三种覆盖率",
    "physician_plan_herbs": "医师处方药材覆盖",
    "supported_items": "有机制证据的药材",
    "missing_items": "缺少机制证据的药材",
    "numerator": "分子",
    "denominator": "分母",
    "value": "计算值",
    "assessment_level": "审计等级",
    "assessment_rule_trace": "审计规则计算轨迹",
    "inputs": "规则输入",
    "contradiction": "是否矛盾",
    "strict_support": "是否达到严格支持",
    "syndrome_available": "证型层是否可用",
    "formula_mapped": "方剂名称是否已映射",
    "mechanism_available": "药材机制层是否可用",
    "syndrome_evidence_available": "主证型的来源证据是否可用",
    "formula_evidence_available": "目标方剂的独立来源证据是否可用",
    "mechanism_evidence_available": "药材机制来源证据是否可用",
    "triggered_rule": "触发的等级规则",
    "boundary_statement": "结论边界声明",
    "generation_roles": "字段生成责任",
    "structured_fields": "结构化字段",
    "support_status": "证据支持状态",
    "statement": "证据说明",
    "expected_missing": "参考中预期缺失项目",
    "reported_missing": "系统报告的缺失项目",
    "reported_claims": "系统报告的证据声明",
    "correct_claims": "正确证据声明数",
}


EXACT_VALUES = {
    "qwen_only": "仅千问",
    "flat_rag": "平面检索增强",
    "layered_rag": "分层检索增强",
    "op_rag": "完整OP-RAG",
    "literature_binli2": "文献病例组2",
    "formula": "方剂",
    "herb": "药材",
    "syndrome": "证型",
    "supported": "有证据支持",
    "insufficient current kb evidence": "当前知识库证据不足",
    "insufficient_current_kb_evidence": "当前知识库证据不足",
    "partial evidence support": "部分证据支持",
    "partial_evidence_support": "部分证据支持",
    "level_2": "第2级规则",
    "level_3": "第3级规则",
    "deterministic_python": "由Python确定性计算",
    "qwen_verbalization_of_structured_audit": "千问仅转述结构化审计结果",
    "assessment_summary": "审计结论说明",
}


PHRASES = {
    "no source-linked evidence layer is available": "目前没有任何层级具有可追溯的来源证据",
    "source-linked evidence is available for herb mechanism, syndrome": "药材机制层和证型层已有可追溯的来源证据",
    "some evidence remains missing": "仍有部分证据缺失",
    "no explicit contradiction is present": "未发现明确的跨层矛盾",
    "insufficient current kb evidence: no source-linked evidence layer is available, some evidence remains missing, and no explicit contradiction is present": "当前知识库证据不足：目前没有任何层级具有可追溯的来源证据，仍有部分证据缺失，且未发现明确的跨层矛盾",
    "insufficient current kb evidence: source-linked evidence is available for herb mechanism, syndrome, some evidence remains missing, and no explicit contradiction is present": "当前知识库证据不足：药材机制层和证型层已有可追溯的来源证据，但仍有部分证据缺失，且未发现明确的跨层矛盾",
    "partial evidence support: source-linked evidence is available for herb mechanism, syndrome, some evidence remains missing, and no explicit contradiction is present": "部分证据支持：药材机制层和证型层已有可追溯的来源证据，但仍有部分证据缺失，且未发现明确的跨层矛盾",
    "This report audits support within the supplied evidence and does not establish diagnosis, treatment efficacy, prescription appropriateness, or clinical decision benefit.": "本报告只审查所提供证据范围内的支持情况，不能用于证明诊断正确、治疗有效、处方适宜或能够改善临床决策。",
}


INTRO = """

> **阅读说明**：这是原始抽查报告的完整中文阅读版。所有病例内容、数值、来源、哈希和模型返回结果均来自正式实验，没有重新调用千问，也没有改动计算结果。为了便于阅读，下文把机器字段名和类别前缀翻译成中文，因此代码块改为“中文结构化记录”，不再是可直接执行的 JSON。原始机器字段仍保存在同目录的 `FORMAL_CASE_018_WALKTHROUGH.md` 中。

## 先用一句话看懂本例

第 18 例记录的是“肾阳虚证—右归饮加减—13 味药材”。仅千问配置没有知识库证据，因此只能报告当前知识库证据不足。平面检索只找回 10 条预期证据中的 5 条；分层检索找回了全部 10 条，但“右归饮加减”仍缺少排除本病例来源后的独立方剂证据，所以仍为第 3 级。完整 OP-RAG 在同一批检索证据上进一步完成方剂名称映射和跨层规则计算，因此归为第 2 级“部分证据支持”，但仍没有达到第 1 级严格支持。

## 四种配置的直观比较

| 配置 | 知识库输入 | 找回预期证据 | 医师处方药材机制覆盖 | 核心药材机制覆盖 | 方剂组成机制覆盖 | 审计等级 |
|---|---|---:|---:|---:|---:|---:|
| 仅千问 | 无 | 不适用 | 0/13 | 0/8 | 0/13 | 第3级 |
| 平面检索增强 | 有，未分层 | 5/10 | 4/13 | 3/8 | 4/13 | 第3级 |
| 分层检索增强 | 有，按层检索 | 10/10 | 9/13 | 5/8 | 9/13 | 第3级 |
| 完整OP-RAG | 分层检索并启用跨层规则 | 10/10 | 9/13 | 5/8 | 9/13 | 第2级 |

这里最重要的是：**“检索到证据”不等于“整条证型—方剂—药材链已经完整成立”**。本例中分层检索已经找全审计参考定义的 10 个预期检索项目，但方剂的独立来源证据和 4 味药材的机制证据仍缺失，所以不能判为完全支持。
"""


def translate(text: str) -> str:
    text = text.replace("# OP-RAG 正式运行后抽查：第 18 例", "# OP-RAG 正式运行后抽查：第18例完整中文阅读版" + INTRO)
    text = text.replace("```json", "```text")

    for key, chinese in KEYS.items():
        text = text.replace(f'"{key}":', f'"{chinese}":')

    for value, chinese in EXACT_VALUES.items():
        text = text.replace(f'"{value}"', f'"{chinese}"')

    for phrase, chinese in sorted(PHRASES.items(), key=lambda item: len(item[0]), reverse=True):
        text = text.replace(phrase, chinese)

    text = re.sub(
        r'"((?:syndrome|herb|formula):[^" ]+) is supported by the supplied source-linked evidence\."',
        lambda match: f'"{match.group(1)} 有所列来源证据支持。"',
        text,
    )

    prefix_replacements = {
        "syndrome:": "证型:",
        "formula:": "方剂:",
        "herb:": "药材:",
        "TCMSP_TARGETS:": "TCMSP靶点记录:",
        "OP_TARGET_INTERSECTION:": "骨质疏松交集靶点:",
        "GPROFILER_KEGG:": "KEGG富集结果:",
        "LITDOC:": "文献来源簇:",
    }
    for old, new in prefix_replacements.items():
        text = text.replace(old, new)
    text = re.sub(r"\bREF(\d{3})\b", r"参考文献\1", text)

    prose_replacements = {
        "`qwen_only`": "“仅千问”",
        "`flat_rag`": "“平面检索增强”",
        "`layered_rag`": "“分层检索增强”",
        "`op_rag`": "“完整OP-RAG”",
        "qwen_only": "仅千问",
        "flat_rag": "平面检索增强",
        "layered_rag": "分层检索增强",
        "op_rag": "完整OP-RAG",
        "physician-recorded plan": "医师记录的处方方案",
        "evidence claims": "证据声明",
        "missing items": "缺失证据项",
        "audit level": "审计等级",
        "narrative_facts": "定性事实",
        "assessment_summary": "审计结论说明",
        "source of truth": "唯一计算依据",
        "strict support": "严格支持",
        "syndrome、formula、herb 和 cross-layer claim": "证型层、方剂层、药材层和跨层证据声明",
        "syndrome、formula、mechanism": "证型层、方剂层、药材机制层",
        "coverage": "覆盖率",
        "precision": "精确率",
        "recall": "召回率",
        "summary": "说明",
        "Level": "第",
        "PASS": "通过",
        "True": "是",
        "False": "否",
        "N/A": "不适用",
    }
    for old, new in prose_replacements.items():
        text = text.replace(old, new)

    text = re.sub(r"\btrue\b", "是", text)
    text = re.sub(r"\bfalse\b", "否", text)
    text = re.sub(r"\bnull\b", "空", text)

    formula_replacements = {
        "`P_ret = |E_expected ∩ E_retrieved| / |E_retrieved|`": "`检索精确率 = |应检索证据集 ∩ 实际检索证据集| / |实际检索证据集|`",
        "`R_ret = |E_expected ∩ E_retrieved| / |E_expected|`": "`检索召回率 = |应检索证据集 ∩ 实际检索证据集| / |应检索证据集|`",
        "`R_claim = N_correct_claims / N_expected_claims`": "`输出证据召回率 = 正确证据声明数 / 预期证据声明数`",
        "`P_link = N_correct_claims / N_reported_claims`": "`来源链接精确率 = 正确证据声明数 / 已报告证据声明数`",
        "`C = |H_supported| / |H_supported ∪ H_missing|`": "`药材覆盖率 = |有机制证据的药材| / |全部目标药材|`",
        "`P_m = TP_m/N_reported_missing`": "`缺失证据精确率 = 正确缺失项数 / 系统报告的缺失项数`",
        "`R_m = TP_m/N_expected_missing`": "`缺失证据召回率 = 正确缺失项数 / 参考中预期的缺失项数`",
        "`F1_m = 2P_mR_m/(P_m+R_m)`": "`缺失证据F1 = 2 × 精确率 × 召回率 /（精确率 + 召回率）`",
        "`physician_plan_herbs`": "“医师处方药材覆盖”",
        "`core_herbs`": "“核心药材覆盖”",
        "`formula_composition_herbs`": "“标准方剂组成药材覆盖”",
    }
    for old, new in formula_replacements.items():
        text = text.replace(old, new)

    text = text.replace("## 配置：仅千问", "## 配置一：仅千问")
    text = text.replace("## 配置：平面检索增强", "## 配置二：平面检索增强")
    text = text.replace("## 配置：分层检索增强", "## 配置三：分层检索增强")
    text = text.replace("## 配置：完整OP-RAG", "## 配置四：完整OP-RAG")
    text = text.replace("Level `", "第`")
    text = text.replace("Level ", "第")
    text = text.replace("可直接执行的 JSON", "可直接执行的机器数据格式")
    text = re.sub(r"第\s+([1-4])\s+级", r"第\1级", text)
    text = re.sub(r"第\s+([1-4])(?=[；。])", r"第\1级", text)
    text = re.sub(r"独立重算\s+第\s*`([1-4])`", r"独立重算为第\1级", text)
    text = re.sub(r"存储值\s+第\s*`([1-4])`", r"存储值为第\1级", text)
    natural_chinese = {
        "输出中 证型层": "输出中证型层",
        "跨层证据声明 的总条数": "跨层证据声明的总条数",
        "即 覆盖率 的分子": "即覆盖率的分子",
        "矛盾优先为 第": "矛盾优先为第",
        "否则 严格支持 为": "否则，达到严格支持时为",
        "否则 证型层": "否则，证型层",
        "时为 第": "时为第",
        "层 均可用": "层均可用",
        "生成 证据声明": "生成证据声明",
        "缺失证据项 和 审计等级": "缺失证据项和审计等级",
        "三种 覆盖率": "三种覆盖率",
        "缺失证据 精确率": "缺失证据精确率",
        "缺失证据 召回率": "缺失证据召回率",
        "定性 说明": "定性说明",
        "与 Python 唯一计算依据 完全一致": "与Python唯一计算依据完全一致",
    }
    for old, new in natural_chinese.items():
        text = text.replace(old, new)
    text = text.replace(
        "### 千问实际输入",
        "### 千问实际输入的中文对应内容（原始载荷见机器报告）",
    )
    text = text.replace(
        "### 千问实际输出（模型唯一允许字段）",
        "### 千问实际输出的中文翻译（模型唯一允许字段）",
    )
    text = text.replace(
        "### Python组装的完整输出",
        "### Python组装结果的完整中文对应内容",
    )
    return text


def main() -> None:
    source_bytes = SOURCE.read_bytes()
    actual_source_hash = hashlib.sha256(source_bytes).hexdigest()
    if actual_source_hash != SOURCE_SHA256:
        raise RuntimeError(
            "The authoritative case-018 source changed; review it before regenerating "
            f"the Chinese reader ({actual_source_hash})."
        )
    source = source_bytes.decode("utf-8").replace("\r\n", "\n").replace("\r", "\n")
    translated = translate(source)
    for key in KEYS:
        if f'"{key}":' in translated:
            raise RuntimeError(f"Untranslated machine field remains: {key}")
    critical_identifiers = (
        "case_018",
        "719068d6-0a31-9e91-a04d-3f60d87a9cb3",
        "3736a251-46bb-92b3-b295-18ffb360b480",
        "05889d8b-eb83-9e47-94db-341cc836a5d7",
        "96362147-f43c-929c-ace4-15729d6b3a52",
        "e33581f0a6c21362d2e3120d540511fac39284119484d904c0aa5ea995719dd1",
        "ccd4f724104856fd1b15a047b6ce8fef52e47079a043f49ec9c0a23c9c4b3578",
        "2a563e86db8ba9d04551bca19e62292c5c11e38d4d4cf7123c247767bc7933b1",
        "7603a4694a850e52e675fd243496a8bf3de6e1365aa9d4ca15c3d09bbf772e39",
    )
    for identifier in critical_identifiers:
        if identifier not in translated:
            raise RuntimeError(f"Critical formal-run identifier is missing: {identifier}")
    required_reading_markers = (
        "平面检索只找回 10 条预期证据中的 5 条",
        "分层检索找回了全部 10 条",
        "归为第2级“部分证据支持”",
        "千问仅转述结构化审计结果",
        "最终判定：通过",
    )
    for marker in required_reading_markers:
        if marker not in translated:
            raise RuntimeError(f"Required Chinese explanation is missing: {marker}")
    OUTPUT.write_text(translated, encoding="utf-8", newline="\n")
    print(f"{OUTPUT}\nsource_sha256={actual_source_hash}\nstatus=pass")


if __name__ == "__main__":
    main()
