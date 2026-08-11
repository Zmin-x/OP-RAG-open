from __future__ import annotations

import html
import json
import os
import socket
import sys
import traceback
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs

from .ablation_runner import AblationCaseInput, AblationRunner
from .config import QWEN_BASE_URL, QWEN_MODEL
from .loader import load_kb
from .llm_client import QwenClient
from .pipeline_trace import configure_pipeline_trace
from .rag_pipeline import OPRagPipeline


DEFAULT_HOST = os.getenv("WEB_HOST", "0.0.0.0")
DEFAULT_PORT = int(os.getenv("WEB_PORT", "8000"))
EXAMPLE_QUERY = (
    "患者女，80岁，因腰背部疼痛伴活动受限1周入院。"
    "1周前无明显诱因出现腰背部疼痛，呈持续性胀痛，坐位、翻身、起床时加重，舌淡红，苔薄白，脉弦。"
    "MRI示T12椎体新鲜压缩性骨折。中医辨证为血瘀气滞证。"
)


HTML_PAGE = """<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>OP-RAG 科研原型</title>
  <style>
    body {{ margin: 0; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", "Microsoft YaHei", sans-serif; background: #f6f7f9; color: #202124; }}
    main {{ max-width: 1080px; margin: 0 auto; padding: 32px 20px 56px; }}
    .card {{ background: #fff; border: 1px solid #e6e8eb; border-radius: 16px; box-shadow: 0 10px 30px rgba(0,0,0,0.04); padding: 24px; margin-bottom: 18px; }}
    h1 {{ margin: 0 0 8px; font-size: 26px; line-height: 1.35; }}
    h2 {{ margin: 0 0 14px; font-size: 18px; }}
    h3 {{ margin: 0 0 8px; font-size: 15px; color: #344054; }}
    .muted {{ color: #667085; line-height: 1.6; }}
    .hint {{ background: #f9fafb; border: 1px solid #eaecf0; border-radius: 10px; padding: 12px 14px; margin: 12px 0 0; font-size: 14px; color: #475467; line-height: 1.6; }}
    .warn {{ background: #fffaeb; border: 1px solid #fedf89; border-radius: 10px; padding: 12px 14px; margin-top: 12px; font-size: 14px; color: #93370d; line-height: 1.6; }}
    .disclaimer {{ background: #f8f9fc; border: 1px solid #d0d5dd; border-radius: 10px; padding: 12px 14px; font-size: 13px; color: #667085; line-height: 1.6; }}
    textarea {{ width: 100%; min-height: 138px; box-sizing: border-box; border: 1px solid #d0d5dd; border-radius: 12px; padding: 14px; resize: vertical; font-size: 15px; line-height: 1.6; outline: none; }}
    textarea:focus {{ border-color: #5b7cfa; box-shadow: 0 0 0 3px rgba(91,124,250,0.12); }}
    .row {{ display: flex; gap: 10px; align-items: center; flex-wrap: wrap; margin-top: 14px; }}
    .sample-row {{ margin-top: 10px; }}
    button {{ border: 0; border-radius: 10px; background: #355cff; color: white; padding: 11px 18px; font-weight: 700; cursor: pointer; font-size: 14px; }}
    button.secondary {{ background: #eef2ff; color: #344054; font-weight: 600; padding: 8px 14px; }}
    button:disabled {{ cursor: not-allowed; opacity: 0.65; }}
    .pill {{ display: inline-block; padding: 4px 10px; border-radius: 999px; background: #eef2ff; color: #3447a8; font-size: 13px; margin-right: 6px; margin-bottom: 6px; }}
    .summary-grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 14px; }}
    @media (max-width: 720px) {{ .summary-grid {{ grid-template-columns: 1fr; }} }}
    .summary-block {{ background: #f9fafb; border: 1px solid #eaecf0; border-radius: 12px; padding: 14px; }}
    .summary-block ul {{ margin: 8px 0 0; padding-left: 18px; line-height: 1.7; }}
    .summary-block li {{ margin-bottom: 4px; }}
    .score-note {{ font-size: 13px; color: #98a2b3; margin-top: 8px; }}
    pre {{ white-space: pre-wrap; word-break: break-word; background: #101828; color: #f8fafc; border-radius: 14px; padding: 18px; line-height: 1.7; overflow-x: auto; margin: 0; }}
    .error {{ background: #fff3f3; color: #b42318; border: 1px solid #ffd6d6; border-radius: 12px; padding: 12px; margin-top: 14px; display: none; }}
    .ok {{ background: #ecfdf3; color: #027a48; border: 1px solid #abefc6; }}
  </style>
</head>
<body>
  <main>
    <section class="card">
      <h1>原发性骨质疏松中医证型-方剂-机制解释 RAG 原型系统</h1>
      <p class="muted">适用于<strong>已确诊原发性骨质疏松</strong>患者的症状、舌脉信息输入。系统执行 L1 证型检索、L2 证型约束下方剂检索、L3 药味机制匹配和一致性校验，再调用千问 API 生成结构化解释。</p>
      <div style="margin-top:12px;">
        <div class="pill">L1/L2/L3 结构化知识库</div>
        <div class="pill">TF-IDF 本地检索</div>
        <div class="pill">规则一致性校验</div>
        <div class="pill">千问生成解释</div>
        <div class="pill">科研原型</div>
      </div>
      <p class="muted" id="configStatus" style="margin-top:12px;">正在读取配置...</p>
    </section>

    <section class="card">
      <label for="query"><strong>患者症状输入</strong></label>
      <div class="hint">请输入已确诊原发性骨质疏松患者的<strong>症状、舌象、脉象</strong>。系统<strong>不进行</strong>影像学或骨密度诊断，也不替代临床处方决策。</div>
      <textarea id="query" placeholder="示例：患者女，80岁，腰背部疼痛伴活动受限1周，舌淡红苔薄白，脉弦。">__EXAMPLE_QUERY__</textarea>
      <div class="row">
        <button type="button" id="runMain">运行 OP-RAG 主流程</button>
        <button type="button" id="toggleAblation" class="secondary">显示/隐藏 G0-G4 消融界面</button>
        <span id="status" class="muted"></span>
      </div>
      <div class="sample-row">
        <span class="muted" style="font-size:13px;">证型测试样例：</span>
        <button type="button" class="secondary" id="sampleS1">肾阳虚</button>
        <button type="button" class="secondary" id="sampleS2">肝肾阴虚</button>
        <button type="button" class="secondary" id="sampleS3">脾肾阳虚</button>
        <button type="button" class="secondary" id="sampleS4">肾虚血瘀</button>
        <button type="button" class="secondary" id="sampleS5">脾胃虚弱</button>
        <button type="button" class="secondary" id="sampleS6">气滞血瘀</button>
      </div>
      <div id="error" class="error"></div>
    </section>

    <section class="card" id="summaryCard" style="display:none;">
      <h2>A. 检索与规则结果（知识库 + 检索器）</h2>
      <div class="summary-grid">
        <div class="summary-block">
          <h3>L1 证型检索</h3>
          <ul id="syndromes"></ul>
          <p class="score-note">分数为文本相似度，仅用于排序，不代表医学置信度。</p>
        </div>
        <div class="summary-block">
          <h3>L2 方剂检索</h3>
          <p class="muted" style="margin:0;font-size:14px;">当前证型：<strong id="selectedSyndrome">—</strong></p>
          <ul id="formulas"></ul>
        </div>
        <div class="summary-block">
          <h3>最终选择</h3>
          <ul id="selection"></ul>
        </div>
        <div class="summary-block">
          <h3>一致性校验 &amp; L3 机制匹配</h3>
          <ul id="reflection"></ul>
        </div>
      </div>
    </section>

    <section class="card" id="ablationCard" style="display:none;">
      <h2>B. G0-G4 消融结果</h2>
      <div class="summary-grid">
        <div class="summary-block"><h3>G0</h3><pre id="g0"></pre></div>
        <div class="summary-block"><h3>G1</h3><pre id="g1"></pre></div>
        <div class="summary-block"><h3>G2</h3><pre id="g2"></pre></div>
        <div class="summary-block"><h3>G3</h3><pre id="g3"></pre></div>
        <div class="summary-block"><h3>G4</h3><pre id="g4"></pre></div>
      </div>
    </section>

    <section class="card" id="resultCard" style="display:none;">
      <h2>C. 千问生成报告（LLM 解释层）</h2>
      <div class="warn">以下报告由千问基于上方检索结果生成。Step 4 基础方组成应来自知识库；若出现加减建议而知识库未提供结构化规则，<strong>不作为已验证知识库内容</strong>。</div>
      <pre id="result"></pre>
    </section>

    <section class="card disclaimer">本系统仅用于科研原型验证与软件流程演示，不进行骨质疏松诊断，不替代医生辨证论治或处方决策。任何用药方案均须由具备资质的临床医师结合患者实际情况判断。</section>
  </main>

  <script>
    const queryEl = document.getElementById("query");
    const runMainEl = document.getElementById("runMain");
    const toggleAblationEl = document.getElementById("toggleAblation");
    const statusEl = document.getElementById("status");
    const errorEl = document.getElementById("error");
    const resultCard = document.getElementById("resultCard");
    const summaryCard = document.getElementById("summaryCard");
    const ablationCard = document.getElementById("ablationCard");
    const resultEl = document.getElementById("result");
    const g0El = document.getElementById("g0");
    const g1El = document.getElementById("g1");
    const g2El = document.getElementById("g2");
    const g3El = document.getElementById("g3");
    const g4El = document.getElementById("g4");
    const syndromesEl = document.getElementById("syndromes");
    const formulasEl = document.getElementById("formulas");
    const selectionEl = document.getElementById("selection");
    const reflectionEl = document.getElementById("reflection");
    const selectedSyndromeEl = document.getElementById("selectedSyndrome");
    const configStatusEl = document.getElementById("configStatus");

    const SAMPLES = {{
      sampleS1: "患者女，68岁，已确诊原发性骨质疏松。腰背冷痛明显，腰膝酸软无力，畏寒肢冷，遇冷加重，驼背弯腰，小便频数，夜尿2-3次，舌淡苔白，脉沉细。",
      sampleS2: "患者女，72岁，已确诊原发性骨质疏松。腰膝酸痛绵绵，膝软无力，下肢偶有抽筋，头晕耳鸣，五心烦热，潮热盗汗，口干咽燥，失眠多梦，舌红少苔，脉沉细数。",
      sampleS3: "患者男，75岁，已确诊原发性骨质疏松。腰膝冷痛，食少纳呆，便溏，畏寒肢冷，双膝行走无力，弯腰驼背，面色㿠白，腹胀，舌淡胖苔白滑，脉沉迟无力。",
      sampleS4: "患者女，70岁，已确诊原发性骨质疏松。腰背周身疼痛，痛有定处、拒按，腰膝酸软，筋肉挛缩，下肢活动不利，既往外伤史，面色晦暗，舌质紫暗有瘀斑，脉涩。",
      sampleS5: "患者女，65岁，已确诊原发性骨质疏松。腰背酸痛，形体消瘦，肌肉瘦削，食少纳呆，神疲倦怠，少气懒言，大便溏薄，面色萎黄，舌质淡苔白，脉细弱。",
      sampleS6: "患者男，68岁，已确诊原发性骨质疏松。骨节疼痛，痛有定处、拒按，转侧不利，疼痛日轻夜重，筋肉挛缩，长期情志不畅，面色晦暗，舌质紫暗，脉弦涩。"
    }};

    async function fetchJsonWithTimeout(url, options = {{}}, timeoutMs = 30000) {{
      const controller = new AbortController();
      const timer = setTimeout(() => controller.abort(), timeoutMs);
      try {{
        const response = await fetch(url, {{ ...options, signal: controller.signal }});
        const data = await response.json();
        if (!response.ok) throw new Error(data.error || "请求失败");
        return data;
      }} finally {{
        clearTimeout(timer);
      }}
    }}

    function setError(message, kind = "error") {{
      errorEl.textContent = message;
      errorEl.style.display = "block";
      if (kind === "ok") {{
        errorEl.classList.add("ok");
      }} else {{
        errorEl.classList.remove("ok");
      }}
    }}

    async function loadStatus() {{
      configStatusEl.textContent = "正在读取配置...";
      try {{
        const data = await fetchJsonWithTimeout("/api/status", {{}}, 8000);
        configStatusEl.textContent = data.qwen_configured
          ? `千问已配置：${{data.model}}（${{data.url}}）`
          : "千问未配置：当前会使用本地模板输出。请检查 .env 中的 QWEN_API_KEY。";
        console.log("STATUS 成功");
      }} catch (err) {{
        configStatusEl.textContent = "配置状态读取失败：" + err.message + ". 它在检查 .env 里的 QWEN_API_KEY / QWEN_BASE_URL / QWEN_MODEL。";
        console.error("STATUS 失败", err);
      }}
    }}

    Object.entries(SAMPLES).forEach(([id, text]) => {{
      document.getElementById(id).onclick = () => {{ queryEl.value = text; }};
    }});

    function renderList(el, items) {{
      el.innerHTML = items.map(item => `<li>${{item}}</li>`).join("");
    }}

    function formatCandidate(item, rank) {{
      const label = rank === 0 ? "首选" : "备选";
      const score = typeof item.score === "number" ? item.score.toFixed(4) : item.score;
      return `${{label}}：${{item.name}}（相似度 ${{score}}）`;
    }}

    function renderAblation(mode, data) {{
      const summary = [
        `模式：${{mode.toUpperCase()}}`,
        `证型：${{data.selected_syndrome_id || "—"}}`,
        `方剂：${{data.selected_formula_id || "—"}}`,
        `一致性：${{data.formula_consistent === null ? "—" : (data.formula_consistent ? "通过" : "未通过")}}`,
        `药味Jaccard：${{typeof data.herb_jaccard === "number" ? data.herb_jaccard.toFixed(3) : "—"}}`,
      ].join("\n");
      if (mode === "g0") g0El.textContent = summary;
      if (mode === "g1") g1El.textContent = summary;
      if (mode === "g2") g2El.textContent = summary;
      if (mode === "g3") g3El.textContent = summary;
      if (mode === "g4") g4El.textContent = summary;
    }}

    async function runMainPipeline() {{
      const query = queryEl.value.trim();
      if (!query) {{ setError("请输入患者症状。"); return; }}
      runMainEl.disabled = true;
      statusEl.textContent = "运行 OP-RAG 主流程中，请稍候...";
      errorEl.style.display = "none";
      resultCard.style.display = "none";
      summaryCard.style.display = "none";
      try {{
        const data = await fetchJsonWithTimeout("/api/ask", {{
          method: "POST",
          headers: {{"Content-Type": "application/x-www-form-urlencoded;charset=UTF-8"}},
          body: new URLSearchParams({{query}})
        }}, 120000);

        const syndromeItems = (data.syndrome_candidates || []).map((item, i) => formatCandidate(item, i));
        const formulaItems = (data.formula_candidates || []).map((item, i) => formatCandidate(item, i));
        selectedSyndromeEl.textContent = data.selected_syndrome?.name || "—";
        renderList(syndromesEl, syndromeItems.length ? syndromeItems : ["无候选"]);
        renderList(formulasEl, formulaItems.length ? formulaItems : ["无候选"]);

        const composition = (data.selected_formula?.composition || []).map(item => `${{item.herb}}（${{item.role || "—"}}，${{item.dose_range || item.dose || "—"}}）`);
        renderList(selectionEl, [
          `证型：${{data.selected_syndrome?.name || "—"}}（${{data.selected_syndrome?.syndrome_id || "—"}}）`,
          `方剂：${{data.selected_formula?.name || "—"}}（${{data.selected_formula?.formula_id || "—"}}）`,
          ...(composition.length ? ["基础方组成：", ...composition.map(c => `　· ${{c}}`)] : [])
        ]);

        const consistent = data.reflection?.is_consistent;
        renderList(reflectionEl, [
          `结果：${{consistent ? "通过" : "未通过"}} — ${{data.reflection?.message || "—"}}`,
          `允许方剂：${{(data.reflection?.allowed_formula_ids || []).join("、") || "—"}}`,
          `L3 机制匹配：${{data.herb_match_count || 0}} / ${{data.herb_total_count || 0}} 味药`,
          ...(data.herb_matched?.length ? [`已匹配：${{data.herb_matched.join("、")}}`] : []),
          ...(data.herb_missing?.length ? [`未匹配：${{data.herb_missing.join("、")}}`] : [])
        ]);

        resultEl.textContent = data.report || "";
        summaryCard.style.display = "block";
        resultCard.style.display = "block";
        setError("OP-RAG 主流程成功返回 JSON", "ok");
        console.log("OP-RAG 主流程成功");
      }} catch (err) {{
        setError(`OP-RAG 主流程失败：${{err.message}}`);
        console.error("OP-RAG 主流程失败", err);
      }} finally {{
        runMainEl.disabled = false;
        statusEl.textContent = "";
      }}
    }}

    function bindAblationButtons() {{
      const buttons = ["g0", "g1", "g2", "g3", "g4"].map(mode => document.getElementById(mode));
      buttons.forEach(btn => {{
        if (!btn) return;
        btn.onclick = async () => {{
          const query = queryEl.value.trim();
          if (!query) {{ setError("请输入患者症状。"); return; }}
          btn.disabled = true;
          const mode = btn.id;
          statusEl.textContent = `运行 ${{mode.toUpperCase()}} 中，请稍候...`;
          try {{
            const data = await fetchJsonWithTimeout("/api/ablation", {{
              method: "POST",
              headers: {{"Content-Type": "application/x-www-form-urlencoded;charset=UTF-8"}},
              body: new URLSearchParams({{query, mode}})
            }}, 90000);
            renderAblation(mode, data);
            setError(`${{mode.toUpperCase()}} 成功返回 JSON`, "ok");
            console.log(`ABLATON ${{mode.toUpperCase()}} 成功`);
          }} catch (err) {{
            setError(`ABLATON ${{mode.toUpperCase()}} 失败：${{err.message}}`);
            console.error(`ABLATON ${{mode.toUpperCase()}} 失败`, err);
          }} finally {{
            btn.disabled = false;
            statusEl.textContent = "";
          }}
        }};
      }});

      const allBtn = document.getElementById("runAll");
      allBtn.onclick = async () => {{
        const query = queryEl.value.trim();
        if (!query) {{ setError("请输入患者症状。"); return; }}
        allBtn.disabled = true;
        statusEl.textContent = "运行 G0-G4 中，请稍候...";
        try {{
          const data = await fetchJsonWithTimeout("/api/ablation_all", {{
            method: "POST",
            headers: {{"Content-Type": "application/x-www-form-urlencoded;charset=UTF-8"}},
            body: new URLSearchParams({{query}})
          }}, 120000);
          const analysisMode = data.g4 || data.g3 || data.g2 || data.g1 || data.g0 || {{}};
          const syndromeItems = (analysisMode.syndrome_candidates || []).map((item, i) => formatCandidate(item, i));
          const formulaItems = (analysisMode.formula_candidates || []).map((item, i) => formatCandidate(item, i));
          selectedSyndromeEl.textContent = analysisMode.selected_syndrome?.name || "—";
          renderList(syndromesEl, syndromeItems.length ? syndromeItems : ["无候选"]);
          renderList(formulasEl, formulaItems.length ? formulaItems : ["无候选"]);
          const composition = (analysisMode.selected_formula?.composition || []).map(item => `${{item.herb}}（${{item.role || "—"}}，${{item.dose_range || item.dose || "—"}}）`);
          renderList(selectionEl, [
            `证型：${{analysisMode.selected_syndrome?.name || "—"}}（${{analysisMode.selected_syndrome?.syndrome_id || "—"}}）`,
            `方剂：${{analysisMode.selected_formula?.name || "—"}}（${{analysisMode.selected_formula?.formula_id || "—"}}）`,
            ...(composition.length ? ["基础方组成：", ...composition.map(c => `　· ${{c}}`)] : [])
          ]);
          const consistent = analysisMode.reflection?.is_consistent;
          renderList(reflectionEl, [
            `结果：${{consistent ? "通过" : "未通过"}} — ${{analysisMode.reflection?.message || "—"}}`,
            `允许方剂：${{(analysisMode.reflection?.allowed_formula_ids || []).join("、") || "—"}}`,
            `L3 机制匹配：${{analysisMode.herb_match_count || 0}} / ${{analysisMode.herb_total_count || 0}} 味药`,
            ...(analysisMode.herb_matched?.length ? [`已匹配：${{analysisMode.herb_matched.join("、")}}`] : []),
            ...(analysisMode.herb_missing?.length ? [`未匹配：${{analysisMode.herb_missing.join("、")}}`] : [])
          ]);
          renderAblation("g0", data.g0 || {{}});
          renderAblation("g1", data.g1 || {{}});
          renderAblation("g2", data.g2 || {{}});
          renderAblation("g3", data.g3 || {{}});
          renderAblation("g4", data.g4 || {{}});
          resultEl.textContent = data.g4?.report || data.g3?.report || "";
          summaryCard.style.display = "block";
          ablationCard.style.display = "block";
          resultCard.style.display = "block";
          setError("G0-G4 成功返回 JSON", "ok");
          console.log("ABLATON ALL 成功");
        }} catch (err) {{
          setError(`ABLATON ALL 失败：${{err.message}}`);
          console.error("ABLATON ALL 失败", err);
        }} finally {{
          allBtn.disabled = false;
          statusEl.textContent = "";
        }}
      }};
    }}

    function bindMainButtons() {{
      runMainEl.onclick = runMainPipeline;
      toggleAblationEl.onclick = () => {{
        ablationCard.style.display = ablationCard.style.display === "none" ? "block" : "none";
      }};
    }}

    bindMainButtons();
    bindAblationButtons();
    loadStatus();
  </script>
</body>
</html>
"""


class RagWebHandler(BaseHTTPRequestHandler):
    pipeline: OPRagPipeline
    ablation_runner: AblationRunner

    def do_GET(self) -> None:
        path = self.path.split("?", 1)[0].rstrip("/") or "/"
        if path in ("/api/status", "/api/status/"):
            client = QwenClient()
            payload = {
                "qwen_configured": client.is_configured,
                "model": QWEN_MODEL,
                "base_url": QWEN_BASE_URL,
                "url": client.chat_completions_url,
            }
            print("STATUS 成功", file=sys.stderr)
            self._send_json(payload)
            return
        if path in ("/api/debug_ablation", "/api/debug_ablation/"):
            print("STATUS 成功", file=sys.stderr)
            self._send_json({"message": "use POST /api/ablation with query and mode"}, status=200)
            return
        if path == "/favicon.ico":
            self.send_response(204)
            self._send_no_cache_headers()
            self.end_headers()
            return
        if path not in ("/", "/index.html"):
            self.send_error(404, f"Not Found: {path}")
            return
        page = HTML_PAGE.replace("__EXAMPLE_QUERY__", html.escape(EXAMPLE_QUERY))
        self._send_text(page, content_type="text/html; charset=utf-8")

    def do_POST(self) -> None:
        if self.path == "/api/ask":
            try:
                content_length = int(self.headers.get("Content-Length", "0"))
                body = self.rfile.read(content_length).decode("utf-8")
                fields = parse_qs(body)
                query = fields.get("query", [""])[0].strip()
                if not query:
                    self._send_json({"error": "请输入患者症状。"}, status=400)
                    return
                result = self.pipeline.run(query, use_llm=True)
                context = result["context"]
                formula = context["selected_formula"]
                composition = formula.get("composition", [])
                herb_records = context["herb_records"]
                matched_names = {h.get("herb_name") for h in herb_records}
                all_herbs = [item.get("herb") for item in composition if item.get("herb")]
                missing = [name for name in all_herbs if name not in matched_names]
                payload = {
                    "report": result["report"],
                    "syndrome_candidates": context["syndrome_candidates"],
                    "formula_candidates": context["formula_candidates"],
                    "selected_syndrome": {
                        "syndrome_id": context["selected_syndrome"].get("syndrome_id"),
                        "name": context["selected_syndrome"].get("name"),
                    },
                    "selected_formula": {
                        "formula_id": formula.get("formula_id"),
                        "name": formula.get("name"),
                        "composition": composition,
                    },
                    "reflection": context["reflection"],
                    "herb_match_count": len(herb_records),
                    "herb_total_count": len(all_herbs),
                    "herb_matched": sorted(n for n in matched_names if n),
                    "herb_missing": missing,
                }
                self._send_json(payload)
                return
            except Exception as exc:
                self._send_json({"error": str(exc)}, status=500)
                return

        if self.path == "/api/ablation":
            try:
                content_length = int(self.headers.get("Content-Length", "0"))
                body = self.rfile.read(content_length).decode("utf-8")
                fields = parse_qs(body)
                query = fields.get("query", [""])[0].strip()
                mode = fields.get("mode", ["g3"])[0].strip().lower()
                if not query:
                    self._send_json({"error": "请输入患者症状。"}, status=400)
                    return
                case = AblationCaseInput(case_id="web_case", patient_text=query, use_llm=True)
                payload = self.ablation_runner.run_case(case, mode).to_dict()
                print(f"ABLATON {mode.upper()} 成功", file=sys.stderr)
                self._send_json(payload)
                return
            except Exception as exc:
                print(f"ABLATON {mode.upper() if 'mode' in locals() else 'UNKNOWN'} 失败", file=sys.stderr)
                self._send_json({"error": str(exc)}, status=500)
                return

        if self.path == "/api/ablation_all":
            try:
                content_length = int(self.headers.get("Content-Length", "0"))
                body = self.rfile.read(content_length).decode("utf-8")
                fields = parse_qs(body)
                query = fields.get("query", [""])[0].strip()
                if not query:
                    self._send_json({"error": "请输入患者症状。"}, status=400)
                    return
                case = AblationCaseInput(case_id="web_case", patient_text=query, use_llm=True)
                payload = self.ablation_runner.run_all(case)
                print("ABLATON ALL 成功", file=sys.stderr)
                self._send_json(payload)
                return
            except Exception as exc:
                print("ABLATON ALL 失败", file=sys.stderr)
                self._send_json({"error": str(exc)}, status=500)
                return

        self.send_error(404, f"Not Found: {self.path}")

    def log_message(self, format: str, *args: object) -> None:
        return

    def _send_no_cache_headers(self) -> None:
        self.send_header("Cache-Control", "no-store, no-cache, must-revalidate, max-age=0")
        self.send_header("Pragma", "no-cache")
        self.send_header("Expires", "0")

    def _send_text(self, text: str, content_type: str, status: int = 200) -> None:
        encoded = text.encode("utf-8")
        self.send_response(status)
        self._send_no_cache_headers()
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)

    def _send_json(self, payload: dict, status: int = 200) -> None:
        text = json.dumps(payload, ensure_ascii=False)
        self._send_text(text, content_type="application/json; charset=utf-8", status=status)

    def log_error(self, format: str, *args: object) -> None:
        message = format % args if args else format
        print(f"[WEB_ERROR] {message}", file=sys.stderr)
        traceback.print_exc()


def _lan_ip_addresses() -> list[str]:
    candidates: list[str] = []
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
            sock.connect(("8.8.8.8", 80))
            ip = sock.getsockname()[0]
            if ip and not ip.startswith("127."):
                candidates.append(ip)
    except OSError:
        pass

    try:
        for info in socket.getaddrinfo(socket.gethostname(), None, socket.AF_INET):
            ip = info[4][0]
            if ip.startswith("127.") or ip.startswith("169.254."):
                continue
            if ip not in candidates:
                candidates.append(ip)
    except OSError:
        pass

    return candidates


def run_server(host: str = DEFAULT_HOST, port: int = DEFAULT_PORT) -> None:
    debug = os.environ.get("OP_RAG_DEBUG", "").strip().lower() in {"1", "true", "yes"}
    configure_pipeline_trace(enabled=debug)
    kb = load_kb()
    pipeline = OPRagPipeline(kb)
    RagWebHandler.pipeline = pipeline
    RagWebHandler.ablation_runner = AblationRunner(kb, pipeline)
    server = ThreadingHTTPServer((host, port), RagWebHandler)
    local_url = f"http://127.0.0.1:{port}"
    print("OP-RAG web UI is running.")
    print(f"  电脑本机：{local_url}")
    lan_ips = _lan_ip_addresses()
    if lan_ips:
        print("  手机访问（需与电脑同一 WiFi）：")
        for ip in lan_ips:
            print(f"    http://{ip}:{port}")
    else:
        print("  手机访问：未能自动检测局域网 IP，请在 cmd 运行 ipconfig 查看 IPv4 地址。")
    print("Press Ctrl+C to stop.")
    print("[WEB_NOTE] 页面顶部会先调用 /api/status；主流程按钮调用 /api/ask；G0-G4 调用 /api/ablation 与 /api/ablation_all。")
    webbrowser.open(local_url)
    server.serve_forever()


if __name__ == "__main__":
    run_server()
