from __future__ import annotations

import html
import json
import os
import socket
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

from .loader import load_kb


DEFAULT_HOST = os.getenv("HERB_UI_HOST", "0.0.0.0")
DEFAULT_PORT = int(os.getenv("HERB_UI_PORT", "8080"))
MAX_TARGETS_PREVIEW = int(os.getenv("HERB_UI_MAX_TARGETS", "18"))
MAX_PATHWAYS_PREVIEW = int(os.getenv("HERB_UI_MAX_PATHWAYS", "10"))
MAX_EVIDENCE_PREVIEW = int(os.getenv("HERB_UI_MAX_EVIDENCE", "6"))


def _normalize_text(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False)


kb = load_kb()
HERBS = kb.get("herbs", [])


HTML_PAGE = """<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Herbs 数据可视化面板</title>
  <style>
    :root {{
      --bg: #f4f7fb;
      --panel: #ffffff;
      --text: #182230;
      --muted: #667085;
      --border: #e5e7eb;
      --primary: #355cff;
      --chip: #eef2ff;
      --shadow: 0 10px 30px rgba(16, 24, 40, 0.06);
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", "Microsoft YaHei", sans-serif;
      background: linear-gradient(180deg, #f8fbff 0%, var(--bg) 100%);
      color: var(--text);
    }}
    .wrap {{ max-width: 1280px; margin: 0 auto; padding: 28px 18px 42px; }}
    .hero {{
      background: rgba(255,255,255,0.88);
      backdrop-filter: blur(10px);
      border: 1px solid rgba(229,231,235,0.8);
      border-radius: 22px;
      padding: 22px;
      box-shadow: var(--shadow);
      margin-bottom: 16px;
    }}
    h1 {{ margin: 0 0 8px; font-size: 28px; }}
    .muted {{ color: var(--muted); line-height: 1.65; }}
    .stats {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 12px; margin-top: 16px; }}
    .stat {{
      background: #fff;
      border: 1px solid var(--border);
      border-radius: 16px;
      padding: 14px;
    }}
    .stat .n {{ font-size: 24px; font-weight: 800; margin-bottom: 4px; }}
    .stat .l {{ color: var(--muted); font-size: 13px; }}
    .controls {{
      display: grid;
      grid-template-columns: 1.4fr 0.8fr 0.8fr;
      gap: 12px;
      margin: 16px 0;
    }}
    input, select {{
      width: 100%;
      border: 1px solid var(--border);
      border-radius: 14px;
      background: #fff;
      padding: 14px 16px;
      font-size: 15px;
      outline: none;
    }}
    input:focus, select:focus {{ border-color: #89a0ff; box-shadow: 0 0 0 4px rgba(53,92,255,0.08); }}
    .layout {{ display: grid; grid-template-columns: 380px 1fr; gap: 16px; align-items: start; }}
    .panel {{ background: rgba(255,255,255,0.9); border: 1px solid var(--border); border-radius: 20px; box-shadow: var(--shadow); }}
    .left {{ padding: 10px; max-height: 76vh; overflow: auto; }}
    .right {{ padding: 18px; min-height: 76vh; }}
    .item {{
      padding: 14px;
      border-radius: 16px;
      border: 1px solid transparent;
      margin-bottom: 10px;
      cursor: pointer;
      transition: all .16s ease;
      background: #fff;
    }}
    .item:hover {{ transform: translateY(-1px); border-color: #d6ddff; box-shadow: 0 8px 18px rgba(53,92,255,0.08); }}
    .item.active {{ border-color: #9bb0ff; background: #f7f9ff; }}
    .name {{ font-size: 17px; font-weight: 800; margin-bottom: 4px; }}
    .tags {{ display: flex; flex-wrap: wrap; gap: 6px; margin-top: 8px; }}
    .tag {{
      display: inline-block;
      background: var(--chip);
      color: #3447a8;
      padding: 4px 10px;
      border-radius: 999px;
      font-size: 12px;
      max-width: 100%;
      overflow: hidden;
      text-overflow: ellipsis;
      white-space: nowrap;
    }}
    .section {{ margin-bottom: 18px; }}
    .section h2 {{ margin: 0 0 10px; font-size: 19px; }}
    .meta-grid {{ display: grid; grid-template-columns: repeat(2, 1fr); gap: 12px; }}
    .box {{ background: #f8fafc; border: 1px solid #e5e7eb; border-radius: 16px; padding: 14px; }}
    .box .label {{ color: var(--muted); font-size: 13px; margin-bottom: 6px; }}
    .box .value {{ line-height: 1.7; white-space: pre-wrap; word-break: break-word; }}
    ul {{ margin: 8px 0 0; padding-left: 18px; line-height: 1.7; }}
    .pill {{ display: inline-block; margin: 0 8px 8px 0; padding: 6px 10px; background: #eef2ff; border-radius: 999px; color: #3447a8; font-size: 13px; }}
    .empty {{ color: var(--muted); padding: 16px; }}
    .foot {{ margin-top: 14px; color: var(--muted); font-size: 13px; }}
    @media (max-width: 1024px) {{
      .stats, .controls, .layout, .meta-grid {{ grid-template-columns: 1fr; }}
      .left {{ max-height: none; }}
      .right {{ min-height: auto; }}
    }}
  </style>
</head>
<body>
  <div class="wrap">
    <div class="hero">
      <h1>Herbs 数据可视化面板</h1>
      <div class="muted">
        本地浏览 `data/kb/herbs.json` 的靶点、通路、证据文献与描述摘要。支持按药名、通路、靶点和文献检索。
      </div>
      <div class="stats">
        <div class="stat"><div class="n" id="statHerbs">—</div><div class="l">药材数量</div></div>
        <div class="stat"><div class="n" id="statTargets">—</div><div class="l">首屏展示靶点总数</div></div>
        <div class="stat"><div class="n" id="statPathways">—</div><div class="l">首屏展示通路总数</div></div>
        <div class="stat"><div class="n" id="statPapers">—</div><div class="l">首屏展示文献数</div></div>
      </div>
    </div>

    <div class="controls">
      <input id="search" type="text" placeholder="搜索药名、靶点、通路、文献 PMID..." />
      <select id="sortBy">
        <option value="name">按药名排序</option>
        <option value="targets">按靶点数排序</option>
        <option value="pathways">按通路数排序</option>
        <option value="papers">按文献数排序</option>
      </select>
      <select id="evidenceFilter">
        <option value="all">全部证据</option>
        <option value="has">仅显示有文献</option>
        <option value="none">仅显示无文献</option>
      </select>
    </div>

    <div class="layout">
      <div class="panel left">
        <div id="list"></div>
      </div>
      <div class="panel right">
        <div id="detail"></div>
      </div>
    </div>

    <div class="foot">提示：若要在手机上打开，把浏览器地址里的 127.0.0.1 换成电脑局域网 IP 即可。</div>
  </div>

  <script>
    const herbs = __HERBS__;
    const searchEl = document.getElementById('search');
    const sortByEl = document.getElementById('sortBy');
    const evidenceFilterEl = document.getElementById('evidenceFilter');
    const listEl = document.getElementById('list');
    const detailEl = document.getElementById('detail');

    document.getElementById('statHerbs').textContent = herbs.length;
    function listLength(value) {{ return Array.isArray(value) ? value.length : 0; }}
    document.getElementById('statTargets').textContent = herbs.reduce((n, h) => n + listLength(h.targets_op_related), 0);
    document.getElementById('statPathways').textContent = herbs.reduce((n, h) => n + listLength(h.pathways), 0);
    document.getElementById('statPapers').textContent = herbs.reduce((n, h) => n + listLength(h.evidence_papers), 0);

    function normalize(text) {{
      return String(text || '').toLowerCase();
    }}

    function matchesHerb(herb, query) {{
      if (!query) return true;
      const q = normalize(query);
      const fields = [
        herb.herb_name,
        herb.tcm_function,
        herb.text_description,
        ...(herb.targets_op_related || []),
        ...(herb.pathways || []),
        ...(herb.evidence_papers || [])
      ];
      return fields.some(v => normalize(v).includes(q));
    }}

    function applyFilters() {{
      const query = searchEl.value.trim();
      const sortBy = sortByEl.value;
      const evidenceFilter = evidenceFilterEl.value;
      let filtered = herbs.filter(h => matchesHerb(h, query));
      if (evidenceFilter === 'has') filtered = filtered.filter(h => (h.evidence_papers || []).length > 0);
      if (evidenceFilter === 'none') filtered = filtered.filter(h => (h.evidence_papers || []).length === 0);
      filtered.sort((a, b) => {{
        if (sortBy === 'targets') return listLength(b.targets_op_related) - listLength(a.targets_op_related) || a.herb_name.localeCompare(b.herb_name, 'zh-Hans-CN');
        if (sortBy === 'pathways') return listLength(b.pathways) - listLength(a.pathways) || a.herb_name.localeCompare(b.herb_name, 'zh-Hans-CN');
        if (sortBy === 'papers') return listLength(b.evidence_papers) - listLength(a.evidence_papers) || a.herb_name.localeCompare(b.herb_name, 'zh-Hans-CN');
        return a.herb_name.localeCompare(b.herb_name, 'zh-Hans-CN');
      }});
      return filtered;
    }}

    let currentName = herbs.length ? (herbs[0].herb_name || '') : '';

    function preview(arr, max) {{
      return (arr || []).slice(0, max);
    }}

    function renderDetail(herb) {{
      if (!herb) {{
        detailEl.innerHTML = '<div class="empty">没有找到匹配的药材。</div>';
        return;
      }}
      const targets = herb.targets_op_related || [];
      const pathways = herb.pathways || [];
      const papers = herb.evidence_papers || [];
      const targetPreview = preview(targets, __MAX_TARGETS__);
      const pathwayPreview = preview(pathways, __MAX_PATHWAYS__);
      const paperPreview = preview(papers, __MAX_PAPERS__);
      detailEl.innerHTML = `
        <div class="section">
          <h2>${{herb.herb_name}}</h2>
          <div class="muted" style="margin-bottom:10px;">${{herb.tcm_function || '—'}}</div>
          <div class="tags">
            <span class="tag">靶点 ${{targets.length}}</span>
            <span class="tag">通路 ${{pathways.length}}</span>
            <span class="tag">文献 ${{papers.length}}</span>
          </div>
        </div>
        <div class="section meta-grid">
          <div class="box">
            <div class="label">文本描述</div>
            <div class="value">${{(herb.text_description || '—').replace(/</g, '&lt;').replace(/>/g, '&gt;')}}</div>
          </div>
          <div class="box">
            <div class="label">基础信息</div>
            <div class="value">
              <strong>${{herb.herb_name}}</strong><br/>
              证据文献 PMID：${{papers.length ? papers.join('，') : '无'}}
            </div>
          </div>
        </div>
        <div class="section">
          <h2>OP 相关靶点</h2>
          <div>${{targetPreview.map(t => `<span class="pill">${{t}}</span>`).join('')}}${{targets.length > __MAX_TARGETS__ ? `<span class="pill">… 另外 ${{targets.length - __MAX_TARGETS__}} 个</span>` : ''}}</div>
        </div>
        <div class="section">
          <h2>富集通路</h2>
          <div>${{pathwayPreview.map(p => `<span class="pill">${{p}}</span>`).join('')}}${{pathways.length > __MAX_PATHWAYS__ ? `<span class="pill">… 另外 ${{pathways.length - __MAX_PATHWAYS__}} 个</span>` : ''}}</div>
        </div>
        <div class="section">
          <h2>证据文献</h2>
          <ul>
            ${{paperPreview.length ? paperPreview.map(p => `<li>${{p}}</li>`).join('') : '<li>暂无文献</li>'}}
          </ul>
          ${{papers.length > __MAX_PAPERS__ ? `<div class="muted" style="margin-top:8px;">还有 ${{papers.length - __MAX_PAPERS__}} 篇未展开。</div>` : ''}}
        </div>
      `;
    }}

    function renderList() {{
      const items = applyFilters();
      listEl.innerHTML = items.map((herb) => `
        <div class="item ${{herb.herb_name === currentName ? 'active' : ''}}" data-name="${{herb.herb_name}}">
          <div class="name">${{herb.herb_name}}</div>
          <div class="muted" style="font-size:13px;">${{herb.tcm_function || ''}}</div>
          <div class="tags">
            <span class="tag">靶点 ${{(herb.targets_op_related || []).length}}</span>
            <span class="tag">通路 ${{(herb.pathways || []).length}}</span>
            <span class="tag">文献 ${{(herb.evidence_papers || []).length}}</span>
          </div>
        </div>
      `).join('') || '<div class="empty">没有找到匹配结果。</div>';

      const current = items.find(h => h.herb_name === currentName) || items[0] || null;
      if (current) currentName = current.herb_name;
      renderDetail(current);

      listEl.querySelectorAll('.item').forEach(el => {{
        el.addEventListener('click', () => {{
          currentName = el.dataset.name;
          renderList();
        }});
      }});
    }}

    searchEl.addEventListener('input', renderList);
    sortByEl.addEventListener('change', renderList);
    evidenceFilterEl.addEventListener('change', renderList);

    renderList();
  </script>
</body>
</html>
"""


class HerbDashboardHandler(BaseHTTPRequestHandler):
    herbs = HERBS

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/api/herbs":
            self._send_json(self.herbs)
            return
        if parsed.path != "/":
            self.send_error(404, "Not Found")
            return
        page = HTML_PAGE.replace("__HERBS__", json.dumps(self.herbs, ensure_ascii=False))
        page = page.replace("__MAX_TARGETS__", str(MAX_TARGETS_PREVIEW))
        page = page.replace("__MAX_PATHWAYS__", str(MAX_PATHWAYS_PREVIEW))
        page = page.replace("__MAX_PAPERS__", str(MAX_EVIDENCE_PREVIEW))
        # The HTML template uses doubled braces so Python does not interpret
        # JavaScript and CSS blocks as formatting fields.
        page = page.replace("{{", "{").replace("}}", "}")
        self._send_text(page, "text/html; charset=utf-8")

    def log_message(self, format: str, *args: object) -> None:
        return

    def _send_text(self, text: str, content_type: str, status: int = 200) -> None:
        payload = text.encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def _send_json(self, payload: object, status: int = 200) -> None:
        self._send_text(json.dumps(payload, ensure_ascii=False), "application/json; charset=utf-8", status)


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
    return candidates


def run_server(host: str = DEFAULT_HOST, port: int = DEFAULT_PORT) -> None:
    server = ThreadingHTTPServer((host, port), HerbDashboardHandler)
    local_url = f"http://127.0.0.1:{port}"
    print("Herbs dashboard is running.")
    print(f"  本机访问：{local_url}")
    for ip in _lan_ip_addresses():
        print(f"  局域网访问：http://{ip}:{port}")
    print("Press Ctrl+C to stop.")
    webbrowser.open(local_url)
    server.serve_forever()


if __name__ == "__main__":
    run_server()
