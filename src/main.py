from __future__ import annotations

import argparse

from rich.console import Console
from rich.panel import Panel

import os

from .loader import load_kb
from .pipeline_trace import log as pipeline_log
from .rag_pipeline import OPRagPipeline


EXAMPLE_PATIENT_TEXT = (
    "患者腰膝酸冷，畏寒肢冷，夜尿频多，神疲乏力，骨痛遇寒加重，"
    "舌淡胖苔白，脉沉迟。"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="骨质疏松中医RAG方剂推荐 + 机制解释 MVP"
    )
    parser.add_argument(
        "--query",
        "-q",
        default=EXAMPLE_PATIENT_TEXT,
        help="患者症状描述文本",
    )
    parser.add_argument(
        "--no-llm",
        action="store_true",
        help="不调用千问，使用本地模板生成输出",
    )
    parser.add_argument(
        "--debug-pipeline",
        action="store_true",
        help="输出 RAG 药味/靶点数据缺口警告（或设环境变量 OP_RAG_DEBUG=1）",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    console = Console()

    from .pipeline_trace import configure_pipeline_trace

    debug = args.debug_pipeline or os.environ.get("OP_RAG_DEBUG", "").strip() in ("1", "true", "yes")
    configure_pipeline_trace(enabled=debug, verbose=args.debug_pipeline)

    kb = load_kb()
    pipeline = OPRagPipeline(kb)
    result = pipeline.run(patient_text=args.query, use_llm=not args.no_llm)

    context = result["context"]
    console.print(Panel.fit("OP-RAG MVP 运行完成", style="green"))
    console.print("[bold]L1 证型候选[/bold]")
    console.print(context["syndrome_candidates"])
    console.print("[bold]L2 方剂候选[/bold]")
    console.print(context["formula_candidates"])
    console.print("[bold]自反思校验[/bold]")
    console.print(context["reflection"])
    console.print(Panel(result["report"], title="推荐与机制解释", border_style="cyan"))


if __name__ == "__main__":
    main()
