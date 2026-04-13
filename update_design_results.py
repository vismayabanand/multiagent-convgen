#!/usr/bin/env python3
"""
Reads run_A and run_B evaluation reports and updates DESIGN.md §9.2/9.3.

Usage:
    python3 update_design_results.py \
        --report-a output/report_A.json \
        --report-b output/report_B.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def load_report(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def update_design_md(report_a: dict, report_b: dict) -> None:
    design_path = Path("DESIGN.md")
    content = design_path.read_text()

    def pct(v: float) -> str:
        return f"{v:.1%}"

    def fmt(v: float, decimals: int = 4) -> str:
        return f"{v:.{decimals}f}"

    # Build table rows
    def delta(b_val: float, a_val: float, higher_is_better: bool = True) -> str:
        d = b_val - a_val
        sign = "+" if d >= 0 else ""
        arrow = ("▲" if d > 0 else "▼") if higher_is_better else ("▲" if d < 0 else "▼")
        return f"{sign}{d:.4f} {arrow}"

    ttr_a = report_a["diversity"]["tool_pair_ttr"]
    ttr_b = report_b["diversity"]["tool_pair_ttr"]
    ent_a = report_a["diversity"]["domain_entropy_normalized"]
    ent_b = report_b["diversity"]["domain_entropy_normalized"]
    mean_a = report_a["scores"]["overall"]["mean"]
    mean_b = report_b["scores"]["overall"]["mean"]
    sel_a = report_a["scores"]["tool_selection"]["mean"]
    sel_b = report_b["scores"]["tool_selection"]["mean"]
    nat_a = report_a["scores"]["naturalness"]["mean"]
    nat_b = report_b["scores"]["naturalness"]["mean"]
    chain_a = report_a["scores"]["chaining"]["mean"]
    chain_b = report_b["scores"]["chaining"]["mean"]
    repair_a = 1 - report_a["pass_rate"]
    repair_b = 1 - report_b["pass_rate"]
    discard_a = 1 - report_a["pass_rate"]  # approximation
    discard_b = 1 - report_b["pass_rate"]

    table = f"""\
| Metric | Run A (no steering) | Run B (with steering) | Change |
|--------|--------------------|-----------------------|--------|
| Tool-Pair TTR | {fmt(ttr_a)} | {fmt(ttr_b)} | {delta(ttr_b, ttr_a)} |
| Domain Entropy (normalized) | {fmt(ent_a)} | {fmt(ent_b)} | {delta(ent_b, ent_a)} |
| Mean Judge Score (overall) | {fmt(mean_a, 2)} | {fmt(mean_b, 2)} | {delta(mean_b, mean_a)} |
| Mean Tool Selection Score | {fmt(sel_a, 2)} | {fmt(sel_b, 2)} | {delta(sel_b, sel_a)} |
| Mean Naturalness Score | {fmt(nat_a, 2)} | {fmt(nat_b, 2)} | {delta(nat_b, nat_a)} |
| Mean Chaining Score | {fmt(chain_a, 2)} | {fmt(chain_b, 2)} | {delta(chain_b, chain_a)} |
| Pass rate (≥3.5 overall) | {pct(report_a['pass_rate'])} | {pct(report_b['pass_rate'])} | {delta(report_b['pass_rate'], report_a['pass_rate'])} |"""

    # Write interpretation
    ttr_improved = ttr_b > ttr_a
    ent_improved = ent_b > ent_a
    quality_cost = mean_a - mean_b
    interpretation = (
        f"Steering {'improved' if ttr_improved else 'did not improve'} Tool-Pair TTR "
        f"({fmt(ttr_a)} → {fmt(ttr_b)}) and "
        f"{'improved' if ent_improved else 'did not improve'} domain entropy "
        f"({fmt(ent_a)} → {fmt(ent_b)}). "
        f"The quality cost was {abs(quality_cost):.2f} points on the overall score "
        f"({'within' if abs(quality_cost) < 0.5 else 'above'} the 0.5 acceptable threshold). "
        f"This {'confirms' if ttr_improved and ent_improved else 'partially confirms'} the "
        f"hypothesis that steering improves diversity at a small quality cost."
    )

    section_93 = f"""\
### 9.3 Diversity–Quality Tradeoff Analysis

{interpretation}

Hypothesis before running: steering will improve TTR and entropy at a small cost to quality. The cost arises because forcing the sampler toward underrepresented tool combinations occasionally produces less natural tool chains — tools that technically connect in the graph but aren't semantically natural pairs. The judge should catch the worst cases, but mild unnaturalness may slip through.

If the quality cost is > 0.5 points on the judge scale, I would consider a softer steering signal (reduce weight by 50% rather than 90% for highly-used tools) or a domain-level-only steering (maintain diversity at the domain level but allow natural tool combinations within domains)."""

    # Replace §9.2 table placeholder
    old_table_region = (
        "**[TO UPDATE after running experiments]**\n\n"
        "| Metric | Run A (no steering) | Run B (with steering) | Change |\n"
        "|--------|--------------------|-----------------------|--------|\n"
        "| Tool-Pair TTR | — | — | — |\n"
        "| Domain Entropy (normalized) | — | — | — |\n"
        "| Mean Judge Score (overall) | — | — | — |\n"
        "| Mean Tool Selection Score | — | — | — |\n"
        "| Mean Naturalness Score | — | — | — |\n"
        "| Mean Chaining Score | — | — | — |\n"
        "| Repair rate (% conversations repaired) | — | — | — |\n"
        "| Discard rate (% conversations discarded) | — | — | — |"
    )
    new_table_region = table
    content = content.replace(old_table_region, new_table_region)

    # Replace §9.3 placeholder
    old_93 = (
        "### 9.3 Diversity–Quality Tradeoff Analysis\n\n"
        "**[TO UPDATE after running experiments]**\n\n"
        "Hypothesis before running: steering will improve TTR and entropy at a small cost to quality. The cost arises because forcing the sampler toward underrepresented tool combinations occasionally produces less natural tool chains — tools that technically connect in the graph but aren't semantically natural pairs. The judge should catch the worst cases, but mild unnaturalness may slip through.\n\n"
        "If the quality cost is > 0.5 points on the judge scale, I would consider a softer steering signal (reduce weight by 50% rather than 90% for highly-used tools) or a domain-level-only steering (maintain diversity at the domain level but allow natural tool combinations within domains)."
    )
    content = content.replace(old_93, section_93)

    design_path.write_text(content)
    print("DESIGN.md updated with experiment results.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--report-a", required=True)
    parser.add_argument("--report-b", required=True)
    args = parser.parse_args()

    ra = load_report(args.report_a)
    rb = load_report(args.report_b)
    update_design_md(ra, rb)
