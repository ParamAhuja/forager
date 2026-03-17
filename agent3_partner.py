"""
Phase 3 — Agent 3: The Partner (v2 — Forensic Investment Committee)
====================================================================
Responsibilities:
  1. Accept a fully enriched EnrichedItem from Agent 2 v2.
  2. Pre-compute deterministic conviction signals from the three new v2
     forensic engines before the LLM is called — these are machine-verified
     facts the LLM cannot override or hallucinate around.
  3. Serialize the full Agent 2 v2 payload (including ArchitecturalAuditReport,
     ExecutionMetrics, EcosystemHealthReport) into a grounded context block.
  4. Call the Groq LLM with a strict, forensically-aware system prompt.
  5. Return a PartnerOutput (chain_of_thought + memo_md) covering:
       §1  Thesis / Core Signal
       §2  Architectural Defensibility & Tech Debt
       §3  Execution & Bus Factor Risk
       §4  Ecosystem Velocity   ← primary conviction signal
       §5  Developer Momentum   ← secondary validation (commit data)
       §6  Missing Intelligence / Data Gaps (verbatim — mandatory)
       §7  Investment Verdict

Hallucination guardrails:
  • Every metric in the memo must be present in the QUANTITATIVE CONTEXT block.
  • The Strong Conviction BUY signal is pre-computed in Python, not inferred by
    the LLM. The LLM is told the machine verdict and must match it.
  • All strings from every repo's data_gaps list are reproduced verbatim in §6.
    Zero paraphrasing, zero reordering, zero omissions.
  • If metrics are absent, the function raises ValueError before calling the API.

Agent 1 version assumption:
  This module is intentionally designed around Agent 1 v1 (ScoutRationale with
  five fields: chain_of_thought, architectural_shift, developer_adoption,
  oss_milestone, noise_signal, one_line_summary). The v2 Agent 1 intelligence
  layers (TalentMigrationReport, CompetitiveLandscapeReport, ReadinessCalibration)
  are not consumed here — this module treats Agent 1 as a black-box signal
  classifier whose only output is ClassifiedItem.rationale.
"""

from __future__ import annotations

import json
import re
import textwrap
from datetime import datetime, timedelta, timezone
from typing import Optional

from groq import Groq
from pydantic import BaseModel, Field

# ── Agent 1 v1 imports (no v2 intelligence layers) ───────────────────────────
from agent1_scout import (
    ClassifiedItem,
    Classification,
    DataSource,
    RawItem,
    ScoutRationale,
)

# ── Agent 2 v2 imports (all three forensic engines + legacy fields) ───────────
from agent2_github_quant import (
    ArchitecturalAuditReport,
    CommitVelocity,
    EnrichedItem,
    EcosystemHealthReport,
    ExecutionMetrics,
    IssueResolutionMetrics,
    QuantitativeMetrics,
    RateLimitSnapshot,
    RepoProfile,
)


# ─────────────────────────────────────────────────────────────────────────────
# §1  PartnerOutput Model
# ─────────────────────────────────────────────────────────────────────────────

class PartnerOutput(BaseModel):
    """
    Structured output contract for Agent 3 v2.

    chain_of_thought:
        The Partner's internal deliberation (3–5 sentences) that explicitly:
        (a) assesses whether the three forensic signals (architectural debt,
            bus factor, ecosystem velocity) support or contradict the Agent 1
            signal verdict;
        (b) states how each data gap limits conviction;
        (c) confirms or challenges the machine-computed `strong_conviction_eligible`
            flag before committing to the memo verdict.

    memo_md:
        The final Markdown investment memo, formatted with exactly 7 sections.
        All metric citations must reference values from the QUANTITATIVE CONTEXT
        block. The verdict in §7 MUST match the machine-computed conviction signal.
    """
    chain_of_thought: str = Field(
        ...,
        description=(
            "Internal deliberation (3–5 sentences) that debates: (1) whether "
            "the three forensic pillars — architectural defensibility, bus factor "
            "risk, and ecosystem velocity — support the verdict, (2) how each "
            "data gap limits conviction, (3) explicit confirmation or challenge "
            "of the STRONG_CONVICTION_ELIGIBLE machine flag."
        ),
    )
    memo_md: str = Field(
        ...,
        description=(
            "The complete 7-section investment memo in Markdown. All metric "
            "citations must match values from the QUANTITATIVE CONTEXT block. "
            "§6 must reproduce all DATA GAP strings verbatim."
        ),
    )


# ─────────────────────────────────────────────────────────────────────────────
# §2  Conviction Signal Pre-Computation (deterministic — runs before LLM call)
# ─────────────────────────────────────────────────────────────────────────────

class ConvictionSignals:
    """
    Machine-verified conviction assessment computed from Agent 2 v2 forensic data.

    Three binary pillars — all three must be True for STRONG CONVICTION:

      Pillar A — Ecosystem Velocity (EcosystemHealthReport)
        True when download_trend is SURGING or GROWING.
        Rationale: Registry downloads are immune to star inflation. SURGING/GROWING
        downloads prove real developer adoption decisions, not social proof.

      Pillar B — Execution Discipline (ExecutionMetrics)
        True when bus_factor_risk is NOT CRITICAL (bus_factor ≥ 2).
        Rationale: A Bus Factor of 1 is an existential operational risk. A single
        departure collapses the project. No amount of ecosystem velocity offsets
        this risk for institutional-grade capital allocation.

      Pillar C — Architectural Defensibility (ArchitecturalAuditReport)
        True when debt_classification is NOT CRITICAL (debt_score < 75).
        Rationale: CRITICAL technical debt signals an unsalvageable codebase or
        AI-generated scaffolding without engineering discipline. The cost-to-
        remediate exceeds the value of the technology lead.

    If all three pillars are True:
        verdict_floor = "HIGH CONVICTION BUY"
        The LLM may still downgrade to MONITOR if data gaps are critical.

    If any pillar is False:
        verdict_ceiling = "MONITOR"
        The LLM MUST NOT issue a HIGH CONVICTION BUY regardless of qualitative signal.
    """

    __slots__ = (
        "ecosystem_surging", "execution_disciplined", "arch_defensible",
        "strong_conviction_eligible", "blockers",
        "eco_detail", "exec_detail", "arch_detail",
    )

    def __init__(self, metrics: list[QuantitativeMetrics]) -> None:
        self.blockers:   list[str] = []
        self.eco_detail  = ""
        self.exec_detail = ""
        self.arch_detail = ""

        if not metrics:
            self.ecosystem_surging       = False
            self.execution_disciplined   = False
            self.arch_defensible         = False
            self.strong_conviction_eligible = False
            self.blockers.append("CRITICAL: No QuantitativeMetrics available — all pillars unverified.")
            return

        m = metrics[0]   # Primary repository — single repo is the norm

        # ── Pillar A: Ecosystem Velocity ──────────────────────────────────────
        eco = m.ecosystem_health
        if eco:
            if eco.download_trend in ("SURGING", "GROWING"):
                self.ecosystem_surging = True
                self.eco_detail = (
                    f"PASS — {eco.download_trend} ({eco.dominant_ecosystem or 'registry'}: "
                    f"{eco.npm_weekly_downloads or eco.pypi_weekly_downloads or 0:,}/wk, "
                    f"health score {eco.ecosystem_health_score}/100)"
                )
            else:
                self.ecosystem_surging = False
                self.eco_detail = (
                    f"FAIL — Trend is {eco.download_trend} "
                    f"(health score {eco.ecosystem_health_score}/100, "
                    f"{eco.package_name_inferred or 'package'})"
                )
                self.blockers.append(
                    f"Ecosystem download trend is {eco.download_trend} — not SURGING or GROWING. "
                    f"Real-world developer adoption is not accelerating."
                )
        else:
            self.ecosystem_surging = False
            self.eco_detail = "UNAVAILABLE — EcosystemHealthReport not populated."
            self.blockers.append(
                "EcosystemHealthReport unavailable — cannot verify developer adoption "
                "via package registry. Pillar A unverified; conviction ceiling is MONITOR."
            )

        # ── Pillar B: Execution Discipline ────────────────────────────────────
        exec_m = m.execution_metrics
        if exec_m:
            if exec_m.bus_factor_risk != "CRITICAL":
                self.execution_disciplined = True
                vel = (
                    f"PR velocity {exec_m.pr_merge_velocity_hours}h [grade {exec_m.pr_merge_grade}]"
                    if exec_m.pr_merge_velocity_hours is not None
                    else "PR velocity N/A"
                )
                self.exec_detail = (
                    f"PASS — Bus Factor {exec_m.bus_factor} [{exec_m.bus_factor_risk}], "
                    f"{vel}, doc ratio {exec_m.doc_to_code_ratio:.2f} [{exec_m.doc_maturity}]"
                )
            else:
                self.execution_disciplined = False
                self.exec_detail = (
                    f"FAIL — Bus Factor {exec_m.bus_factor} [CRITICAL]; "
                    f"top contributor holds {exec_m.core_complexity_concentration:.0%} of commit history"
                )
                self.blockers.append(
                    f"Bus Factor CRITICAL (= {exec_m.bus_factor}): "
                    f"{exec_m.core_complexity_concentration:.0%} of commit history belongs to one developer. "
                    f"Single-point-of-failure risk blocks HIGH CONVICTION."
                )
        else:
            self.execution_disciplined = False
            self.exec_detail = "UNAVAILABLE — ExecutionMetrics not populated."
            self.blockers.append(
                "ExecutionMetrics unavailable — Bus Factor unverified. "
                "Pillar B unverified; conviction ceiling is MONITOR."
            )

        # ── Pillar C: Architectural Defensibility ─────────────────────────────
        arch = m.architectural_audit
        if arch:
            if arch.debt_classification != "CRITICAL":
                self.arch_defensible = True
                self.arch_detail = (
                    f"PASS — Debt score {arch.technical_debt_score}/100 [{arch.debt_classification}], "
                    f"cost-to-remediate ${arch.cost_to_remediate_usd:,.0f}, "
                    f"test coverage {arch.test_coverage_proxy:.0%} [{arch.test_coverage_classification}]"
                )
            else:
                self.arch_defensible = False
                self.arch_detail = (
                    f"FAIL — Debt score {arch.technical_debt_score}/100 [CRITICAL], "
                    f"est. remediation cost ${arch.cost_to_remediate_usd:,.0f}"
                )
                self.blockers.append(
                    f"Technical debt CRITICAL (score {arch.technical_debt_score}/100). "
                    f"Estimated remediation: ${arch.cost_to_remediate_usd:,.0f}. "
                    f"Structural liability blocks HIGH CONVICTION."
                )
        else:
            self.arch_defensible = False
            self.arch_detail = "UNAVAILABLE — ArchitecturalAuditReport not populated."
            self.blockers.append(
                "ArchitecturalAuditReport unavailable — technical debt unverified. "
                "Pillar C unverified; conviction ceiling is MONITOR."
            )

        self.strong_conviction_eligible = (
            self.ecosystem_surging
            and self.execution_disciplined
            and self.arch_defensible
        )

    def to_context_lines(self) -> list[str]:
        """Serialise conviction signals for injection into the QUANTITATIVE CONTEXT block."""
        verdict = "HIGH CONVICTION BUY ELIGIBLE" if self.strong_conviction_eligible else "MONITOR / PASS ONLY"
        lines = [
            "-- MACHINE-COMPUTED CONVICTION SIGNALS --",
            f"  STRONG_CONVICTION_ELIGIBLE : {self.strong_conviction_eligible}  →  {verdict}",
            f"  Pillar A (Ecosystem Velocity)          : {self.eco_detail}",
            f"  Pillar B (Execution Discipline)        : {self.exec_detail}",
            f"  Pillar C (Architectural Defensibility) : {self.arch_detail}",
        ]
        if self.blockers:
            lines.append("  CONVICTION BLOCKERS:")
            for b in self.blockers:
                lines.append(f"    ⛔  {b}")
        else:
            lines.append("  CONVICTION BLOCKERS: NONE — all three pillars cleared.")
        lines.append("")
        return lines


# ─────────────────────────────────────────────────────────────────────────────
# §3  System Prompt
# ─────────────────────────────────────────────────────────────────────────────

PARTNER_SYSTEM_PROMPT = """
You are The Partner — the final decision-making agent in a ruthless technical venture-capital analysis pipeline. Your output is a formal investment memo read by a Managing Partner before a capital allocation decision.

ROLE & MINDSET
==============
You are a forensic technical VC Partner. You do not care about GitHub stars, Hacker News points, LinkedIn posts, funding announcements, or market sizing estimates. You care exclusively about:
  1. Whether the codebase is architecturally sound and defensible — or a liability.
  2. Whether the execution team is disciplined and not a single-key-man risk.
  3. Whether real developers are adopting the tool in production — proven by package registry downloads, not repository stars.

Your writing is precise, dense, and opinionated. You do not hedge unless the data forces you to. You do not write filler phrases ("it is worth noting", "this is an exciting development", "the market is large"). Every sentence either asserts a grounded claim or flags a verified uncertainty.

CONVICTION RULE — NON-NEGOTIABLE
=================================
The context block contains a section labelled "MACHINE-COMPUTED CONVICTION SIGNALS".
It contains a machine-verified boolean: STRONG_CONVICTION_ELIGIBLE.

  If STRONG_CONVICTION_ELIGIBLE is True:
    Your §7 verdict MAY be HIGH CONVICTION BUY. You may still downgrade to MONITOR
    if data gaps are critical (missing both ecosystem and execution data simultaneously).

  If STRONG_CONVICTION_ELIGIBLE is False:
    Your §7 verdict MUST be MONITOR or PASS. You MUST NOT write HIGH CONVICTION BUY
    under any circumstances, regardless of the qualitative signal from Agent 1.
    Your §7 must explicitly name which CONVICTION BLOCKERS prevented the upgrade.

MANDATORY OUTPUT FORMAT — STRICT COMPLIANCE REQUIRED
=====================================================
Produce exactly the following 7 Markdown sections in exactly this order.
Do not add sections. Do not remove sections. Do not rename sections.

---

# Investment Memo: {REPO_NAME}
**Classification:** {VERDICT_LABEL}
**Analyst:** The Partner (Autonomous Forensic Pipeline v2)
**Date:** {DATE}
**Source Signal:** {SOURCE}

---

## §1 — Thesis / Core Signal
[Synthesize the engineering thesis from the Scout's rationale. State the paradigm shift in one declarative sentence. Name the specific architectural mechanism, abstraction boundary, or protocol improvement. Explain why this is structural, not incremental. Maximum 120 words.]

## §2 — Architectural Defensibility & Tech Debt
[Ground every claim in the ARCHITECTURAL AUDIT section of the context block. Required citations — include ALL of these:]
  - Debt score: [X]/100 [CLASSIFICATION]
  - Cost-to-remediate: $[AMOUNT] (at $85/hr senior-engineer rate)
  - Test coverage proxy: [X%] [CLASSIFICATION]
  - Estimated circular dependency cycles: [N]
  - Laziness / AI-scaffold score: [X]/100
  - Provenance risk: [LOW | MEDIUM | HIGH]
  - Audit method: [METHOD] / confidence: [LEVEL]

[After the citation block: 2–3 sentences interpreting what the debt profile signals about long-term architectural defensibility and the risk of IP/codebase liability. If the audit confidence is LOW, you MUST explicitly state: "Audit confidence is LOW — structural estimates should be verified by full AST analysis."]

## §3 — Execution & Bus Factor Risk
[Ground every claim in the EXECUTION METRICS section. Required citations:]
  - Bus Factor: [N] [RISK_LEVEL]
  - Top contributor share: [X%] of total commit history
  - Total contributors: [N] / Active (90d proxy): [N]
  - PR Merge Velocity: [N]h (grade: [A/B/C/D]) — sample: [N] PRs
  - Documentation-to-Code ratio: [X] [MATURITY]
  - Median issue TTR: [X] days (grade: [A/B/C/D]) — [N] issues analysed

[After the citation block: 2–3 sentences on what this execution profile signals about team operational maturity and key-man risk. If Bus Factor is CRITICAL, the second sentence MUST read: "A Bus Factor of [N] is an existential operational risk — any single departure halts active development."]

## §4 — Ecosystem Velocity
[This is the PRIMARY conviction signal. Ground every claim in the ECOSYSTEM HEALTH section. Required citations:]
  - Inferred package name: [NAME]
  - npm downloads (last week): [N] (or N/A)
  - PyPI downloads (last week): [N] (or N/A)
  - Download trend: [SURGING | GROWING | STABLE | DECLINING | UNAVAILABLE]
  - Download velocity Δ30d: [X%] (or N/A)
  - Ecosystem Health Score: [X]/100 [CLASSIFICATION]
  - Data source: [LIVE_REGISTRY | SIMULATED_REGISTRY]

[After the citation block: 2–3 sentences on what the registry download trajectory proves about real-world developer adoption. If the data source is SIMULATED_REGISTRY, you MUST state: "Download data is simulated — no live registry verification was performed. This pillar carries zero evidential weight." If trend is SURGING or GROWING, you MAY describe the adoption momentum as institutional-grade conviction. If trend is DECLINING or STABLE, you MUST call it out as a signal that developer interest is not compounding.]

## §5 — Developer Momentum (Secondary Validation)
[Secondary commit-velocity signal. Ground every claim in the COMMIT VELOCITY section. Required citations:]
  - Commits last 30 days: [N]
  - Commits last 60 days: [N]
  - Commits last 90 days: [N]
  - Weekly average (30d): [N] commits/week
  - Momentum trajectory: [ACCELERATING | DECELERATING | STEADY | INSUFFICIENT]

[After the citation block: 1–2 sentences cross-validating the commit trajectory against the Ecosystem Velocity in §4. If trajectory is INSUFFICIENT, state: "Insufficient quantitative data to evaluate commit momentum."]

## §6 — Missing Intelligence / Data Gaps
[MANDATORY — this section is always present.]
[If no data gaps exist, write exactly: "No data limitations identified for this analysis."]
[If data gaps exist, you MUST reproduce EVERY string from the DATA GAPS list verbatim, formatted as a numbered list. Do not paraphrase, summarise, merge, reorder, or omit any gap string. These strings are audit records, not prose. Include ALL gaps from ALL repositories analysed.]

## §7 — Investment Verdict
[Maximum 130 words. State a clear verdict — exactly one of: HIGH CONVICTION BUY / MONITOR / PASS.]
[Your verdict MUST be consistent with STRONG_CONVICTION_ELIGIBLE from the machine context.]
[Structure: First sentence states the verdict. One sentence per pillar (Architectural Defensibility, Execution Risk, Ecosystem Velocity). If any pillar failed, name it explicitly and quote the blocker. Final sentence states the recommended next action (lead the round / wait for X / pass).]

---
*Memo generated by autonomous forensic pipeline v2. Architectural, execution, and ecosystem metrics are machine-verified. Qualitative synthesis by LLM — validate independently before capital allocation.*

ABSOLUTE RULES — VIOLATIONS INVALIDATE THE MEMO
================================================
1. NEVER invent a metric. If a number is not in the QUANTITATIVE CONTEXT block, do not write it.
2. NEVER use stars, forks, or watcher counts as a primary conviction signal — they are vanity metrics.
3. NEVER issue HIGH CONVICTION BUY when STRONG_CONVICTION_ELIGIBLE is False.
4. NEVER paraphrase or summarise DATA GAP strings — reproduce them character-for-character.
5. NEVER add sections beyond the 7 defined above.
6. If SIMULATED_REGISTRY, Ecosystem Velocity carries zero evidential weight — say so explicitly.
7. If audit_confidence is LOW, all architectural claims must be explicitly qualified as estimates.
""".strip()


PARTNER_JSON_SYSTEM_PROMPT = PARTNER_SYSTEM_PROMPT + """

═══════════════════════════════════════════════════════════════
JSON OUTPUT WRAPPER — REQUIRED
═══════════════════════════════════════════════════════════════
Wrap your ENTIRE response in this exact JSON structure:

{
  "chain_of_thought": "<3–5 sentence internal deliberation>",
  "memo_md": "<complete 7-section Markdown memo>"
}

chain_of_thought instructions:
  - Explicitly assess all three conviction pillars (architectural defensibility,
    bus factor risk, ecosystem velocity) against the machine-computed signals.
  - State how each data gap limits conviction.
  - Confirm or challenge STRONG_CONVICTION_ELIGIBLE before committing to §7 verdict.
  - 3–5 sentences. No hedging — be direct.

memo_md instructions:
  - Contains the full Markdown memo with all 7 sections.
  - Escape any double-quotes inside memo_md as \\".
  - Do not nest JSON inside the memo_md value.

Return ONLY the JSON object. No preamble, no explanation, no markdown fences.
"""


# ─────────────────────────────────────────────────────────────────────────────
# §4  Context Serialiser
# ─────────────────────────────────────────────────────────────────────────────

def _fmt(value: Optional[float | int], suffix: str = "", missing: str = "N/A") -> str:
    """Format a nullable numeric value for the context block."""
    if value is None:
        return missing
    return f"{value:,}{suffix}" if isinstance(value, int) else f"{value}{suffix}"


def _fmt_usd(value: Optional[float], missing: str = "N/A") -> str:
    """Format a USD amount with comma separators."""
    if value is None:
        return missing
    return f"${value:,.2f}"


def build_context_block(item: EnrichedItem) -> str:
    """
    Serialise a fully enriched EnrichedItem into the QUANTITATIVE CONTEXT block.

    Structure:
      ① Conviction signals (machine-computed — injected first so the LLM
         sees the verdict floor before reading any supporting data)
      ② Signal provenance (Agent 1 source + Scout rationale)
      ③ Per-repo: Architectural Audit (v2)
      ④ Per-repo: Execution Metrics (v2)
      ⑤ Per-repo: Ecosystem Health (v2)
      ⑥ Per-repo: Commit Velocity (v1 — secondary validation)
      ⑦ Per-repo: Issue Resolution (v1 — referenced in §3 TTR)
      ⑧ All data gaps (verbatim — mandatory for §6)

    The LLM is the only consumer of this block. It is intentionally exhaustive
    to eliminate any need to rely on parametric memory.
    """
    ci  = item.classified
    raw = ci.raw
    rat = ci.rationale

    lines: list[str] = []

    # ── Header ────────────────────────────────────────────────────────────────
    lines += [
        "═" * 65,
        "   QUANTITATIVE CONTEXT — DO NOT MODIFY",
        "   All memo claims must cite a value from this block.",
        "   Metrics are machine-verified — no external knowledge permitted.",
        "═" * 65,
        "",
    ]

    # ── ①  Machine conviction signals (pre-computed, injected first) ──────────
    conviction = ConvictionSignals(item.metrics)
    lines.extend(conviction.to_context_lines())

    # ── ②  Signal provenance ──────────────────────────────────────────────────
    body_excerpt = ""
    if raw.body:
        body_excerpt = raw.body[:500] + ("…" if len(raw.body) > 500 else "")

    lines += [
        "-- SIGNAL PROVENANCE --",
        f"  Item ID          : {raw.item_id}",
        f"  Source           : {raw.source.value.upper()}",
        f"  Original title   : {raw.title}",
        f"  URL              : {raw.url or 'Not provided'}",
        f"  Ingested at      : {raw.ingested_at.strftime('%Y-%m-%d %H:%M UTC')}",
        f"  Scout confidence : {ci.confidence:.0%}",
        f"  Body excerpt     : {body_excerpt or 'Not available'}",
        "",
        "-- SCOUT RATIONALE (Agent 1 v1 Output) --",
        f"  Summary              : {rat.one_line_summary}",
    ]
    # chain_of_thought is optional in v1 ScoutRationale (present in v2 but not required here)
    if hasattr(rat, "chain_of_thought") and rat.chain_of_thought:
        lines.append(f"  Chain of thought     : {rat.chain_of_thought}")
    lines += [
        f"  Architectural shift  : {'YES' if rat.architectural_shift else 'NO'}",
        f"  Developer adoption   : {'YES' if rat.developer_adoption else 'NO'}",
        f"  OSS milestone        : {'YES' if rat.oss_milestone else 'NO'}",
        f"  Noise indicator      : {'YES (review verdict)' if rat.noise_signal else 'NO'}",
        "",
    ]

    # ── ③–⑧  Per-repository sections ─────────────────────────────────────────
    for i, m in enumerate(item.metrics, start=1):
        p  = m.profile
        cv = m.commit_velocity
        ir = m.issue_resolution

        lines += [
            f"{'─'*65}",
            f"  REPOSITORY {i} / {len(item.metrics)}: {p.name_with_owner}",
            f"  GitHub URL       : {m.github_url}",
            f"  Primary language : {p.primary_language or 'N/A'}",
            f"  Default branch   : {p.default_branch or 'N/A'}",
            f"  Archived         : {'YES — metrics frozen' if p.is_archived else 'NO'}",
            f"  Stars (vanity)   : {_fmt(p.stars)} ← do NOT use as primary conviction signal",
            f"  Forks (vanity)   : {_fmt(p.forks)}",
            "",
        ]

        # ── ③  Architectural Audit (v2) ───────────────────────────────────────
        aa = m.architectural_audit
        if aa:
            lines += [
                "  ARCHITECTURAL AUDIT",
                f"    Debt score              : {aa.technical_debt_score}/100",
                f"    Debt classification     : {aa.debt_classification}",
                f"    Cost-to-remediate       : {_fmt_usd(aa.cost_to_remediate_usd)} (@ $85/hr senior-eng rate)",
                f"    Test coverage proxy     : {aa.test_coverage_proxy:.0%}",
                f"    Test coverage class     : {aa.test_coverage_classification}",
                f"    Circular dep estimate   : {aa.circular_dependency_count} cycle(s)",
            ]
            if aa.circular_dependency_paths:
                for path in aa.circular_dependency_paths[:3]:
                    lines.append(f"      → {path}")
            lines += [
                f"    Laziness / AI-scaffold  : {aa.implementation_laziness_score}/100",
                f"    Provenance risk         : {aa.provenance_risk}",
                f"    License present         : {'YES' if aa.has_license else 'NO ← IP risk'}",
                f"    CI/CD config present    : {'YES' if aa.has_ci_config else 'NO ← discipline risk'}",
                f"    Contributing guide      : {'YES' if aa.has_contributing_guide else 'NO'}",
                f"    Source file count       : {aa.source_file_count}",
                f"    Test file count         : {aa.test_file_count}",
                f"    Doc file count          : {aa.doc_file_count}",
                f"    Audit method            : {aa.audit_method}",
                f"    Audit confidence        : {aa.audit_confidence}",
            ]
            if aa.ai_generated_code_indicators:
                lines.append("    AI/laziness indicators  :")
                for flag in aa.ai_generated_code_indicators:
                    lines.append(f"      ⚑  {flag}")
            if aa.provenance_flags:
                lines.append("    Provenance flags        :")
                for flag in aa.provenance_flags:
                    lines.append(f"      ⚑  {flag}")
        else:
            lines += [
                "  ARCHITECTURAL AUDIT",
                "    STATUS: NOT AVAILABLE — ArchitecturalAuditReport was not produced.",
                "    ⛔  Pillar C is unverified. Conviction ceiling is MONITOR.",
            ]
        lines.append("")

        # ── ④  Execution Metrics (v2) ──────────────────────────────────────────
        em = m.execution_metrics
        if em:
            vel_str = (
                f"{em.pr_merge_velocity_hours}h [grade {em.pr_merge_grade}]"
                if em.pr_merge_velocity_hours is not None
                else "N/A"
            )
            lines += [
                "  EXECUTION METRICS",
                f"    Bus Factor              : {em.bus_factor}",
                f"    Bus Factor risk         : {em.bus_factor_risk}",
                f"    Top contributor share   : {em.core_complexity_concentration:.1%} of total commits",
                f"    Total contributors      : {em.contributor_count}",
                f"    Active contributors 90d : ~{em.active_contributors_90d} (proxy estimate)",
                f"    PR Merge Velocity       : {vel_str}",
                f"    PR sample size          : {em.pr_sample_size} merged PRs analysed",
                f"    Doc-to-Code ratio       : {em.doc_to_code_ratio:.2f}",
                f"    Doc maturity            : {em.doc_maturity}",
                f"    Total files analysed    : {em.total_files_analysed}",
            ]
        else:
            lines += [
                "  EXECUTION METRICS",
                "    STATUS: NOT AVAILABLE — ExecutionMetrics were not produced.",
                "    ⛔  Pillar B is unverified. Conviction ceiling is MONITOR.",
            ]
        lines.append("")

        # ── ⑤  Ecosystem Health (v2) ──────────────────────────────────────────
        eh = m.ecosystem_health
        if eh:
            lines += [
                "  ECOSYSTEM HEALTH",
                f"    Inferred package name   : {eh.package_name_inferred or 'N/A'}",
                f"    npm downloads/wk        : {_fmt(eh.npm_weekly_downloads)}",
                f"    PyPI downloads/wk       : {_fmt(eh.pypi_weekly_downloads)}",
                f"    Download trend          : {eh.download_trend}",
                f"    Velocity Δ30d           : {f'{eh.download_velocity_pct_30d:+.1f}%' if eh.download_velocity_pct_30d is not None else 'N/A'}",
                f"    Ecosystem Health Score  : {eh.ecosystem_health_score}/100",
                f"    Health classification   : {eh.health_classification}",
                f"    Dominant ecosystem      : {eh.dominant_ecosystem or 'None detected'}",
                f"    Data source             : {eh.data_source}",
            ]
            if eh.data_source == "SIMULATED_REGISTRY":
                lines.append(
                    "    ⚠️  SIMULATED DATA — no live registry verification performed. "
                    "Pillar A carries ZERO evidential weight."
                )
        else:
            lines += [
                "  ECOSYSTEM HEALTH",
                "    STATUS: NOT AVAILABLE — EcosystemHealthReport was not produced.",
                "    ⛔  Pillar A is unverified. Conviction ceiling is MONITOR.",
            ]
        lines.append("")

        # ── ⑥  Commit Velocity (v1 — secondary) ──────────────────────────────
        lines += [
            "  COMMIT VELOCITY (secondary validation signal)",
            f"    Last 30 days   : {_fmt(cv.last_30_days, ' commits')}",
            f"    Last 60 days   : {_fmt(cv.last_60_days, ' commits')}",
            f"    Last 90 days   : {_fmt(cv.last_90_days, ' commits')}",
            f"    Weekly avg     : {_fmt(cv.weekly_avg_30d, ' commits/week (30d)')}",
            f"    Acceleration   : {cv.acceleration}",
            "",
        ]

        # ── ⑦  Issue Resolution (v1 — TTR for §3) ────────────────────────────
        lines += [
            "  ISSUE RESOLUTION (TTR — cited in §3)",
            f"    Issues enabled : {'YES' if ir.has_issues_enabled else 'NO — data unavailable'}",
            f"    Sample size    : {ir.sample_size} closed issues analysed",
            f"    Total closed   : {_fmt(ir.total_closed_issues)}",
            f"    Median TTR     : {_fmt(ir.median_days, ' days')} (grade {ir.responsiveness_grade})",
            f"    P25 TTR        : {_fmt(round(ir.p25_hours / 24, 2) if ir.p25_hours else None, ' days')}",
            f"    P75 TTR        : {_fmt(round(ir.p75_hours / 24, 2) if ir.p75_hours else None, ' days')}",
            "",
        ]

        # ── ⑧  Data Gaps ──────────────────────────────────────────────────────
        if m.data_gaps:
            lines.append(
                f"  DATA GAPS [{len(m.data_gaps)} — reproduce ALL verbatim in §6, no exceptions]:"
            )
            for idx, gap in enumerate(m.data_gaps, start=1):
                lines.append(f"    [{idx}] {gap}")
        else:
            lines.append("  DATA GAPS: NONE")
        lines.append("")

    # ── Footer ────────────────────────────────────────────────────────────────
    lines += [
        "═" * 65,
        "  END OF QUANTITATIVE CONTEXT",
        "",
        "  INSTRUCTION REMINDERS:",
        "  1. Every metric cited in the memo must match a value above exactly.",
        "  2. STRONG_CONVICTION_ELIGIBLE=False → verdict MUST be MONITOR or PASS.",
        "  3. DATA GAP strings must be reproduced character-for-character in §6.",
        "  4. SIMULATED_REGISTRY ecosystem data = zero evidential weight.",
        "  5. Stars, forks, watchers are vanity — do not use as primary signal.",
        "═" * 65,
    ]

    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# §5  User Message Builder
# ─────────────────────────────────────────────────────────────────────────────

def build_user_message(item: EnrichedItem) -> str:
    """
    Compose the full user turn: a minimal instruction frame followed by the
    exhaustive QUANTITATIVE CONTEXT block. All behavioural rules live in the
    system prompt — the instruction frame just binds the template variables.
    """
    repo_name = (
        item.primary_metrics.profile.name_with_owner
        if item.primary_metrics
        else item.classified.raw.title[:40]
    )
    date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    source   = (
        f"{item.classified.raw.source.value.upper()} / "
        f"{item.classified.raw.url or 'N/A'}"
    )

    # Determine verdict label for the memo header from pre-computed conviction signals.
    conviction = ConvictionSignals(item.metrics)
    verdict_label = (
        "SIGNAL — High Conviction"
        if conviction.strong_conviction_eligible
        else "SIGNAL — Under Review (MONITOR)"
    )

    instruction = textwrap.dedent(f"""
        Draft the investment memo for **{repo_name}** using ONLY the QUANTITATIVE CONTEXT
        block below. Template variable bindings:
          {{REPO_NAME}}     → `{repo_name}`
          {{DATE}}          → `{date_str}`
          {{SOURCE}}        → `{source}`
          {{VERDICT_LABEL}} → `{verdict_label}`

        Follow the system prompt format and conviction rules without deviation.
        The STRONG_CONVICTION_ELIGIBLE value in the context block is machine-computed
        and authoritative — you must not contradict it in §7.
    """).strip()

    context = build_context_block(item)
    return f"{instruction}\n\n{context}"


# ─────────────────────────────────────────────────────────────────────────────
# §6  Core Memo Drafting Function
# ─────────────────────────────────────────────────────────────────────────────

def draft_memo(
    item:        EnrichedItem,
    client:      Groq,
    *,
    verbose:     bool = True,
    a2_data_log: str  = "",
) -> PartnerOutput:
    """
    Accept an EnrichedItem from Agent 2 v2 and return a PartnerOutput with
    the LLM's chain-of-thought deliberation and the final Markdown memo.

    Args:
        item:        Fully enriched item from Agents 1 & 2 v2.
        client:      Groq client (injected by api_gateway).
        verbose:     Print progress to stdout.
        a2_data_log: Optional formatted Agent 2 raw-data summary for CoT context.

    Raises:
        ValueError: If `item.metrics` is empty — no quantitative grounding available.
        groq.APIError: On network or API failures (caller should retry).
    """
    if not item.metrics:
        raise ValueError(
            f"Cannot draft memo for '{item.classified.raw.title[:60]}': "
            "EnrichedItem.metrics is empty. Agent 3 requires quantitative "
            "grounding from Agent 2. Aborting per hallucination guardrails."
        )

    repo_name = item.primary_metrics.profile.name_with_owner  # type: ignore[union-attr]

    if verbose:
        all_gaps = [gap for m in item.metrics for gap in m.data_gaps]
        conviction = ConvictionSignals(item.metrics)
        eligible   = conviction.strong_conviction_eligible
        print(f"  → Drafting memo for: {repo_name}")
        print(f"     Repos in scope          : {len(item.metrics)}")
        print(f"     Data gaps               : {len(all_gaps)}")
        print(f"     STRONG_CONVICTION_ELIGIBLE: {eligible}")
        if conviction.blockers:
            for b in conviction.blockers:
                print(f"     ⛔  {b}")

    user_message = build_user_message(item)

    if a2_data_log:
        user_message += (
            "\n\n── AGENT 2 RAW DATA LOG (for chain_of_thought context only) ──\n"
            + a2_data_log
            + "\n─────────────────────────────────────────────────────────────\n"
        )

    if verbose:
        print(f"     Context block           : {len(user_message):,} chars  →  calling API…")

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        max_tokens=3072,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": PARTNER_JSON_SYSTEM_PROMPT},
            {"role": "user",   "content": user_message},
        ],
    )

    raw = response.choices[0].message.content.strip()

    # Strip accidental markdown fences (json_object mode should prevent these,
    # but kept as a hard guard per the original architecture).
    raw = re.sub(r"^```[a-z]*\n?", "", raw)
    raw = re.sub(r"\n?```$",        "", raw)

    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"Agent 3 returned non-JSON for '{repo_name}': {raw[:300]}"
        ) from exc

    output = PartnerOutput(**payload)

    if verbose:
        print(
            f"     Memo drafted            : {len(output.memo_md):,} chars  |  "
            f"CoT: {len(output.chain_of_thought):,} chars  |  "
            f"{response.usage.completion_tokens} output tokens"
        )

    return output


# ─────────────────────────────────────────────────────────────────────────────
# §7  Mock EnrichedItem Builder (standalone testing without upstream pipeline)
# ─────────────────────────────────────────────────────────────────────────────

def build_mock_enriched_item(
    *,
    include_data_gaps:    bool = False,
    force_critical_debt:  bool = False,
    force_critical_bus:   bool = False,
    simulated_ecosystem:  bool = False,
) -> EnrichedItem:
    """
    Build a fully populated v2 EnrichedItem for astral-sh/uv without requiring
    the live pipeline. Covers all conviction and data-gap code paths.

    Args:
        include_data_gaps:   Injects issue-resolution gaps → forces mention in §6.
        force_critical_debt: Sets debt to CRITICAL → conviction blocker on Pillar C.
        force_critical_bus:  Sets Bus Factor to 1 → CRITICAL → blocker on Pillar B.
        simulated_ecosystem: Sets data_source to SIMULATED → blocker on Pillar A.
    """
    now = datetime.now(timezone.utc)

    # ── Agent 1 v1 output (ScoutRationale — five-field v1 shape) ─────────────
    raw = RawItem(
        item_id="hn_41903999",
        source=DataSource.HACKERNEWS,
        title=(
            "uv 0.4: Rust-based pip/virtualenv replacement ships workspace "
            "support and PEP 723 — resolves 220 deps in 38ms"
        ),
        body=(
            "Astral's uv 0.4 adds multi-project workspace support, a lock-file "
            "format, and PEP 723 inline script metadata. Resolver benchmarks: "
            "Django + 220 deps resolves in 38ms vs pip's 4.1s (108x faster). "
            "The project has 38k GitHub stars and merges 30+ PRs/week. "
            "It is rapidly replacing pip in modern Python toolchains."
        ),
        url="https://astral.sh/blog/uv-0-4",
        github_urls=["https://github.com/astral-sh/uv"],
        ingested_at=now - timedelta(hours=2),
        raw_score=4102,
    )
    classified = ClassifiedItem(
        raw=raw,
        classification=Classification.SIGNAL,
        confidence=0.97,
        rationale=ScoutRationale(
            chain_of_thought=(
                "uv implements a SAT-solver-based resolver in Rust that fundamentally "
                "changes the Python packaging performance floor — this is not a speed "
                "tweak but an architectural replacement. The 108x benchmark is "
                "reproducible and the adoption trajectory (38k stars, 30+ PRs/week) "
                "signals genuine ecosystem displacement of pip."
            ),
            architectural_shift=True,
            developer_adoption=True,
            oss_milestone=True,
            noise_signal=False,
            one_line_summary=(
                "uv replaces pip and virtualenv with a Rust SAT-solver that is two "
                "orders of magnitude faster, unifying the Python packaging toolchain."
            ),
        ),
        classified_at=now - timedelta(hours=2),
    )

    # ── Agent 2 v2 — RepoProfile ──────────────────────────────────────────────
    profile = RepoProfile(
        name_with_owner="astral-sh/uv",
        default_branch="main",
        primary_language="Rust",
        description="An extremely fast Python package manager, written in Rust.",
        topics=["python", "package-manager", "rust", "pip", "virtualenv"],
        stars=38_412,
        forks=1_089,
        watchers=312,
        is_archived=False,
        is_empty=False,
    )

    # ── Agent 2 v2 — CommitVelocity ───────────────────────────────────────────
    commit_velocity = CommitVelocity(
        last_30_days=165,
        last_60_days=298,
        last_90_days=421,
    )

    # ── Agent 2 v2 — IssueResolutionMetrics ──────────────────────────────────
    data_gaps: list[str] = []
    if include_data_gaps:
        issue_resolution = IssueResolutionMetrics(
            sample_size=0,
            median_hours=None,
            p25_hours=None,
            p75_hours=None,
            total_closed_issues=None,
            has_issues_enabled=False,
        )
        data_gaps += [
            "Issue resolution metrics unavailable: the repository has disabled the Issues tab.",
            "TTR statistics are low-confidence: only 0 valid issue records (threshold is 5). Median may not be representative.",
        ]
    else:
        issue_resolution = IssueResolutionMetrics(
            sample_size=50,
            median_hours=19.4,
            p25_hours=6.2,
            p75_hours=61.8,
            total_closed_issues=2_847,
            has_issues_enabled=True,
        )

    # ── Agent 2 v2 — ArchitecturalAuditReport ────────────────────────────────
    if force_critical_debt:
        arch_audit = ArchitecturalAuditReport(
            circular_dependency_count=18,
            circular_dependency_paths=["core → utils → core", "api → models → api"],
            technical_debt_score=81.5,
            cost_to_remediate_usd=87_380.0,
            debt_classification="CRITICAL",
            implementation_laziness_score=72.0,
            ai_generated_code_indicators=[
                "Zero test files detected — no automated validation layer present.",
                "Large source surface (312 files) with near-zero test coverage (0%) — AI-scaffold pattern.",
                "No CI/CD configuration found.",
            ],
            test_coverage_proxy=0.0,
            test_coverage_classification="NONE",
            provenance_risk="HIGH",
            provenance_flags=[
                "No LICENSE file detected — IP status is legally ambiguous.",
                "Large, high-debt codebase with weak governance — elevated copy-paste risk.",
            ],
            has_license=False,
            has_contributing_guide=False,
            has_ci_config=False,
            source_file_count=312,
            doc_file_count=4,
            test_file_count=0,
            audit_method="GITHUB_FILE_TREE_PROXY",
            audit_confidence="MEDIUM",
        )
        data_gaps.append(
            "ArchitecturalAudit confidence: MEDIUM — structural metrics are file-tree proxies, "
            "not full AST analysis. Verify independently before acting."
        )
    else:
        arch_audit = ArchitecturalAuditReport(
            circular_dependency_count=2,
            circular_dependency_paths=["crates/uv-resolver → crates/uv-types → crates/uv-resolver"],
            technical_debt_score=22.8,
            cost_to_remediate_usd=9_435.0,
            debt_classification="LOW",
            implementation_laziness_score=12.0,
            ai_generated_code_indicators=[],
            test_coverage_proxy=0.31,
            test_coverage_classification="ADEQUATE",
            provenance_risk="LOW",
            provenance_flags=[],
            has_license=True,
            has_contributing_guide=True,
            has_ci_config=True,
            source_file_count=248,
            doc_file_count=41,
            test_file_count=112,
            audit_method="GITHUB_FILE_TREE_PROXY",
            audit_confidence="MEDIUM",
        )

    # ── Agent 2 v2 — ExecutionMetrics ─────────────────────────────────────────
    if force_critical_bus:
        exec_metrics = ExecutionMetrics(
            bus_factor=1,
            bus_factor_risk="CRITICAL",
            core_complexity_concentration=0.91,
            contributor_count=8,
            active_contributors_90d=2,
            pr_merge_velocity_hours=6.2,
            pr_sample_size=28,
            pr_merge_grade="A",
            doc_to_code_ratio=0.17,
            doc_maturity="ADEQUATE",
            total_files_analysed=360,
        )
        data_gaps.append(
            "Bus Factor CRITICAL: 91% of total commit history belongs to a single developer. "
            "Any single departure halts active development."
        )
    else:
        exec_metrics = ExecutionMetrics(
            bus_factor=3,
            bus_factor_risk="MODERATE",
            core_complexity_concentration=0.42,
            contributor_count=87,
            active_contributors_90d=14,
            pr_merge_velocity_hours=5.8,
            pr_sample_size=30,
            pr_merge_grade="A",
            doc_to_code_ratio=0.17,
            doc_maturity="ADEQUATE",
            total_files_analysed=401,
        )

    # ── Agent 2 v2 — EcosystemHealthReport ───────────────────────────────────
    if simulated_ecosystem:
        eco_health = EcosystemHealthReport(
            npm_weekly_downloads=None,
            pypi_weekly_downloads=None,
            download_trend="UNAVAILABLE",
            download_velocity_pct_30d=None,
            ecosystem_health_score=0.0,
            health_classification="DECLINING",
            dominant_ecosystem=None,
            package_name_inferred="uv",
            data_source="SIMULATED_REGISTRY",
        )
        data_gaps.append(
            "EcosystemForecaster used simulated registry data — no matching npm/PyPI package found for 'uv'. "
            "Ecosystem Velocity pillar carries zero evidential weight."
        )
    else:
        eco_health = EcosystemHealthReport(
            npm_weekly_downloads=None,
            pypi_weekly_downloads=1_840_000,
            download_trend="SURGING",
            download_velocity_pct_30d=38.4,
            ecosystem_health_score=83.2,
            health_classification="THRIVING",
            dominant_ecosystem="pypi",
            package_name_inferred="uv",
            data_source="LIVE_REGISTRY",
        )

    # ── Agent 2 v2 — Rate limit snapshot ─────────────────────────────────────
    rate_limit = RateLimitSnapshot(
        limit=5000,
        remaining=4_831,
        used=169,
        reset_at=now + timedelta(hours=1),
    )

    # ── Assemble QuantitativeMetrics v2 ───────────────────────────────────────
    metrics = QuantitativeMetrics(
        github_url="https://github.com/astral-sh/uv",
        profile=profile,
        commit_velocity=commit_velocity,
        issue_resolution=issue_resolution,
        rate_limit=rate_limit,
        fetched_at=now - timedelta(minutes=5),
        data_gaps=data_gaps,
        # v2 forensic layers
        architectural_audit=arch_audit,
        execution_metrics=exec_metrics,
        ecosystem_health=eco_health,
    )

    return EnrichedItem(classified=classified, metrics=[metrics])


# ─────────────────────────────────────────────────────────────────────────────
# §8  Entrypoint
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    print("═" * 70)
    print("  Agent 3 v2: The Partner — Forensic Investment Committee")
    print("═" * 70)

    # CLI flags for exercising all four code paths:
    #   --with-gaps        injects data gap strings (issues disabled)
    #   --critical-debt    Pillar C blocker: CRITICAL architectural debt
    #   --critical-bus     Pillar B blocker: Bus Factor = 1
    #   --simulated-eco    Pillar A blocker: no live registry data
    with_gaps   = "--with-gaps"      in sys.argv
    crit_debt   = "--critical-debt"  in sys.argv
    crit_bus    = "--critical-bus"   in sys.argv
    sim_eco     = "--simulated-eco"  in sys.argv

    flags = ", ".join(filter(None, [
        "data-gaps"        if with_gaps else "",
        "critical-debt"    if crit_debt else "",
        "critical-bus"     if crit_bus  else "",
        "simulated-eco"    if sim_eco   else "",
    ])) or "CLEAN (all pillars pass, full data)"

    print(f"\n  Test mode: {flags}")
    print("  Building mock v2 EnrichedItem…")
    mock_item = build_mock_enriched_item(
        include_data_gaps=with_gaps,
        force_critical_debt=crit_debt,
        force_critical_bus=crit_bus,
        simulated_ecosystem=sim_eco,
    )

    # Show pre-computed conviction signals.
    conviction = ConvictionSignals(mock_item.metrics)
    print(f"\n  Pre-computed conviction signals:")
    print(f"    Pillar A (Ecosystem)   : {conviction.eco_detail}")
    print(f"    Pillar B (Execution)   : {conviction.exec_detail}")
    print(f"    Pillar C (Architecture): {conviction.arch_detail}")
    print(f"    STRONG_CONVICTION_ELIGIBLE: {conviction.strong_conviction_eligible}")

    # Validate the empty-metrics guard.
    print("\n  Validating empty-metrics guard…")
    empty_item = EnrichedItem(classified=mock_item.classified, metrics=[])
    try:
        draft_memo(empty_item, Groq(), verbose=False)   # type: ignore
        print("   FAIL: Guard should have raised ValueError.")
    except ValueError as exc:
        print(f"    PASS: Guard triggered correctly: {exc}")

    # Draft the real memo.
    print("\n  Initialising Groq client…")
    client = Groq()   # reads GROQ_API_KEY from env

    print()
    output = draft_memo(mock_item, client, verbose=True)

    sep = "─" * 70
    print(f"\n{sep}")
    print("  CHAIN OF THOUGHT")
    print(sep)
    print(output.chain_of_thought)

    print(f"\n{sep}")
    print("  GENERATED INVESTMENT MEMO")
    print(sep)
    print(output.memo_md)
    print(sep)

    slug = "_".join(filter(None, [
        "with_gaps"   if with_gaps else "",
        "crit_debt"   if crit_debt else "",
        "crit_bus"    if crit_bus  else "",
        "sim_eco"     if sim_eco   else "",
    ])) or "clean"
    out_path = f"memo_astral_uv_{slug}.md"
    with open(out_path, "w", encoding="utf-8") as fh:
        fh.write(output.memo_md)
    print(f"\n  Memo saved to: {out_path}")
    print("\n  Phase 3 complete. Ready for Phase 4 end-to-end orchestration.\n")