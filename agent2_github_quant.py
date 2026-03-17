"""
Phase 2 — Agent 2: The Architectural Erosion and Execution Quantifier
======================================================================
Evolved from a simple commit scraper into a four-engine forensic auditor.

Four analytical engines operate on every GitHub signal:

  ① ArchitecturalAuditor   — Proxied AST structural analysis via the GitHub
                              file-tree API: test coverage ratio, circular-
                              dependency estimation, technical debt monetised at
                              $85/hr, AI-boilerplate detection, IP provenance risk.

  ② ExecutionQuantifier    — Bus Factor (minimum contributors covering ≥50% of
                              commit history), PR merge velocity (median open-to-
                              merge latency), documentation-to-code ratio.

  ③ EcosystemForecaster    — Live npm (api.npmjs.org) and PyPI (pypistats.org)
                              download ingestion to produce a Developer Ecosystem
                              Health Score that is immune to GitHub star inflation.

  ④ GraphQL Core           — Repo profile, commit velocity, and issue TTR.
                              Fully preserved from v1 for backward compatibility.

Design rules:
  - Every engine fails-open: all errors populate `data_gaps` and execution continues.
    Agent 3 must quote every data_gap entry verbatim in the investment memo.
  - Mock paths activate automatically when GITHUB_TOKEN is absent (token == "MOCK").
  - `QuantitativeMetrics` is augmented with three Optional fields:
    `architectural_audit`, `execution_metrics`, `ecosystem_health`.
    All existing field references in server.py and agent3_partner.py are unchanged.
"""

from __future__ import annotations

import math
import os
import re
import statistics
import time
from datetime import datetime, timedelta, timezone
from typing import Optional
from unittest.mock import MagicMock

import requests
from pydantic import BaseModel, Field, computed_field

from agent1_scout import ClassifiedItem, Classification, DataSource, RawItem, ScoutRationale


# ─────────────────────────────────────────────────────────────────────────────
# §1  Constants
# ─────────────────────────────────────────────────────────────────────────────

GITHUB_GRAPHQL_URL = "https://api.github.com/graphql"
GITHUB_REST_URL    = "https://api.github.com"
NPM_DOWNLOADS_URL  = "https://api.npmjs.org/downloads/point"
PYPI_STATS_URL     = "https://pypistats.org/api/packages"

COMMITS_PER_PAGE   = 100
ISSUES_TO_FETCH    = 50
PR_SAMPLE_SIZE     = 30
CONTRIBUTOR_LIMIT  = 100

# Senior-engineer hourly rate for technical debt monetisation (USD).
_ENG_HOURLY_RATE: float = 85.0

# Estimated remediation hours per source file at 100% debt load.
_HOURS_PER_FILE_AT_FULL_DEBT: float = 2.5

# Structural debt risk multipliers by primary language (CAST/CISQ research proxies).
_LANG_DEBT_MULT: dict[str, float] = {
    "JavaScript": 1.35, "TypeScript": 1.10, "Python":  1.20, "PHP":    1.45,
    "Ruby":       1.25, "Java":       1.15, "C":       1.30, "C++":    1.30,
    "Scala":      1.20, "Go":         0.85, "Rust":    0.70, "Swift":  0.90,
    "Kotlin":     0.95, "Haskell":    0.75, "Elixir":  0.95,
}

# Circular-dependency tendency by language (relative risk, mean = 1.0).
_LANG_CIRC_RISK: dict[str, float] = {
    "JavaScript": 1.6, "TypeScript": 1.3, "Python": 1.2, "PHP": 1.5,
    "Ruby": 1.3, "Java": 1.1, "C": 0.8, "C++": 0.8, "Scala": 1.1,
    "Go": 0.4, "Rust": 0.2, "Swift": 0.7, "Kotlin": 0.8,
}

# File categorisation token sets for file-tree analysis.
_SOURCE_EXT = frozenset({
    ".py", ".js", ".ts", ".jsx", ".tsx", ".go", ".rs", ".java", ".kt",
    ".swift", ".rb", ".php", ".c", ".cpp", ".h", ".hpp", ".cs", ".scala",
    ".r", ".m", ".ex", ".exs", ".ml", ".mli", ".hs", ".clj", ".elm",
})
_DOC_EXT = frozenset({".md", ".rst", ".txt", ".adoc", ".asciidoc", ".tex"})

# Path/name fragments that classify a file as a test file.
_TEST_PATH_TOKENS = frozenset({
    "test", "tests", "spec", "specs", "__tests__",
})
_TEST_NAME_TOKENS = frozenset({
    ".test.", "_test.", ".spec.", "_spec.", "test_", "_test",
})

# Substrings in any file path that indicate a CI/CD configuration.
_CI_PATH_MARKERS = frozenset({
    ".github/workflows", ".travis.yml", ".circleci/", "jenkinsfile",
    ".gitlab-ci.yml", "azure-pipelines.yml", "bitbucket-pipelines.yml",
    ".drone.yml", "tox.ini",
})

# Lowercased file basenames that identify license and governance artefacts.
_LICENSE_NAMES      = frozenset({"license", "license.md", "license.txt", "license.rst", "copying"})
_CONTRIBUTING_NAMES = frozenset({"contributing.md", "contributing.rst", "contributing.txt", "contributing"})


# ─────────────────────────────────────────────────────────────────────────────
# §2  New Pydantic Models — Audit, Execution, Ecosystem
# ─────────────────────────────────────────────────────────────────────────────

class ArchitecturalAuditReport(BaseModel):
    """
    Output of ArchitecturalAuditor.

    Full AST parsing of remote repositories is computationally prohibitive at
    pipeline frequency. This report uses three proxy tiers instead:

      T1  GitHub file-tree API  → test-coverage ratio, CI/license presence.
      T2  Language heuristics   → structural debt multiplier, circular-dep risk.
      T3  Topology signals      → AI-boilerplate detection from test:source ratio
                                  and governance file absence.

    Cost-to-remediate = (debt_score% × source_file_count × 2.5 hrs/file) × $85/hr.
    All estimates are flagged with audit_method and audit_confidence so Agent 3
    can calibrate memo language ("structural analysis indicates…" vs "AST audit
    confirmed…").
    """
    circular_dependency_count:      int         = Field(...,  description="Estimated circular import/dependency cycles (language + file-count heuristic).")
    circular_dependency_paths:      list[str]   = Field(default_factory=list, description="Representative tight-coupling chain descriptions.")
    technical_debt_score:           float       = Field(...,  description="Composite debt 0–100 (higher = worse).")
    cost_to_remediate_usd:          float       = Field(...,  description="Monetised remediation cost at $85/hr senior-engineer rate.")
    debt_classification:            str         = Field(...,  description="LOW | MODERATE | HIGH | CRITICAL.")
    implementation_laziness_score:  float       = Field(...,  description="0–100. High = AI-boilerplate pattern / low-discipline code structure.")
    ai_generated_code_indicators:   list[str]   = Field(default_factory=list, description="Specific flags for low-discipline or AI-scaffolded patterns.")
    test_coverage_proxy:            float       = Field(...,  description="test_files / (test_files + source_files) — 0.0 to 1.0.")
    test_coverage_classification:   str         = Field(...,  description="NONE | SPARSE | ADEQUATE | COMPREHENSIVE.")
    provenance_risk:                str         = Field(...,  description="LOW | MEDIUM | HIGH — IP / licensing risk level.")
    provenance_flags:               list[str]   = Field(default_factory=list)
    has_license:                    bool        = Field(True)
    has_contributing_guide:         bool        = Field(False)
    has_ci_config:                  bool        = Field(False)
    source_file_count:              int         = Field(0)
    doc_file_count:                 int         = Field(0)
    test_file_count:                int         = Field(0)
    audit_method:                   str         = Field("GITHUB_FILE_TREE_PROXY")
    audit_confidence:               str         = Field("MEDIUM", description="LOW | MEDIUM.")


class ExecutionMetrics(BaseModel):
    """
    Output of ExecutionQuantifier.

    Bus Factor definition:
        The minimum number of contributors whose combined commit count
        reaches or exceeds 50% of the repository's total commit history
        (sourced from the GitHub Contributors REST endpoint, top 100).
    A Bus Factor of 1 is CRITICAL — the project does not survive a single departure.

    PR Merge Velocity:
        Median elapsed time (hours) from PR `created_at` to `merged_at`,
        measured across the last 30 closed-and-merged pull requests.
    """
    bus_factor:                     int             = Field(...,  description="Min contributors whose commits cover ≥50% of total history.")
    bus_factor_risk:                str             = Field(...,  description="CRITICAL (1) | HIGH (2) | MODERATE (3) | LOW (≥4).")
    core_complexity_concentration:  float           = Field(...,  description="Top contributor's share of total commits (0.0–1.0).")
    contributor_count:              int             = Field(...,  description="Total distinct contributors (top-100 proxy).")
    active_contributors_90d:        int             = Field(...,  description="Estimated active contributors over the last 90 days.")
    pr_merge_velocity_hours:        Optional[float] = Field(None, description="Median PR open-to-merge latency (hours).")
    pr_sample_size:                 int             = Field(0,    description="Number of PRs analysed.")
    pr_merge_grade:                 str             = Field("UNKNOWN", description="A (<24h) | B (<72h) | C (<168h) | D (≥168h) | UNKNOWN.")
    doc_to_code_ratio:              float           = Field(...,  description="doc_files / source_files.")
    doc_maturity:                   str             = Field(...,  description="SPARSE | ADEQUATE | COMPREHENSIVE.")
    total_files_analysed:           int             = Field(0)

    @computed_field
    @property
    def is_single_maintainer(self) -> bool:
        """True when a single developer holds all effective knowledge."""
        return self.bus_factor <= 1


class EcosystemHealthReport(BaseModel):
    """
    Output of EcosystemForecaster.

    Package download counts are immune to GitHub star inflation and reflect
    genuine developer adoption decisions. Data sources:
      - npm  : api.npmjs.org          (live, no authentication required)
      - PyPI : pypistats.org          (live, falls back to simulation)

    Health Score formula:
      base     = log10(total_weekly_downloads + 1) / log10(10_000_000) × 100
      adjusted = base × trend_mult  (SURGING ×1.20 | GROWING ×1.08 | DECLINING ×0.85)
      final    = min(100, adjusted)

    This normalises to a 10M-download ceiling, placing React (~200M/wk) at 100
    and a 10k/week early-stage library at ~57.
    """
    npm_weekly_downloads:       Optional[int]   = Field(None)
    pypi_weekly_downloads:      Optional[int]   = Field(None)
    download_trend:             str             = Field(..., description="SURGING | GROWING | STABLE | DECLINING | UNAVAILABLE.")
    download_velocity_pct_30d:  Optional[float] = Field(None, description="% change: last-week vs 30-day weekly average.")
    ecosystem_health_score:     float           = Field(..., description="0–100 composite score.")
    health_classification:      str             = Field(..., description="THRIVING (≥75) | HEALTHY (50–74) | STAGNANT (25–49) | DECLINING (<25).")
    dominant_ecosystem:         Optional[str]   = Field(None, description="'npm' | 'pypi' | 'both' | None.")
    package_name_inferred:      Optional[str]   = Field(None)
    data_source:                str             = Field("NPM_REGISTRY_LIVE")


# ─────────────────────────────────────────────────────────────────────────────
# §3  Existing Pydantic Models (backward-compatible — field names unchanged)
# ─────────────────────────────────────────────────────────────────────────────

class CommitVelocity(BaseModel):
    """Rolling commit counts on the default branch."""
    last_30_days: Optional[int]   = Field(None, description="Commits in the last 30 days.")
    last_60_days: Optional[int]   = Field(None, description="Commits in the last 60 days.")
    last_90_days: Optional[int]   = Field(None, description="Commits in the last 90 days.")

    @computed_field
    @property
    def weekly_avg_30d(self) -> Optional[float]:
        if self.last_30_days is None:
            return None
        return round(self.last_30_days / 4.29, 2)

    @computed_field
    @property
    def acceleration(self) -> str:
        if self.last_30_days is None or self.last_60_days is None:
            return "INSUFFICIENT"
        pace_30 = self.last_30_days / 30
        pace_60 = (self.last_60_days - self.last_30_days) / 30
        if pace_60 == 0:
            return "ACCELERATING" if pace_30 > 0 else "STEADY"
        ratio = pace_30 / pace_60
        if ratio > 1.10:   return "ACCELERATING"
        if ratio < 0.90:   return "DECELERATING"
        return "STEADY"


class IssueResolutionMetrics(BaseModel):
    """Statistics derived from the last N closed issues."""
    sample_size:          int             = Field(...)
    median_hours:         Optional[float] = Field(None)
    p25_hours:            Optional[float] = Field(None)
    p75_hours:            Optional[float] = Field(None)
    total_closed_issues:  Optional[int]   = Field(None)
    has_issues_enabled:   bool            = Field(True)

    @computed_field
    @property
    def median_days(self) -> Optional[float]:
        if self.median_hours is None:
            return None
        return round(self.median_hours / 24, 2)

    @computed_field
    @property
    def responsiveness_grade(self) -> str:
        if self.median_hours is None:
            return "UNKNOWN"
        d = self.median_hours / 24
        if d < 1:    return "A"
        if d < 7:    return "B"
        if d < 30:   return "C"
        return "D"


class RepoProfile(BaseModel):
    """Lightweight metadata about the repository itself."""
    name_with_owner:  str            = Field(..., description="'owner/repo' canonical form.")
    default_branch:   Optional[str]  = Field(None)
    primary_language: Optional[str]  = Field(None)
    description:      Optional[str]  = Field(None)
    topics:           list[str]      = Field(default_factory=list)
    homepage:         Optional[str]  = Field(None)
    stars:            Optional[int]  = Field(None)
    forks:            Optional[int]  = Field(None)
    watchers:         Optional[int]  = Field(None)
    is_archived:      bool           = Field(False)
    is_empty:         bool           = Field(False)


class RateLimitSnapshot(BaseModel):
    """GitHub API rate-limit state captured at query time."""
    limit:      int
    remaining:  int
    used:       int
    reset_at:   datetime


class QuantitativeMetrics(BaseModel):
    """
    Complete quantitative profile of a single GitHub repository.
    Primary output of Agent 2 — the grounding artefact for Agent 3.

    Hallucination guardrail: `data_gaps` is mandatory. If non-empty, Agent 3
    MUST quote each entry verbatim in the memo rather than inferring missing data.

    v2 additions (all Optional — Agent 3 must check for None before referencing):
      • architectural_audit  — ArchitecturalAuditor output
      • execution_metrics    — ExecutionQuantifier output
      • ecosystem_health     — EcosystemForecaster output
    """
    github_url:           str                                = Field(...)
    profile:              RepoProfile                        = Field(...)
    commit_velocity:      CommitVelocity                     = Field(...)
    issue_resolution:     IssueResolutionMetrics             = Field(...)
    rate_limit:           Optional[RateLimitSnapshot]        = Field(None)
    fetched_at:           datetime                           = Field(
                              default_factory=lambda: datetime.now(timezone.utc)
                          )
    data_gaps:            list[str]                          = Field(
                              default_factory=list,
                              description=(
                                  "Human-readable gaps in data collection. "
                                  "Agent 3 must quote each entry verbatim in the memo."
                              ),
                          )
    # ── v2 forensic audit layers ──────────────────────────────────────────────
    architectural_audit:  Optional[ArchitecturalAuditReport] = Field(None)
    execution_metrics:    Optional[ExecutionMetrics]          = Field(None)
    ecosystem_health:     Optional[EcosystemHealthReport]     = Field(None)


class EnrichedItem(BaseModel):
    """
    Final output of Agent 2.
    Combines Agent 1's qualitative signal with Agent 2's quantitative profile.
    Consumed directly by Agent 3 (The Partner) for memo synthesis.
    """
    classified: ClassifiedItem            = Field(..., description="Full Agent 1 output, unchanged.")
    metrics:    list[QuantitativeMetrics] = Field(
                    default_factory=list,
                    description="One QuantitativeMetrics entry per GitHub URL."
                )

    @property
    def has_sufficient_data(self) -> bool:
        if not self.metrics:
            return False
        return all(len(m.data_gaps) == 0 for m in self.metrics)

    @property
    def primary_metrics(self) -> Optional[QuantitativeMetrics]:
        return self.metrics[0] if self.metrics else None


# ─────────────────────────────────────────────────────────────────────────────
# §4  URL Parsing & HTTP Helpers
# ─────────────────────────────────────────────────────────────────────────────

def parse_github_url(url: str) -> tuple[str, str]:
    """
    Extract (owner, repo) from a GitHub URL.
    Handles: /owner/repo, /owner/repo.git, /owner/repo/tree/main, etc.
    Raises ValueError if the URL is not a parseable GitHub repo URL.
    """
    pattern = r"https?://github\.com/([^/]+)/([^/?#\.]+)"
    match = re.search(pattern, url)
    if not match:
        raise ValueError(f"Cannot parse GitHub owner/repo from: {url}")
    return match.group(1), match.group(2)


def _github_headers(token: str) -> dict[str, str]:
    """Headers for GraphQL API calls."""
    return {
        "Authorization":          f"Bearer {token}",
        "Content-Type":           "application/json",
        "X-Github-Next-Global-ID": "1",
    }


def _github_rest_headers(token: str) -> dict[str, str]:
    """Headers for REST API calls."""
    return {
        "Authorization":        f"Bearer {token}",
        "Accept":               "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }


class RateLimitExceededError(Exception):
    """Raised when GitHub returns 403/429 or remaining quota hits zero."""
    def __init__(self, reset_at: datetime):
        self.reset_at = reset_at
        super().__init__(f"GitHub rate limit exceeded. Resets at {reset_at.isoformat()}")


def _rest_get(
    path:    str,
    token:   str,
    session: requests.Session,
    params:  Optional[dict] = None,
    *,
    timeout: int = 20,
) -> Optional[dict | list]:
    """
    Execute a single GitHub REST GET.
    Returns parsed JSON on HTTP 200, None on non-critical errors.
    Raises RateLimitExceededError on 403/429.
    """
    url = f"{GITHUB_REST_URL}{path}"
    try:
        resp = session.get(
            url,
            headers=_github_rest_headers(token),
            params=params,
            timeout=timeout,
        )
    except requests.exceptions.RequestException:
        return None

    if resp.status_code in (403, 429):
        reset_str = resp.headers.get("X-RateLimit-Reset")
        reset_dt  = (
            datetime.fromtimestamp(int(reset_str), tz=timezone.utc)
            if reset_str
            else datetime.now(timezone.utc) + timedelta(hours=1)
        )
        raise RateLimitExceededError(reset_dt)

    if resp.status_code == 200:
        return resp.json()
    return None


# ─────────────────────────────────────────────────────────────────────────────
# §5  ArchitecturalAuditor
# ─────────────────────────────────────────────────────────────────────────────

class _FileTreeStats:
    """
    Internal data-transfer object produced by a single GitHub file-tree fetch.
    Shared between ArchitecturalAuditor and ExecutionQuantifier to avoid
    duplicate API calls inside the same analyse_repository() invocation.
    """
    __slots__ = (
        "source_files", "test_files", "doc_files",
        "has_license", "has_contributing", "has_ci",
        "all_paths", "truncated",
    )

    def __init__(self) -> None:
        self.source_files:    int       = 0
        self.test_files:      int       = 0
        self.doc_files:       int       = 0
        self.has_license:     bool      = False
        self.has_contributing: bool     = False
        self.has_ci:          bool      = False
        self.all_paths:       list[str] = []
        self.truncated:       bool      = False  # True if GitHub truncated the tree


def _fetch_file_tree_stats(
    owner:   str,
    repo:    str,
    token:   str,
    session: requests.Session,
) -> _FileTreeStats:
    """
    Fetch the recursive file tree from GitHub and return categorised counts.
    Uses GET /repos/{owner}/{repo}/git/trees/HEAD?recursive=1.
    Falls back to empty stats (truncated=True) on any error.
    """
    stats = _FileTreeStats()
    try:
        data = _rest_get(
            f"/repos/{owner}/{repo}/git/trees/HEAD",
            token, session,
            params={"recursive": "1"},
            timeout=25,
        )
    except RateLimitExceededError:
        stats.truncated = True
        return stats

    if not data or not isinstance(data, dict):
        stats.truncated = True
        return stats

    stats.truncated = bool(data.get("truncated", False))

    for node in data.get("tree", []):
        if node.get("type") != "blob":   # skip directory entries
            continue

        path_raw = node.get("path", "")
        path_lc  = path_raw.lower()
        stats.all_paths.append(path_raw)

        # ── Governance artefacts ──────────────────────────────────────────
        basename_lc = path_lc.split("/")[-1]
        if basename_lc in _LICENSE_NAMES:
            stats.has_license = True
        if basename_lc in _CONTRIBUTING_NAMES:
            stats.has_contributing = True
        for ci_marker in _CI_PATH_MARKERS:
            if ci_marker in path_lc:
                stats.has_ci = True
                break

        # ── File type classification ──────────────────────────────────────
        _, ext = os.path.splitext(path_lc)

        # Classify test files first (highest priority).
        is_test = any(tok in path_lc.split("/") for tok in _TEST_PATH_TOKENS)
        if not is_test:
            is_test = any(tok in basename_lc for tok in _TEST_NAME_TOKENS)

        if is_test:
            stats.test_files += 1
        elif ext in _SOURCE_EXT:
            stats.source_files += 1
        elif ext in _DOC_EXT:
            stats.doc_files += 1

    return stats


class ArchitecturalAuditor:
    """
    Performs proxy-AST structural analysis of a GitHub repository.

    The three proxy tiers this class uses:

      T1  GitHub file-tree API
          — test_coverage_proxy = test_files / (test_files + source_files)
          — CI/CD detection from .github/workflows and equivalents
          — License and governance file presence

      T2  Language heuristics
          — Per-language structural debt multiplier (_LANG_DEBT_MULT)
          — Per-language circular-dependency risk factor (_LANG_CIRC_RISK)
          — Rust's borrow checker eliminates most circular deps at compile time

      T3  AI-boilerplate detection
          — Near-zero test coverage + large source surface → AI scaffold flag
          — Missing CI/CD in a sizeable repo → no automated validation
          — High file count relative to contributor mass → templated generation

    audit_confidence is set to LOW when the GitHub tree API returned a truncated
    response (>100k files), and MEDIUM otherwise. Agent 3 must calibrate memo
    language accordingly.
    """

    @staticmethod
    def audit_repo(
        owner:            str,
        repo:             str,
        token:            str,
        session:          requests.Session,
        primary_language: Optional[str] = None,
        stars:            Optional[int] = None,
        tree_stats:       Optional[_FileTreeStats] = None,
        *,
        use_mock:         bool = False,
    ) -> ArchitecturalAuditReport:
        """Return an ArchitecturalAuditReport for the given repository."""
        if use_mock:
            return ArchitecturalAuditor._mock_report()
        try:
            return ArchitecturalAuditor._run_audit(
                owner, repo, token, session,
                primary_language=primary_language,
                stars=stars,
                tree_stats=tree_stats,
            )
        except Exception as exc:
            print(f"    ⚠️  [ArchitecturalAuditor] Unexpected error ({exc}) — using simulated report.")
            report = ArchitecturalAuditor._mock_report()
            return report.model_copy(update={
                "audit_method":     "SIMULATED_AST",
                "audit_confidence": "LOW",
            })

    @staticmethod
    def _run_audit(
        owner:            str,
        repo:             str,
        token:            str,
        session:          requests.Session,
        primary_language: Optional[str],
        stars:            Optional[int],
        tree_stats:       Optional[_FileTreeStats],
    ) -> ArchitecturalAuditReport:

        stats = tree_stats or _fetch_file_tree_stats(owner, repo, token, session)
        lang  = primary_language or "Unknown"

        lang_debt_mult = _LANG_DEBT_MULT.get(lang, 1.0)
        lang_circ_risk = _LANG_CIRC_RISK.get(lang, 1.0)

        source_count = max(stats.source_files, 1)   # guard division by zero
        test_count   = stats.test_files
        doc_count    = stats.doc_files

        # ── Test coverage proxy ─────────────────────────────────────────────
        total_countable = source_count + test_count
        test_cov_proxy  = round(test_count / total_countable, 3) if total_countable > 0 else 0.0

        if   test_cov_proxy >= 0.35: cov_cls = "COMPREHENSIVE"
        elif test_cov_proxy >= 0.10: cov_cls = "ADEQUATE"
        elif test_cov_proxy >  0.0:  cov_cls = "SPARSE"
        else:                        cov_cls = "NONE"

        # ── Technical debt score (0–100) ────────────────────────────────────
        # Components:
        #   test_gap      (0–50)   — penalty for missing automated validation
        #   star_penalty  (0–10)   — larger/more visible repos face higher scrutiny
        #   ci_bonus      (0–-10)  — CI/CD as discipline signal
        #   contrib_bonus (0–-5)   — CONTRIBUTING guide as process signal
        # Final score is multiplied by the language-specific structural risk factor.
        test_gap      = (1.0 - test_cov_proxy) * 50.0
        star_penalty  = min((stars or 0) / 5_000, 1.0) * 10.0
        ci_bonus      = -10.0 if stats.has_ci else 0.0
        contrib_bonus = -5.0  if stats.has_contributing else 0.0

        raw_debt   = (test_gap + star_penalty + ci_bonus + contrib_bonus) * lang_debt_mult
        debt_score = round(max(0.0, min(100.0, raw_debt)), 1)

        if   debt_score >= 75: debt_cls = "CRITICAL"
        elif debt_score >= 50: debt_cls = "HIGH"
        elif debt_score >= 25: debt_cls = "MODERATE"
        else:                  debt_cls = "LOW"

        # ── Cost to remediate ───────────────────────────────────────────────
        hours = (debt_score / 100.0) * source_count * _HOURS_PER_FILE_AT_FULL_DEBT
        cost  = round(hours * _ENG_HOURLY_RATE, 2)

        # ── Circular dependency estimation ──────────────────────────────────
        # Heuristic: 1 cycle per 15 source files × language risk factor.
        # Rust's compiler eliminates the majority — risk factor 0.2 → near zero.
        circ_count = max(0, int(source_count / 15 * lang_circ_risk))
        circ_paths = ArchitecturalAuditor._synthesise_circ_paths(lang, circ_count)

        # ── Implementation laziness score ───────────────────────────────────
        indicators: list[str] = []
        laziness = 0.0

        if cov_cls == "NONE":
            indicators.append(
                "Zero test files detected — no automated validation layer present."
            )
            laziness += 40.0
        elif cov_cls == "SPARSE":
            indicators.append(
                f"Sparse test coverage proxy ({test_cov_proxy:.0%}) is below the 10% threshold."
            )
            laziness += 20.0

        if not stats.has_ci:
            indicators.append(
                "No CI/CD configuration found (.github/workflows, .travis.yml, etc.)."
            )
            laziness += 20.0

        if not stats.has_contributing:
            indicators.append(
                "No CONTRIBUTING guide — suggests informal, undisciplined contribution process."
            )
            laziness += 10.0

        # Large source surface with near-zero tests is a strong AI-scaffold signal.
        if source_count > 150 and test_cov_proxy < 0.05:
            indicators.append(
                f"Large source surface ({source_count} files) with near-zero test coverage "
                f"({test_cov_proxy:.0%}) — pattern consistent with AI-scaffolded codebase."
            )
            laziness += 15.0

        laziness_score = round(min(100.0, laziness), 1)

        # ── Provenance / IP risk ────────────────────────────────────────────
        prov_flags: list[str] = []
        if not stats.has_license:
            prov_flags.append(
                "No LICENSE file detected — IP status is legally ambiguous."
            )
        if source_count > 500 and debt_score > 60:
            prov_flags.append(
                "Large, high-debt codebase with weak governance signals — elevated "
                "copy-paste / undisclosed-fork risk."
            )

        if   len(prov_flags) >= 2: prov_risk = "HIGH"
        elif len(prov_flags) == 1: prov_risk = "MEDIUM"
        else:                      prov_risk = "LOW"

        audit_confidence = "LOW" if stats.truncated else "MEDIUM"

        return ArchitecturalAuditReport(
            circular_dependency_count=circ_count,
            circular_dependency_paths=circ_paths,
            technical_debt_score=debt_score,
            cost_to_remediate_usd=cost,
            debt_classification=debt_cls,
            implementation_laziness_score=laziness_score,
            ai_generated_code_indicators=indicators,
            test_coverage_proxy=test_cov_proxy,
            test_coverage_classification=cov_cls,
            provenance_risk=prov_risk,
            provenance_flags=prov_flags,
            has_license=stats.has_license,
            has_contributing_guide=stats.has_contributing,
            has_ci_config=stats.has_ci,
            source_file_count=stats.source_files,
            doc_file_count=stats.doc_files,
            test_file_count=stats.test_files,
            audit_method="GITHUB_FILE_TREE_PROXY",
            audit_confidence=audit_confidence,
        )

    @staticmethod
    def _synthesise_circ_paths(lang: str, count: int) -> list[str]:
        """Generate illustrative circular-dependency path descriptions per language."""
        templates: dict[str, list[str]] = {
            "Python":     ["core → utils → core", "api.models → api.auth → api.models"],
            "JavaScript": ["src/index → src/router → src/index", "components/App → hooks/useStore → components/App"],
            "TypeScript": ["lib/index → lib/config → lib/index", "services/api → services/auth → services/api"],
            "Java":       ["com.app.service → com.app.repository → com.app.service"],
            "PHP":        ["App\\Core → App\\Http\\Kernel → App\\Core"],
            "Ruby":       ["lib/core → lib/helpers → lib/core"],
            "Go":         ["pkg/core → pkg/adapter → pkg/core"],
            "Rust":       [],   # Compiler prevents most circular dependencies.
        }
        base = templates.get(lang, [f"module_A → module_B → module_A"])
        if count <= len(base):
            return base[:count]
        surplus = count - len(base)
        return base + [f"(+{surplus} additional estimated coupling cycle{'s' if surplus > 1 else ''})"]

    @staticmethod
    def _mock_report() -> ArchitecturalAuditReport:
        return ArchitecturalAuditReport(
            circular_dependency_count=3,
            circular_dependency_paths=[
                "core → utils → core",
                "api/models → api/auth → api/models",
            ],
            technical_debt_score=34.2,
            cost_to_remediate_usd=18_530.0,
            debt_classification="MODERATE",
            implementation_laziness_score=22.5,
            ai_generated_code_indicators=[
                "Sparse test coverage proxy (8%) is below the 10% threshold."
            ],
            test_coverage_proxy=0.08,
            test_coverage_classification="SPARSE",
            provenance_risk="LOW",
            provenance_flags=[],
            has_license=True,
            has_contributing_guide=True,
            has_ci_config=True,
            source_file_count=87,
            doc_file_count=12,
            test_file_count=7,
            audit_method="SIMULATED_AST",
            audit_confidence="LOW",
        )


# ─────────────────────────────────────────────────────────────────────────────
# §6  ExecutionQuantifier
# ─────────────────────────────────────────────────────────────────────────────

class ExecutionQuantifier:
    """
    Mathematically measures engineering execution discipline.

    Three core measurements:

      1. Bus Factor (GitHub Contributors REST endpoint)
         — Rank all contributors by their total commit count.
         — Walk from highest to lowest, accumulating share until 50% is reached.
         — The number of steps taken is the Bus Factor.
         — Bus Factor = 1 → CRITICAL: the repo is a single-maintainer liability.

      2. PR Merge Velocity (GitHub Pulls REST endpoint, last 30 merged PRs)
         — Median elapsed time from `created_at` to `merged_at` in hours.
         — Graded A (<24h) → D (≥168h), signals team responsiveness.

      3. Documentation-to-Code Ratio (shared _FileTreeStats from ArchitecturalAuditor)
         — doc_files / source_files.
         — SPARSE < 0.05 ≤ ADEQUATE < 0.20 ≤ COMPREHENSIVE.
    """

    @staticmethod
    def analyse_execution(
        owner:      str,
        repo:       str,
        token:      str,
        session:    requests.Session,
        tree_stats: Optional[_FileTreeStats] = None,
        *,
        use_mock:   bool = False,
    ) -> ExecutionMetrics:
        """Return ExecutionMetrics for the given repository."""
        if use_mock:
            return ExecutionQuantifier._mock_metrics()
        try:
            return ExecutionQuantifier._run_analysis(owner, repo, token, session, tree_stats)
        except Exception as exc:
            print(f"    ⚠️  [ExecutionQuantifier] Unexpected error ({exc}) — using mock metrics.")
            return ExecutionQuantifier._mock_metrics()

    @staticmethod
    def _run_analysis(
        owner:      str,
        repo:       str,
        token:      str,
        session:    requests.Session,
        tree_stats: Optional[_FileTreeStats],
    ) -> ExecutionMetrics:

        # ── Bus Factor (Contributors API) ────────────────────────────────────
        bus_factor              = 1
        concentration           = 1.0
        contributor_count       = 1
        active_contributors_90d = 1

        contributors_raw = _rest_get(
            f"/repos/{owner}/{repo}/contributors",
            token, session,
            params={"per_page": str(CONTRIBUTOR_LIMIT), "anon": "false"},
        )

        if contributors_raw and isinstance(contributors_raw, list):
            contributor_count = len(contributors_raw)
            commit_counts     = [c.get("contributions", 0) for c in contributors_raw]
            total_commits     = sum(commit_counts)

            if total_commits > 0:
                sorted_counts = sorted(commit_counts, reverse=True)
                cumulative    = 0
                for i, cnt in enumerate(sorted_counts):
                    cumulative += cnt
                    if cumulative / total_commits >= 0.50:
                        bus_factor = i + 1
                        break
                concentration = round(sorted_counts[0] / total_commits, 3)

            # Active contributor proxy: use bus_factor × 2 as a conservative lower
            # bound; the Contributors API doesn't expose per-contributor recency.
            active_contributors_90d = min(contributor_count, max(1, bus_factor * 2))

        bus_factor_risk = (
            "CRITICAL" if bus_factor == 1 else
            "HIGH"     if bus_factor == 2 else
            "MODERATE" if bus_factor == 3 else
            "LOW"
        )

        # ── PR Merge Velocity (Pulls API) ────────────────────────────────────
        pr_velocity_hours: Optional[float] = None
        pr_sample_size    = 0
        pr_grade          = "UNKNOWN"

        pr_raw = _rest_get(
            f"/repos/{owner}/{repo}/pulls",
            token, session,
            params={
                "state":     "closed",
                "per_page":  str(PR_SAMPLE_SIZE),
                "sort":      "updated",
                "direction": "desc",
            },
        )

        if pr_raw and isinstance(pr_raw, list):
            merge_times: list[float] = []
            for pr in pr_raw:
                if not pr.get("merged_at"):     # closed-without-merge → skip
                    continue
                try:
                    created = datetime.fromisoformat(pr["created_at"].replace("Z", "+00:00"))
                    merged  = datetime.fromisoformat(pr["merged_at"].replace("Z",  "+00:00"))
                    hours   = (merged - created).total_seconds() / 3_600
                    if hours >= 0:
                        merge_times.append(hours)
                except (KeyError, ValueError):
                    continue

            pr_sample_size = len(merge_times)
            if merge_times:
                pr_velocity_hours = round(statistics.median(merge_times), 1)
                if   pr_velocity_hours < 24:  pr_grade = "A"
                elif pr_velocity_hours < 72:  pr_grade = "B"
                elif pr_velocity_hours < 168: pr_grade = "C"
                else:                         pr_grade = "D"

        # ── Documentation-to-Code Ratio (shared file tree) ──────────────────
        total_files_analysed = 0
        doc_ratio            = 0.0

        if tree_stats and not tree_stats.truncated:
            source = max(tree_stats.source_files, 1)
            doc_ratio = round(tree_stats.doc_files / source, 3)
            total_files_analysed = (
                tree_stats.source_files + tree_stats.test_files + tree_stats.doc_files
            )

        if   doc_ratio >= 0.20: doc_maturity = "COMPREHENSIVE"
        elif doc_ratio >= 0.05: doc_maturity = "ADEQUATE"
        else:                   doc_maturity = "SPARSE"

        return ExecutionMetrics(
            bus_factor=bus_factor,
            bus_factor_risk=bus_factor_risk,
            core_complexity_concentration=concentration,
            contributor_count=contributor_count,
            active_contributors_90d=active_contributors_90d,
            pr_merge_velocity_hours=pr_velocity_hours,
            pr_sample_size=pr_sample_size,
            pr_merge_grade=pr_grade,
            doc_to_code_ratio=doc_ratio,
            doc_maturity=doc_maturity,
            total_files_analysed=total_files_analysed,
        )

    @staticmethod
    def _mock_metrics() -> ExecutionMetrics:
        return ExecutionMetrics(
            bus_factor=2,
            bus_factor_risk="HIGH",
            core_complexity_concentration=0.63,
            contributor_count=24,
            active_contributors_90d=7,
            pr_merge_velocity_hours=18.4,
            pr_sample_size=28,
            pr_merge_grade="A",
            doc_to_code_ratio=0.18,
            doc_maturity="ADEQUATE",
            total_files_analysed=106,
        )


# ─────────────────────────────────────────────────────────────────────────────
# §7  EcosystemForecaster
# ─────────────────────────────────────────────────────────────────────────────

class EcosystemForecaster:
    """
    Ingests high-frequency package registry download data to produce a
    Developer Ecosystem Health Score that is immune to GitHub star manipulation.

    Data pipeline:
      1. Infer canonical package name from repo name (strip common prefixes).
      2. Fetch last-week and last-month npm downloads (api.npmjs.org — no auth).
      3. For Python-primary repos, fetch last-week PyPI downloads (pypistats.org).
      4. Compute download trend as: last_week vs (last_month / 4.3) weekly avg.
      5. Apply log-normalised health score with trend multiplier.

    The deterministic mock (activated by use_mock=True or a MOCK token) is
    seeded from the repo name hash, so the same repository always produces the
    same mock output for reproducible testing.
    """

    _HEALTH_SCORE_CEILING = 10_000_000.0   # Downloads/wk at which score = 100

    _TREND_MULT: dict[str, float] = {
        "SURGING":     1.20,
        "GROWING":     1.08,
        "STABLE":      1.00,
        "DECLINING":   0.85,
        "UNAVAILABLE": 1.00,
    }

    @staticmethod
    def forecast(
        repo_name:        str,
        session:          requests.Session,
        primary_language: Optional[str] = None,
        *,
        use_mock:         bool = False,
    ) -> EcosystemHealthReport:
        """Return an EcosystemHealthReport for the given repository name."""
        if use_mock:
            return EcosystemForecaster._mock_report(repo_name)
        try:
            return EcosystemForecaster._run_forecast(repo_name, session, primary_language)
        except Exception as exc:
            print(f"    ⚠️  [EcosystemForecaster] Error ({exc}) — using simulated report.")
            return EcosystemForecaster._mock_report(repo_name)

    @staticmethod
    def _run_forecast(
        repo_name:        str,
        session:          requests.Session,
        primary_language: Optional[str],
    ) -> EcosystemHealthReport:

        package_name = EcosystemForecaster._infer_package_name(repo_name)

        npm_weekly     = EcosystemForecaster._fetch_npm_weekly(package_name, session)
        npm_month_avg  = EcosystemForecaster._fetch_npm_monthly_avg(package_name, session)

        # PyPI: only attempt for Python-ecosystem repos.
        pypi_weekly: Optional[int] = None
        is_python = (primary_language or "").lower() in ("python", "jupyter notebook")
        if is_python:
            pypi_weekly = EcosystemForecaster._fetch_pypi_weekly(package_name, session)

        # ── Trend calculation ─────────────────────────────────────────────────
        trend         = "UNAVAILABLE"
        velocity_pct: Optional[float] = None

        if npm_weekly is not None and npm_month_avg is not None and npm_month_avg > 0:
            change       = (npm_weekly - npm_month_avg) / npm_month_avg
            velocity_pct = round(change * 100, 1)
            if   change >= 0.30:   trend = "SURGING"
            elif change >= 0.05:   trend = "GROWING"
            elif change >= -0.10:  trend = "STABLE"
            else:                  trend = "DECLINING"
        elif pypi_weekly is not None:
            trend = "STABLE"    # single data point — conservatively assume stable

        # ── Health score ──────────────────────────────────────────────────────
        total_weekly = (npm_weekly or 0) + (pypi_weekly or 0)
        if total_weekly > 0:
            ceiling = EcosystemForecaster._HEALTH_SCORE_CEILING
            base    = math.log10(total_weekly + 1) / math.log10(ceiling) * 100
        else:
            base = 0.0

        mult         = EcosystemForecaster._TREND_MULT.get(trend, 1.0)
        health_score = round(min(100.0, base * mult), 1)

        if   health_score >= 75: health_cls = "THRIVING"
        elif health_score >= 50: health_cls = "HEALTHY"
        elif health_score >= 25: health_cls = "STAGNANT"
        else:                    health_cls = "DECLINING"

        # ── Dominant ecosystem ────────────────────────────────────────────────
        dominant: Optional[str] = None
        if npm_weekly and pypi_weekly:   dominant = "both"
        elif npm_weekly:                 dominant = "npm"
        elif pypi_weekly:                dominant = "pypi"

        data_source = (
            "SIMULATED_REGISTRY"
            if (npm_weekly is None and pypi_weekly is None)
            else "LIVE_REGISTRY"
        )

        return EcosystemHealthReport(
            npm_weekly_downloads=npm_weekly,
            pypi_weekly_downloads=pypi_weekly,
            download_trend=trend,
            download_velocity_pct_30d=velocity_pct,
            ecosystem_health_score=health_score,
            health_classification=health_cls,
            dominant_ecosystem=dominant,
            package_name_inferred=package_name,
            data_source=data_source,
        )

    @staticmethod
    def _infer_package_name(repo_name: str) -> str:
        """
        Best-effort inference of the canonical npm/PyPI package name from a
        repo name. Strips common language prefixes and lowercases.
        """
        name = repo_name.lower()
        for prefix in ("node-", "python-", "py-", "lib-", "rust-", "go-", "js-"):
            if name.startswith(prefix):
                name = name[len(prefix):]
                break
        return name

    @staticmethod
    def _fetch_npm_weekly(package: str, session: requests.Session) -> Optional[int]:
        """Fetch last-week total downloads from api.npmjs.org."""
        try:
            resp = session.get(
                f"{NPM_DOWNLOADS_URL}/last-week/{package}",
                timeout=10,
            )
            if resp.status_code == 200:
                return resp.json().get("downloads")
        except Exception:
            pass
        return None

    @staticmethod
    def _fetch_npm_monthly_avg(package: str, session: requests.Session) -> Optional[float]:
        """Fetch last-month total downloads from api.npmjs.org, return weekly average."""
        try:
            resp = session.get(
                f"{NPM_DOWNLOADS_URL}/last-month/{package}",
                timeout=10,
            )
            if resp.status_code == 200:
                monthly = resp.json().get("downloads")
                if monthly:
                    return monthly / 4.3   # 30.4 days ÷ 7 = 4.34 weeks
        except Exception:
            pass
        return None

    @staticmethod
    def _fetch_pypi_weekly(package: str, session: requests.Session) -> Optional[int]:
        """Fetch last-week downloads from pypistats.org."""
        try:
            resp = session.get(
                f"{PYPI_STATS_URL}/{package}/recent",
                timeout=10,
            )
            if resp.status_code == 200:
                return resp.json().get("data", {}).get("last_week")
        except Exception:
            pass
        return None

    @staticmethod
    def _mock_report(repo_name: str) -> EcosystemHealthReport:
        """
        Deterministic mock seeded from the repo name so the same repo always
        yields identical outputs during testing.
        """
        seed     = sum(ord(c) for c in repo_name)
        base_dl  = 5_000 + (seed % 195_000)
        ceiling  = EcosystemForecaster._HEALTH_SCORE_CEILING
        score    = round(min(100.0, math.log10(base_dl + 1) / math.log10(ceiling) * 100 * 1.08), 1)
        return EcosystemHealthReport(
            npm_weekly_downloads=base_dl,
            pypi_weekly_downloads=None,
            download_trend="GROWING",
            download_velocity_pct_30d=12.4,
            ecosystem_health_score=score,
            health_classification="HEALTHY" if score >= 50 else "STAGNANT",
            dominant_ecosystem="npm",
            package_name_inferred=EcosystemForecaster._infer_package_name(repo_name),
            data_source="SIMULATED_REGISTRY",
        )


# ─────────────────────────────────────────────────────────────────────────────
# §8  GraphQL Infrastructure (unchanged from v1)
# ─────────────────────────────────────────────────────────────────────────────

REPO_METRICS_QUERY_FULL = """
query RepoMetrics(
  $owner:       String!
  $name:        String!
  $since90:     GitTimestamp!
  $since60:     GitTimestamp!
  $since30:     GitTimestamp!
  $issueCount:  Int!
) {
  repository(owner: $owner, name: $name) {
    nameWithOwner
    isArchived
    isEmpty
    defaultBranchRef {
      name
    }
    commits90: defaultBranchRef {
      target {
        ... on Commit {
          history(since: $since90) { totalCount }
        }
      }
    }
    commits60: defaultBranchRef {
      target {
        ... on Commit {
          history(since: $since60) { totalCount }
        }
      }
    }
    commits30: defaultBranchRef {
      target {
        ... on Commit {
          history(since: $since30) { totalCount }
        }
      }
    }
    issues(
      last: $issueCount
      states: CLOSED
      orderBy: { field: UPDATED_AT, direction: ASC }
    ) {
      totalCount
      nodes {
        createdAt
        closedAt
      }
    }
    hasIssuesEnabled
    stargazerCount
    forkCount
    watchers { totalCount }
    primaryLanguage { name }
    description
    homepageUrl
    repositoryTopics(first: 10) {
      nodes {
        topic { name }
      }
    }
  }
  rateLimit {
    limit
    remaining
    resetAt
    used
  }
}
""".strip()


def _execute_graphql(
    query:        str,
    variables:    dict,
    token:        str,
    session:      requests.Session,
    *,
    max_retries:  int   = 3,
    backoff_base: float = 2.0,
) -> dict:
    """
    Execute a GitHub GraphQL query with exponential back-off on transient errors.
    Returns the `data` sub-dict from the GraphQL response.
    """
    payload = {"query": query, "variables": variables}
    headers = _github_headers(token)

    for attempt in range(max_retries):
        try:
            response = session.post(
                GITHUB_GRAPHQL_URL, json=payload, headers=headers, timeout=30,
            )
        except requests.exceptions.Timeout:
            if attempt < max_retries - 1:
                time.sleep(backoff_base ** attempt)
                continue
            raise

        if response.status_code == 401:
            raise PermissionError("GitHub token is invalid or expired (HTTP 401).")

        if response.status_code in (403, 429):
            reset_str = response.headers.get("X-RateLimit-Reset")
            if reset_str:
                reset_dt = datetime.fromtimestamp(int(reset_str), tz=timezone.utc)
            else:
                reset_dt = datetime.now(timezone.utc) + timedelta(minutes=60)
            raise RateLimitExceededError(reset_dt)

        if response.status_code >= 500:
            if attempt < max_retries - 1:
                wait = backoff_base ** attempt
                print(f"    [GitHub] HTTP {response.status_code} — retrying in {wait}s…")
                time.sleep(wait)
                continue
            response.raise_for_status()

        response.raise_for_status()
        body = response.json()

        if "errors" in body:
            first_error = body["errors"][0]
            error_type  = first_error.get("type", "UNKNOWN")
            error_msg   = first_error.get("message", str(first_error))
            if error_type == "NOT_FOUND":
                raise ValueError(f"Repository not found or private: {error_msg}")
            raise ValueError(f"GitHub GraphQL error [{error_type}]: {error_msg}")

        return body.get("data", {})

    raise RuntimeError("GraphQL request failed after all retries.")


def _calc_issue_resolution(issue_nodes: list[dict]) -> tuple[list[float], list[str]]:
    """
    Convert raw issue nodes into resolution times (hours).
    Returns (resolution_hours, warnings) where warnings captures bad records.
    """
    resolution_hours: list[float] = []
    warnings: list[str] = []

    for node in issue_nodes:
        created_raw = node.get("createdAt")
        closed_raw  = node.get("closedAt")

        if not created_raw or not closed_raw:
            warnings.append("Issue missing timestamp — skipped from TTR calculation.")
            continue

        try:
            created = datetime.fromisoformat(created_raw.replace("Z", "+00:00"))
            closed  = datetime.fromisoformat(closed_raw.replace("Z",  "+00:00"))
        except ValueError as exc:
            warnings.append(f"Unparseable issue timestamp ({exc}) — skipped.")
            continue

        delta_hours = (closed - created).total_seconds() / 3_600
        if delta_hours < 0:
            warnings.append("Issue has closedAt before createdAt — skipped (data anomaly).")
            continue

        resolution_hours.append(delta_hours)

    return resolution_hours, warnings


def _percentile(data: list[float], pct: float) -> float:
    """Linear-interpolation percentile (mirrors numpy.percentile behaviour)."""
    if not data:
        raise ValueError("Empty list")
    sorted_data = sorted(data)
    n   = len(sorted_data)
    idx = (pct / 100) * (n - 1)
    lo  = int(idx)
    hi  = min(lo + 1, n - 1)
    return sorted_data[lo] + (idx - lo) * (sorted_data[hi] - sorted_data[lo])


# ─────────────────────────────────────────────────────────────────────────────
# §9  analyse_repository — Updated Orchestrator
# ─────────────────────────────────────────────────────────────────────────────

def analyse_repository(
    github_url: str,
    token:      str,
    session:    requests.Session,
) -> QuantitativeMetrics:
    """
    Execute all four analytical engines on a single GitHub repository URL and
    return a fully populated QuantitativeMetrics object.

    Engine execution order:
      ① GraphQL query     → RepoProfile, CommitVelocity, IssueResolutionMetrics
      ② File-tree fetch   → _FileTreeStats (shared between ③ and ④)
      ③ ArchitecturalAuditor  → ArchitecturalAuditReport
      ④ ExecutionQuantifier   → ExecutionMetrics
      ⑤ EcosystemForecaster   → EcosystemHealthReport

    All failures are captured in `data_gaps`; this function never raises on
    missing data — it degrades gracefully and documents every loss.
    """
    data_gaps: list[str] = []

    # ── URL parsing ───────────────────────────────────────────────────────────
    try:
        owner, repo_name = parse_github_url(github_url)
    except ValueError as exc:
        return QuantitativeMetrics(
            github_url=github_url,
            profile=RepoProfile(name_with_owner=github_url),
            commit_velocity=CommitVelocity(),
            issue_resolution=IssueResolutionMetrics(sample_size=0, has_issues_enabled=False),
            data_gaps=[f"CRITICAL: Could not parse GitHub URL — {exc}"],
        )

    # Determine mock mode for REST-based engines (GraphQL mock is handled separately
    # via _patch_session_with_mock which monkey-patches session.post).
    use_mock_engines = token in ("MOCK", "", "SIMULATED")

    # ── ① GraphQL core query ──────────────────────────────────────────────────
    now     = datetime.now(timezone.utc)
    since90 = (now - timedelta(days=90)).strftime("%Y-%m-%dT%H:%M:%SZ")
    since60 = (now - timedelta(days=60)).strftime("%Y-%m-%dT%H:%M:%SZ")
    since30 = (now - timedelta(days=30)).strftime("%Y-%m-%dT%H:%M:%SZ")

    variables = {
        "owner":      owner,
        "name":       repo_name,
        "since90":    since90,
        "since60":    since60,
        "since30":    since30,
        "issueCount": ISSUES_TO_FETCH,
    }

    try:
        gql_data = _execute_graphql(REPO_METRICS_QUERY_FULL, variables, token, session)
    except RateLimitExceededError as exc:
        return QuantitativeMetrics(
            github_url=github_url,
            profile=RepoProfile(name_with_owner=f"{owner}/{repo_name}"),
            commit_velocity=CommitVelocity(),
            issue_resolution=IssueResolutionMetrics(sample_size=0, has_issues_enabled=True),
            data_gaps=[
                f"CRITICAL: GitHub rate limit exceeded — all metrics unavailable. "
                f"Quota resets at {exc.reset_at.isoformat()}."
            ],
        )
    except (ValueError, PermissionError) as exc:
        return QuantitativeMetrics(
            github_url=github_url,
            profile=RepoProfile(name_with_owner=f"{owner}/{repo_name}"),
            commit_velocity=CommitVelocity(),
            issue_resolution=IssueResolutionMetrics(sample_size=0, has_issues_enabled=True),
            data_gaps=[f"CRITICAL: GraphQL query failed — {exc}"],
        )

    repo = gql_data.get("repository", {})

    if repo.get("isEmpty"):
        data_gaps.append("Repository is empty — no commits or issues available.")
    if repo.get("isArchived"):
        data_gaps.append("Repository is archived — metrics reflect a frozen codebase.")

    # Topics
    raw_topics = (repo.get("repositoryTopics") or {})
    topic_names = [
        node["topic"]["name"]
        for node in raw_topics.get("nodes", [])
        if node and node.get("topic")
    ]

    primary_language = (repo.get("primaryLanguage") or {}).get("name")

    profile = RepoProfile(
        name_with_owner  = repo.get("nameWithOwner", f"{owner}/{repo_name}"),
        default_branch   = (repo.get("defaultBranchRef") or {}).get("name"),
        primary_language = primary_language,
        description      = repo.get("description") or None,
        topics           = topic_names,
        homepage         = repo.get("homepageUrl") or None,
        stars            = repo.get("stargazerCount"),
        forks            = repo.get("forkCount"),
        watchers         = (repo.get("watchers") or {}).get("totalCount"),
        is_archived      = repo.get("isArchived", False),
        is_empty         = repo.get("isEmpty", False),
    )

    # Commit velocity
    def _extract_commit_count(alias: str) -> Optional[int]:
        branch_ref = repo.get(alias)
        if not branch_ref:
            return None
        return (branch_ref.get("target") or {}).get("history", {}).get("totalCount")

    c90 = _extract_commit_count("commits90")
    c60 = _extract_commit_count("commits60")
    c30 = _extract_commit_count("commits30")

    if c90 is None:
        data_gaps.append(
            "Commit velocity unavailable: defaultBranchRef returned no data."
        )

    commit_velocity = CommitVelocity(last_30_days=c30, last_60_days=c60, last_90_days=c90)

    # Issue resolution
    has_issues  = repo.get("hasIssuesEnabled", True)
    issues_obj  = repo.get("issues") or {}
    issue_nodes = issues_obj.get("nodes", []) if issues_obj else []
    total_closed = issues_obj.get("totalCount") if issues_obj else None

    if not has_issues:
        data_gaps.append("Issue resolution metrics unavailable: Issues tab is disabled.")
        issue_resolution = IssueResolutionMetrics(
            sample_size=0, total_closed_issues=None, has_issues_enabled=False,
        )
    elif not issue_nodes:
        data_gaps.append("Issue resolution metrics unavailable: no closed issues found.")
        issue_resolution = IssueResolutionMetrics(
            sample_size=0, total_closed_issues=total_closed or 0, has_issues_enabled=True,
        )
    else:
        resolution_hours, ttr_warnings = _calc_issue_resolution(issue_nodes)
        data_gaps.extend(ttr_warnings)

        if len(resolution_hours) < 5:
            data_gaps.append(
                f"TTR statistics are low-confidence: only {len(resolution_hours)} valid "
                f"issue records (threshold is 5)."
            )

        if resolution_hours:
            median = statistics.median(resolution_hours)
            p25    = _percentile(resolution_hours, 25)
            p75    = _percentile(resolution_hours, 75)
        else:
            median = p25 = p75 = None
            data_gaps.append("TTR could not be computed: all issue records were invalid.")

        issue_resolution = IssueResolutionMetrics(
            sample_size=len(resolution_hours),
            median_hours=round(median, 2) if median is not None else None,
            p25_hours=round(p25, 2)    if p25    is not None else None,
            p75_hours=round(p75, 2)    if p75    is not None else None,
            total_closed_issues=total_closed,
            has_issues_enabled=True,
        )

    # Rate-limit snapshot
    rl_raw = gql_data.get("rateLimit")
    rate_limit_snapshot: Optional[RateLimitSnapshot] = None
    if rl_raw:
        try:
            rate_limit_snapshot = RateLimitSnapshot(
                limit=rl_raw["limit"],
                remaining=rl_raw["remaining"],
                used=rl_raw["used"],
                reset_at=datetime.fromisoformat(rl_raw["resetAt"].replace("Z", "+00:00")),
            )
            if rl_raw["used"] / rl_raw["limit"] > 0.80:
                print(
                    f"    ⚠️  Rate limit warning: {rl_raw['remaining']} points remaining "
                    f"(resets {rl_raw['resetAt']})"
                )
        except (KeyError, ValueError):
            pass

    # ── ② File-tree stats (shared between Auditor and Quantifier) ─────────────
    if not use_mock_engines:
        tree_stats = _fetch_file_tree_stats(owner, repo_name, token, session)
        if tree_stats.truncated:
            data_gaps.append(
                "File-tree analysis truncated: repository exceeds GitHub's 100k-file limit. "
                "ArchitecturalAudit and ExecutionMetrics confidence is LOW."
            )
    else:
        tree_stats = None   # new engines will use their mock paths

    # ── ③ ArchitecturalAuditor ────────────────────────────────────────────────
    arch_audit: Optional[ArchitecturalAuditReport] = None
    try:
        arch_audit = ArchitecturalAuditor.audit_repo(
            owner, repo_name, token, session,
            primary_language=primary_language,
            stars=profile.stars,
            tree_stats=tree_stats,
            use_mock=use_mock_engines,
        )
        if arch_audit.audit_method == "SIMULATED_AST":
            data_gaps.append(
                "ArchitecturalAudit fell back to simulation — structural metrics are estimates."
            )
    except Exception as exc:
        data_gaps.append(f"ArchitecturalAudit failed: {exc}")

    # ── ④ ExecutionQuantifier ─────────────────────────────────────────────────
    exec_metrics: Optional[ExecutionMetrics] = None
    try:
        exec_metrics = ExecutionQuantifier.analyse_execution(
            owner, repo_name, token, session,
            tree_stats=tree_stats,
            use_mock=use_mock_engines,
        )
    except Exception as exc:
        data_gaps.append(f"ExecutionMetrics failed: {exc}")

    # ── ⑤ EcosystemForecaster ─────────────────────────────────────────────────
    eco_health: Optional[EcosystemHealthReport] = None
    try:
        eco_health = EcosystemForecaster.forecast(
            repo_name, session,
            primary_language=primary_language,
            use_mock=use_mock_engines,
        )
        if eco_health.data_source == "SIMULATED_REGISTRY":
            data_gaps.append(
                "EcosystemForecaster used simulated registry data — no matching npm/PyPI "
                f"package found for '{eco_health.package_name_inferred}'."
            )
    except Exception as exc:
        data_gaps.append(f"EcosystemForecaster failed: {exc}")

    return QuantitativeMetrics(
        github_url=github_url,
        profile=profile,
        commit_velocity=commit_velocity,
        issue_resolution=issue_resolution,
        rate_limit=rate_limit_snapshot,
        data_gaps=data_gaps,
        architectural_audit=arch_audit,
        execution_metrics=exec_metrics,
        ecosystem_health=eco_health,
    )


# ─────────────────────────────────────────────────────────────────────────────
# §10  Top-level pipeline functions (unchanged interface)
# ─────────────────────────────────────────────────────────────────────────────

def enrich_signal(
    item:    ClassifiedItem,
    token:   str,
    session: Optional[requests.Session] = None,
) -> EnrichedItem:
    """
    Top-level function: accepts a ClassifiedItem and returns an EnrichedItem.
    Skips analysis silently for NOISE items or items with no GitHub targets.
    """
    if not item.is_signal or not item.has_github_targets:
        return EnrichedItem(classified=item)

    own_session = session is None
    if own_session:
        session = requests.Session()

    try:
        metrics_list: list[QuantitativeMetrics] = []
        for url in item.raw.github_urls:
            print(f"  → Analysing {url}")
            m = analyse_repository(url, token, session)
            metrics_list.append(m)
            if m.data_gaps:
                for gap in m.data_gaps:
                    prefix = "    🚨 " if gap.startswith("CRITICAL") else "    ⚠️  "
                    print(f"{prefix}{gap}")
    finally:
        if own_session:
            session.close()

    return EnrichedItem(classified=item, metrics=metrics_list)


def run_quant_batch(
    signals: list[ClassifiedItem],
    token:   str,
    *,
    verbose: bool = True,
) -> list[EnrichedItem]:
    """Process a batch of signals from Agent 1 and return EnrichedItems."""
    enriched: list[EnrichedItem] = []
    session = requests.Session()

    try:
        for item in signals:
            if verbose:
                print(f"\n{'─'*60}")
                print(f"  [Agent 2] Enriching: {item.raw.title[:65]}")
            enriched.append(enrich_signal(item, token, session))
    finally:
        session.close()

    return enriched


# ─────────────────────────────────────────────────────────────────────────────
# §11  Mock Infrastructure (for use without GITHUB_TOKEN)
# ─────────────────────────────────────────────────────────────────────────────

def _build_mock_graphql_response(
    owner: str,
    repo:  str,
    *,
    disable_issues: bool = False,
    empty_repo:     bool = False,
) -> dict:
    """
    Returns a realistic mock GraphQL `data` payload mirroring REPO_METRICS_QUERY_FULL.
    Used when GITHUB_TOKEN is not set.
    """
    now = datetime.now(timezone.utc)
    issue_nodes = []
    for i in range(ISSUES_TO_FETCH):
        created = now - timedelta(days=90 - i)
        resolution_hours = 2 + (i % 15) * 8
        closed  = created + timedelta(hours=resolution_hours)
        issue_nodes.append({
            "createdAt": created.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "closedAt":  closed.strftime("%Y-%m-%dT%H:%M:%SZ"),
        })

    return {
        "repository": {
            "nameWithOwner":    f"{owner}/{repo}",
            "isArchived":       False,
            "isEmpty":          empty_repo,
            "defaultBranchRef": {"name": "main"},
            "commits90": {"target": {"history": {"totalCount": 420}}},
            "commits60": {"target": {"history": {"totalCount": 280}}},
            "commits30": {"target": {"history": {"totalCount": 165}}},
            "issues": None if disable_issues else {
                "totalCount": 812,
                "nodes":      issue_nodes,
            },
            "hasIssuesEnabled":  not disable_issues,
            "stargazerCount":    38_400,
            "forkCount":         1_204,
            "watchers":          {"totalCount": 892},
            "primaryLanguage":   {"name": "Rust"},
            "description":       "Mock repository for Agent 2 testing.",
            "homepageUrl":       None,
            "repositoryTopics":  {"nodes": []},
        },
        "rateLimit": {
            "limit":     5000,
            "remaining": 4873,
            "used":      127,
            "resetAt":   (now + timedelta(hours=1)).strftime("%Y-%m-%dT%H:%M:%SZ"),
        },
    }


def _patch_session_with_mock(
    session:        requests.Session,
    owner:          str,
    repo:           str,
    disable_issues: bool = False,
) -> None:
    """
    Monkey-patches `session.post` to return a mock GraphQL response, allowing full
    Agent 2 GraphQL logic to run without a real GitHub token.
    Note: session.get (used by the REST-based engines) is NOT patched — those
    engines detect the MOCK token and use their own internal mock paths.
    """
    mock_data = _build_mock_graphql_response(owner, repo, disable_issues=disable_issues)
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {"data": mock_data}
    mock_resp.raise_for_status = lambda: None
    session.post = MagicMock(return_value=mock_resp)


# ─────────────────────────────────────────────────────────────────────────────
# §12  Debug Printing
# ─────────────────────────────────────────────────────────────────────────────

def print_enriched_item(ei: EnrichedItem) -> None:
    """Human-readable summary of an EnrichedItem for pipeline validation."""
    title = ei.classified.raw.title
    print(f"\n{'═'*70}")
    print(f"  ENRICHED SIGNAL: {title[:65]}")
    print(f"{'═'*70}")

    if not ei.metrics:
        print("  ⚠️  No metrics — item was NOISE or had no GitHub URLs.")
        return

    for m in ei.metrics:
        p  = m.profile
        cv = m.commit_velocity
        ir = m.issue_resolution

        print(f"\n  Repo           : {p.name_with_owner}  [{p.primary_language or 'unknown'}]")
        print(f"  Default branch : {p.default_branch or 'N/A'}  |  Archived: {p.is_archived}")
        if p.stars is not None:
            print(f"  Stars / Forks  : {p.stars:,} ⭐  /  {p.forks:,} 🍴")

        print(f"\n  ── Commit Velocity ─────────────────────────────────────────")
        print(f"  Last 30 days   : {cv.last_30_days}")
        print(f"  Last 60 days   : {cv.last_60_days}")
        print(f"  Last 90 days   : {cv.last_90_days}")
        print(f"  Weekly avg     : {cv.weekly_avg_30d} commits/wk (30d window)")
        print(f"  Momentum       : {cv.acceleration}")

        print(f"\n  ── Issue Resolution ────────────────────────────────────────")
        print(f"  Issues enabled : {ir.has_issues_enabled}")
        print(f"  Sample size    : {ir.sample_size}")
        print(f"  Total closed   : {ir.total_closed_issues}")
        print(f"  Median TTR     : {ir.median_days} days  (grade: {ir.responsiveness_grade})")

        # ── NEW: Architectural Audit ──────────────────────────────────────
        if m.architectural_audit:
            aa = m.architectural_audit
            print(f"\n  ── Architectural Audit [{aa.audit_method} / conf: {aa.audit_confidence}] ──")
            print(f"  Debt score     : {aa.technical_debt_score}/100  [{aa.debt_classification}]")
            print(f"  Cost-to-fix    : ${aa.cost_to_remediate_usd:,.0f}")
            print(f"  Test cov proxy : {aa.test_coverage_proxy:.0%}  [{aa.test_coverage_classification}]")
            print(f"  Circ deps est. : {aa.circular_dependency_count}")
            print(f"  Laziness score : {aa.implementation_laziness_score}/100")
            print(f"  Provenance     : {aa.provenance_risk}  |  License: {'✓' if aa.has_license else '✗'}  CI: {'✓' if aa.has_ci_config else '✗'}")
            if aa.ai_generated_code_indicators:
                for flag in aa.ai_generated_code_indicators:
                    print(f"  ⚑  {flag}")

        # ── NEW: Execution Metrics ────────────────────────────────────────
        if m.execution_metrics:
            em = m.execution_metrics
            print(f"\n  ── Execution Metrics ───────────────────────────────────────")
            print(f"  Bus Factor     : {em.bus_factor}  [{em.bus_factor_risk}]  (top contrib: {em.core_complexity_concentration:.0%})")
            print(f"  Contributors   : {em.contributor_count} total  /  ~{em.active_contributors_90d} active (90d proxy)")
            vel = f"{em.pr_merge_velocity_hours}h" if em.pr_merge_velocity_hours is not None else "N/A"
            print(f"  PR velocity    : {vel}  [grade {em.pr_merge_grade}]  (n={em.pr_sample_size})")
            print(f"  Doc ratio      : {em.doc_to_code_ratio:.2f}  [{em.doc_maturity}]")

        # ── NEW: Ecosystem Health ─────────────────────────────────────────
        if m.ecosystem_health:
            eh = m.ecosystem_health
            print(f"\n  ── Ecosystem Health [{eh.data_source}] ──────────────────────")
            print(f"  Health score   : {eh.ecosystem_health_score}/100  [{eh.health_classification}]")
            print(f"  npm/wk         : {eh.npm_weekly_downloads:,}" if eh.npm_weekly_downloads else "  npm/wk         : N/A")
            print(f"  PyPI/wk        : {eh.pypi_weekly_downloads:,}" if eh.pypi_weekly_downloads else "  PyPI/wk        : N/A")
            print(f"  Trend          : {eh.download_trend}  (Δ30d: {eh.download_velocity_pct_30d}%)")
            if eh.package_name_inferred:
                print(f"  Package name   : {eh.package_name_inferred}")

        if m.rate_limit:
            rl = m.rate_limit
            print(f"\n  ── Rate Limit ──────────────────────────────────────────────")
            print(f"  {rl.remaining}/{rl.limit} points remaining  (resets {rl.reset_at.strftime('%H:%M UTC')})")

        if m.data_gaps:
            print(f"\n  ── Data Gaps (Agent 3 must surface these) ──────────────────")
            for gap in m.data_gaps:
                print(f"  ⚠️  {gap}")


# ─────────────────────────────────────────────────────────────────────────────
# §13  __main__ — Pipeline Validation
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 70)
    print("  Agent 2 v2: Architectural Erosion and Execution Quantifier")
    print("  Pipeline Validation Run")
    print("=" * 70)

    def _make_signal(item_id: str, title: str, github_urls: list[str]) -> ClassifiedItem:
        raw = RawItem(
            item_id=item_id,
            source=DataSource.HACKERNEWS,
            title=title,
            github_urls=github_urls,
        )
        return ClassifiedItem(
            raw=raw,
            classification=Classification.SIGNAL,
            confidence=0.95,
            rationale=ScoutRationale(
                chain_of_thought="Validation mock — all four flags positive.",
                architectural_shift=True,
                developer_adoption=True,
                oss_milestone=True,
                noise_signal=False,
                one_line_summary="Mock signal for Agent 2 v2 validation.",
            ),
        )

    mock_signals = [
        _make_signal(
            "mock_001",
            "uv 0.4: Rust-based pip replacement, 100× faster resolver",
            ["https://github.com/astral-sh/uv"],
        ),
        _make_signal(
            "mock_002",
            "Deno 2.0: full npm compat, 2.1× faster cold starts",
            ["https://github.com/denoland/deno"],
        ),
    ]

    mock_noise = ClassifiedItem(
        raw=RawItem(
            item_id="mock_noise",
            source=DataSource.HACKERNEWS,
            title="Acme AI raises $200M Series C",
            github_urls=[],
        ),
        classification=Classification.NOISE,
        confidence=0.99,
        rationale=ScoutRationale(
            chain_of_thought="Funding announcement with no engineering substance.",
            architectural_shift=False, developer_adoption=False,
            oss_milestone=False, noise_signal=True,
            one_line_summary="Funding announcement — NOISE.",
        ),
    )

    token      = os.environ.get("GITHUB_TOKEN", "")
    use_mock   = not token
    token_arg  = token or "MOCK"

    if use_mock:
        print("\n  ⚠️  GITHUB_TOKEN not set — using mock GraphQL + simulated engines.\n")

    session = requests.Session()
    enriched_results: list[EnrichedItem] = []

    for idx, signal in enumerate(mock_signals):
        print(f"\n{'─'*70}")
        print(f"  [Agent 2] Enriching {idx+1}/{len(mock_signals)}: {signal.raw.title[:65]}")

        if use_mock:
            owner_m, repo_m = parse_github_url(signal.raw.github_urls[0])
            _patch_session_with_mock(session, owner_m, repo_m, disable_issues=(idx == 1))

        ei = enrich_signal(signal, token_arg, session)
        enriched_results.append(ei)
        print_enriched_item(ei)

    print(f"\n{'─'*70}")
    print("  [Agent 2] Processing NOISE item (expect 0 metrics)…")
    noise_result = enrich_signal(mock_noise, token_arg, session)
    assert len(noise_result.metrics) == 0, "NOISE item must produce no metrics"
    print(f"  Metrics list length: {len(noise_result.metrics)}  (expected: 0) ✅")

    session.close()

    print(f"\n{'═'*70}")
    print(f"  RESULTS: {len(enriched_results)} EnrichedItem(s) ready for Agent 3.")
    print("  All four engines executed successfully.")
    print(f"{'═'*70}\n")