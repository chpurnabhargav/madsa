"""The verification layer: the checks that keep the framework's claims honest.

This module exists because of a gap between what the project claimed and what the
code showed. The contribution is described as *verification as an architectural
layer rather than a prompting strategy* -- yet the checks lived as a scattering
of ``warnings.append`` calls across nine agent files. They could not be
enumerated, counted, or tested as a set, and the documented "ten mechanisms" was
an assertion no reader could confirm. Two agents had also grown near-identical
copies of the same negation-aware matcher.

A layer that cannot be listed is not a layer. So the checks live here, each
registered with a name and a description of what it compares, and
``MECHANISMS`` is the single enumerable answer to "what does this framework
actually verify?".

Every check follows one rule: **compare what the model said against what a tool
computed.** None of them asks a model whether its own output is correct, because
a confidently wrong claim reviews itself as correct. That is the distinction the
project rests on, and keeping the checks in one module makes it inspectable.
"""
from __future__ import annotations

import functools
import re
import math
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any, Callable, Iterator

from config import CONFIG
from dataclasses import dataclass

if TYPE_CHECKING:  # pragma: no cover - typing only
    from state import WorkflowState

# --------------------------------------------------------------------------- #
# The registry
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Mechanism:
    """One verification mechanism: what it checks and what it checks against."""

    name: str
    stage: str
    checks: str          # what the model might get wrong
    compares_against: str  # the computed ground truth it is held to


MECHANISMS: list[Mechanism] = []

#: How many times each mechanism has found something during the current run.
#: The orchestrator clears it when a run starts and copies it into the run's
#: state when the run ends.
#:
#: The evaluation previously counted corrections by searching agent warnings
#: for marker phrases. That saw seven of the twelve mechanisms -- segmentation,
#: feature-benefit, invented-feature, non-linear-signal and suppressed-result
#: findings were never counted -- and it broke silently whenever a warning was
#: reworded. Counting at the mechanism itself cannot miss one.
FIRINGS: dict[str, int] = {}

# Set while `auditing()` is active: the checks then run whatever the ablation
# switch says, and what they find is measured rather than counted as a
# correction the framework applied.
_AUDITING = False


def register(name: str, stage: str, checks: str, compares_against: str) -> Callable:
    """Record a check in the registry and count what it finds.

    Registration makes the set of mechanisms enumerable, so the count in the
    documentation is derived from the code rather than maintained beside it and
    allowed to drift. The wrapper records each call that finds something -- a
    non-empty list, a True, a refusal reason -- in `FIRINGS`, so the evaluation
    can report how often each mechanism intervened rather than infer it.
    """
    def decorator(fn: Callable) -> Callable:
        MECHANISMS.append(Mechanism(name, stage, checks, compares_against))

        @functools.wraps(fn)
        def checked(*args: Any, **kwargs: Any) -> Any:
            result = fn(*args, **kwargs)
            if result and not _AUDITING:
                FIRINGS[name] = FIRINGS.get(name, 0) + 1
            return result

        checked.mechanism = name  # type: ignore[attr-defined]
        return checked
    return decorator


def reset_firings() -> None:
    """Clear the per-run counts. Called by the orchestrator as a run starts."""
    FIRINGS.clear()


def firings() -> dict[str, int]:
    """Every registered mechanism with how often it fired, zeros included."""
    return {m.name: FIRINGS.get(m.name, 0) for m in MECHANISMS}


@contextmanager
def auditing() -> Iterator[None]:
    """Run the checks to measure a finished run, not to correct it.

    The ablation switches the layer off, and the layer's own counts then read
    zero by construction -- which says nothing about what reached the reader.
    Inside this block every check runs regardless of the switch and nothing is
    counted as a correction, so the same measurement can be applied to the
    output of a run with the layer on and a run with it off.
    """
    global _AUDITING
    previous, _AUDITING = _AUDITING, True
    try:
        yield
    finally:
        _AUDITING = previous


def _off() -> bool:
    """Whether the verification layer has been switched off for an ablation.

    Read here rather than at each call site, so the ablation cannot half-apply.
    Every detector below returns its empty value when this is true, which is
    what a framework with no verification layer produces: not an error, just
    silence. An audit overrides it, because an audit measures rather than
    corrects.
    """
    return not CONFIG.verification_enabled and not _AUDITING


# --------------------------------------------------------------------------- #
# Shared language analysis
# --------------------------------------------------------------------------- #

# Negations that invert a claim. Correctly reporting that data does *not*
# separate into groups uses the same words as claiming that it does, so a plain
# substring search flags the honest statement exactly as readily as the
# dishonest one. This list is the difference between a check and a nuisance.
NEGATIONS = (
    "no ", "not ", "n't", "without", "fails to", "does not", "did not",
    "cannot", "too weak", "only", "lack", "absence of", "rather than",
    "would have", "did little", "neither", "insufficient", "unable",
)

# Sixty characters reaches back across a clause without crossing into the
# previous sentence, where an unrelated negation would produce a false pass.
_NEGATION_WINDOW = 60


def asserts_any(text: str, terms: tuple[str, ...]) -> list[str]:
    """Terms used as affirmative claims, ignoring negated occurrences.

    Returns the terms the text actually asserts. A term appearing only inside a
    negated clause is not an assertion of it, and reporting it would punish the
    agent for being honest.
    """
    lowered = text.lower()
    found: list[str] = []
    for term in terms:
        start = 0
        while (index := lowered.find(term, start)) != -1:
            window = lowered[max(0, index - _NEGATION_WINDOW):index]
            window = window.rsplit(".", 1)[-1]
            if not any(negation in window for negation in NEGATIONS):
                found.append(term)
                break
            start = index + len(term)
    return found


# --------------------------------------------------------------------------- #
# Numeric provenance
# --------------------------------------------------------------------------- #

# A leading minus is a sign only where it cannot be a hyphen: not directly
# after a letter, digit or decimal point. Without that condition the range
# "30-40" was read as 30 and -40, a date "2024-01-15" produced -1 and -15, and
# each negative value was then reported as a figure nothing had computed.
_NUMBER = re.compile(r"(?:(?<![\w.])-)?\d[\d,]*\.?\d*")


def numbers_in(text: str) -> list[tuple[str, float, float]]:
    """Numbers stated in prose, as (original, value, tolerance) triples.

    The tolerance comes from how precisely the number was written rather than
    being fixed. A report writing "0.81" is rounding to two places and should
    match a measured 0.8099; one writing "96.4" claims a tenth and must not
    match a computed 96. A fixed relative tolerance would accept an invented
    value that merely lands near a real one.
    """
    out = []
    # Models emit the typographic minus (U+2212) as readily as the ASCII one;
    # read as unsigned it would never match the negative value it renders.
    for raw in _NUMBER.findall(text.replace("−", "-")):
        cleaned = raw.replace(",", "")
        try:
            value = float(cleaned)
        except ValueError:
            continue
        decimals = len(cleaned.split(".")[1]) if "." in cleaned else 0
        out.append((raw, value, 0.5 * (10 ** -decimals)))
    return out


def collect_numbers(obj: Any, into: set[str]) -> None:
    """Gather every numeric value appearing anywhere in computed results.

    This becomes the set of values a report is permitted to state. Booleans are
    skipped: in Python they are integers, and counting True as 1 would let a
    report state "1" about anything.
    """
    if isinstance(obj, dict):
        for value in obj.values():
            collect_numbers(value, into)
    elif isinstance(obj, (list, tuple)):
        for value in obj:
            collect_numbers(value, into)
    elif isinstance(obj, bool):
        return
    elif isinstance(obj, (int, float)):
        into.add(normalise(obj))
    elif isinstance(obj, str):
        for match in _NUMBER.findall(obj):
            try:
                into.add(normalise(float(match.replace(",", ""))))
            except ValueError:
                continue


def normalise(value: float) -> str:
    """A comparable string form that preserves precision.

    An earlier version rounded to four decimal places, which destroyed the
    information the check then needed: a report citing 0.00801 was compared
    against a computed 0.008014 already flattened to 0.008, and the difference
    exceeded the tolerance derived from the five decimals written. The checker
    manufactured its own false positives, and flagging correct numbers is worse
    than not checking, because it teaches the reader to ignore the warning.
    """
    # NaN and the infinities reach here from real calculations -- a Pearson
    # correlation on a constant column, a Spark skewness on a column with no
    # variation -- and `int()` raises on both, failing the entire report stage
    # over a number that was never going to be cited anyway. They are rendered
    # rather than converted: a report claiming "nan" should still be matched
    # against the "nan" that was computed, not crash the check that would have
    # caught it.
    if not math.isfinite(value):
        return f"{value:.10g}"
    if value == int(value):
        return str(int(value))
    return f"{value:.10g}"


def as_floats(allowed: set[str]) -> list[float]:
    """The computed values as floats, for numeric comparison."""
    values = []
    for candidate in allowed:
        try:
            values.append(float(candidate))
        except ValueError:
            continue
    return values


def is_supported(value: float, tolerance: float, computed: list[float]) -> bool:
    """Whether a stated number is a faithful rendering of a computed one."""
    # Small integers are structural ("3 classes", "two sections") rather than
    # claims about measurements; checking them adds noise without catching errors.
    if value == int(value) and abs(value) <= 12:
        return True

    for other in computed:
        if abs(value - other) <= tolerance:
            return True
        # A computed fraction stated as a percentage, or the reverse.
        if abs(value - other * 100) <= tolerance:
            return True
        # The reverse: a percentage computed, a fraction written. This divided
        # `value` where it needed to multiply, so the branch could never match
        # -- a report correctly writing 0.15 for a computed 15.0 was recorded as
        # citing a number no stage produced, which the validator promotes to a
        # critical issue and which forces a revision round that cannot succeed,
        # because there is nothing wrong to fix. A checker that flags correct
        # numbers is worse than no checker, for the reason `normalise` gives:
        # it teaches the reader to ignore the warnings.
        if abs(value * 100 - other) <= tolerance * 100:
            return True
    return False


# --------------------------------------------------------------------------- #
# The mechanisms
# --------------------------------------------------------------------------- #

# The union of the two lists that had drifted apart in the report and EDA
# agents. Neither was wrong; they were simply maintained separately, so a term
# added to one never reached the other. "makes" and "produces" came from the EDA
# copy and are kept.
CAUSAL_TERMS = (
    "causes", "caused", "causing", "leads to", "led to", "results in",
    "resulted in", "drives", "driven by", "due to", "because of",
    "responsible for", "the reason for", "makes", "produces",
)

SEGMENT_CLAIMS = ("segment", "distinct group", "natural group", "clear cluster",
                  "well-separated", "subpopulation")

BENEFIT_CLAIMS = ("improved", "improves", "boost", "increased accuracy",
                  "better performance", "enhanced", "gain in",
                  "stronger predictor", "more predictive")

SIGNAL_ADMISSIONS = ("no signal", "not distinguishable", "no predictive", "chance",
                     "did not beat", "no better than", "not significant",
                     "no reliable")


# The registration below previously decorated `_off`, the ablation switch,
# rather than this check. It was harmless while registration did nothing at
# call time; with counting, every read of the switch would have been recorded
# as an unsupported number.
@register("unsupported_numbers", "report",
          "a number stated in the report that no stage computed",
          "every numeric value any tool computed during the run")
def unsupported_numbers(text: str, computed_values: set[str]) -> list[str]:
    """Numbers in the text that match no computed value."""
    if _off():
        return []
    computed = as_floats(computed_values)
    return sorted({
        raw for raw, value, tolerance in numbers_in(text)
        if not is_supported(value, tolerance, computed)
    })


@register("causal_language", "report",
          "an association described as though it were a cause",
          "a fixed list of causal verbs and phrases")
def causal_language(text: str) -> list[str]:
    """Causal phrasing used to describe observational findings."""
    if _off():
        return []
    lowered = text.lower()
    return [term for term in CAUSAL_TERMS if term in lowered]


@register("hallucinated_columns", "any",
          "a column named in prose that is not in the dataset",
          "the actual column names of the frame")
def hallucinated_columns(named: list[str], actual: list[str]) -> list[str]:
    """Column names cited by the model that do not exist."""
    if _off():
        return []
    available = set(actual)
    return [name for name in named
            if isinstance(name, str) and name not in available]


@register("overstated_segmentation", "multivariate",
          "weak clustering presented as established segmentation",
          "the measured silhouette score")
def overstated_segmentation(text: str) -> list[str]:
    """Segmentation claimed affirmatively despite a weak silhouette."""
    if _off():
        return []
    return asserts_any(text, SEGMENT_CLAIMS)


@register("overstated_feature_benefit", "features",
          "derived features described as helpful when measurement says otherwise",
          "the cross-validated gain from the derived features")
def overstated_feature_benefit(text: str) -> list[str]:
    """A benefit claimed affirmatively despite a negligible measured gain."""
    if _off():
        return []
    return asserts_any(text, BENEFIT_CLAIMS)


@register("suppressed_negative_result", "report",
          "a negative modelling result softened into apparent success",
          "the beats_baseline flag and permutation test")
def suppressed_negative_result(text: str, beats_baseline: bool | None) -> bool:
    """Whether a no-signal result failed to survive into the report.

    Returns True when the modelling stage found nothing and the report does not
    say so. A negative result is a real result; softening it is the failure this
    framework exists to prevent.
    """
    if _off():
        return False
    if beats_baseline is not False:
        return False
    lowered = text.lower()
    return not any(phrase in lowered for phrase in SIGNAL_ADMISSIONS)


@register("unsupported_non_linear_signal", "multivariate",
          "a feature cited as carrying non-linear signal that ranks uninformative",
          "the mutual information ranking")
def unsupported_non_linear_signal(claimed: list[str],
                                  informative: set[str]) -> list[str]:
    """Features named as non-linearly informative that the ranking contradicts."""
    if _off():
        return []
    return [feature for feature in claimed
            if isinstance(feature, str) and feature not in informative]


@register("invented_features", "features",
          "a feature cited as useful that was never derived",
          "the list of features actually built")
def invented_features(claimed: list[str], built: set[str]) -> list[str]:
    """Features cited by the model that the engineering stage never produced."""
    if _off():
        return []
    return [feature for feature in claimed
            if isinstance(feature, str) and feature not in built]


@register("overridden_signal_claim", "modelling",
          "a signal assessment or model recommendation that contradicts the "
          "measurements",
          "the baseline lift and the permutation test")
def overridden_signal_claim(claimed: str, beats_baseline: bool,
                            recommended_model: Any = None) -> bool:
    """Whether a claimed signal, or a recommended model, contradicts what was measured.

    The strongest of the mechanisms, because it does not warn and move on: the
    modelling agent replaces the claim with 'none' and withdraws the
    recommendation. A negative result is the one a language model is most
    likely to soften, and the one that most misleads a reader when softened.

    The recommendation is covered as well as the assessment. Checking only for
    "strong" or "moderate" let a model assessed as "weak" still be recommended
    when nothing beat the baseline -- a recommendation of a model that is
    indistinguishable from guessing, which the validator then flagged as a
    critical issue that no rewrite of the report could clear.
    """
    if _off() or beats_baseline:
        return False
    return (str(claimed).lower() in ("strong", "moderate")
            or bool(recommended_model))


@register("unsupported_importance_claim", "modelling",
          "a feature cited as important that permutation importance cannot "
          "distinguish from noise",
          "held-out permutation importance and its standard deviation")
def unsupported_importance_claim(claimed: list[str],
                                 informative: set[str]) -> list[str]:
    """Features named as important that the importance analysis does not support.

    Permutation importance below its own standard deviation is noise: shuffling
    the feature moved the score no more than shuffling does by chance. A model
    naming such a feature as a driver is describing a pattern the measurement
    did not find.

    This check existed as an unregistered `warnings.append` in the modelling
    agent, which put it outside the ablation switch and outside the count this
    module exists to make honest.
    """
    if _off():
        return []
    return [feature for feature in claimed
            if isinstance(feature, str) and feature not in informative]


@register("overstated_effect", "eda",
          "an insight marked high importance resting on a negligible effect",
          "the measured effect sizes (eta-squared, Cramer's V, correlation)")
def overstated_effect(effect_sizes: dict[str, float],
                      threshold: float = 0.06) -> bool:
    """Whether every effect an insight rests on is below the reporting floor.

    Significance is not importance. On a large dataset everything is
    significant, so an insight is judged on effect size instead.
    """
    if _off():
        return False
    return bool(effect_sizes) and all(v < threshold for v in effect_sizes.values())


#: Repairs that change values, and so need a numeric column to act on.
_NUMERIC_ONLY = frozenset({"impute_median", "clip_outliers"})
_IMPUTATIONS = frozenset({"impute_median", "impute_mode"})


@register("refused_cleaning_action", "quality",
          "a repair that would fabricate data, alter the target, or act on a "
          "column it cannot apply to",
          "the allow-list of executable repairs, the target column, and each "
          "column's measured type and missing fraction")
def refused_cleaning_action(action: str, permitted: set[str], *,
                            column: str | None = None,
                            columns: Any = None,
                            target: str | None = None,
                            missing_fraction: float | None = None,
                            numeric: bool | None = None) -> str | None:
    """Why a requested repair must not run, or None when it may.

    The quality agent proposes repairs; it does not perform them. Every rule
    that stands between a proposal and its execution lives here, so the
    ablation switch reaches all of them. They previously ran inline in the
    quality agent, where switching verification off left every refusal in
    place and the ablation measured a framework that still screened its
    repairs.

    The rules, each enforced rather than requested in the prompt:

    - An action outside the allow-list could be any transformation at all.
    - A column that does not exist cannot be repaired.
    - The target is the thing being predicted, not an input to tidy: clipping a
      regression target's tails or rewriting a classification target's labels
      changes what every score in the report is measured against.
    - Imputing a column missing more than half its values manufactures data
      rather than recovering it.
    - A median or an outlier fence is undefined on text.
    """
    if _off():
        return None
    if action not in permitted:
        return f"Refused unsupported action '{action}'."
    if action == "drop_duplicates":
        return None
    if columns is not None:
        try:
            exists = column in columns
        except TypeError:  # an unhashable name, as model output sometimes is
            exists = False
        if not exists:
            return f"Refused '{action}' on '{column}': no such column."
    if target is not None and column == target:
        return (f"Refused '{action}' on target column '{column}': modifying the "
                "target would change what every score in this report is "
                "measured against.")
    if action in _IMPUTATIONS and missing_fraction is not None and missing_fraction > 0.5:
        return (f"Refused '{action}' on '{column}': {missing_fraction:.0%} missing "
                "is too much to impute without fabricating data.")
    if action in _NUMERIC_ONLY and numeric is False:
        return f"Refused '{action}' on non-numeric column '{column}'."
    return None


@register("failed_chart_request", "visualization",
          "a chart specification the data cannot support",
          "the actual columns, types and cardinality of the frame")
def failed_chart_request(error: str | None) -> bool:
    """Whether a requested chart could not be built from the real data."""
    if _off():
        return False
    return error is not None


# --------------------------------------------------------------------------- #
# What a report may cite, and what reached the reader
# --------------------------------------------------------------------------- #

def computed_values(state: "WorkflowState") -> set[str]:
    """Every number a tool computed during the run: the figures a report may cite.

    Drawn from each stage's *evidence*, which the `AgentOutput` contract
    reserves for computed values, plus the handful of facts about the run
    itself -- its size, and the name and goal the operator supplied.

    It was previously drawn from the prompt material the report is composed
    from, which also carries text written by language models: every stage's
    narrative and the validator's objections to the previous draft. Numbers
    there are claims, not calculations. A draft citing an invented 0.99 drew
    the objection "the report claims 0.99 but ..."; that objection was passed
    back as revision material, its 0.99 entered the permitted set, and the
    second draft repeated the figure and passed. Any number an upstream agent
    invented in its own narrative was laundered the same way.
    """
    from state import LIVE_ONLY_EVIDENCE

    allowed: set[str] = set()
    collect_numbers({
        "dataset": state.dataset_name,
        "goal": state.user_goal or "",
        "rows": int(state.full_row_count or state.data.shape[0]),
        "rows_in_memory": int(state.data.shape[0]),
        "rows_loaded": int(state.df.shape[0]),
        "columns": int(state.data.shape[1]),
        "columns_loaded": int(state.df.shape[1]),
    }, allowed)
    for output in state.outputs.values():
        collect_numbers({k: v for k, v in output.evidence.items()
                         if k not in LIVE_ONLY_EVIDENCE}, allowed)
    return allowed


def report_text(report: dict[str, Any]) -> str:
    """The prose of a composed report that its numbers are checked in.

    Coerced rather than indexed: these are model-shaped values, and a response
    can satisfy the schema while returning takeaways as objects or sections as
    bare strings.
    """
    from agents.llm import as_text

    return " ".join([
        as_text(report.get("executive_summary")),
        as_text([s.get("body", "") if isinstance(s, dict) else s
                 for s in report.get("sections") or []]),
        as_text(report.get("key_takeaways")),
        as_text(report.get("limitations")),
    ])


def audit(state: "WorkflowState") -> dict[str, int]:
    """Defects in what a finished run delivered, with every check forced on.

    This is the measurement the ablation needs. With the layer switched off its
    own counts are zero by construction, so comparing them across the two arms
    shows nothing. What differs between the arms is what reached the reader:
    invented figures, causal wording, a buried null result, a model recommended
    although nothing beat guessing. Measuring those in the delivered output,
    identically for both arms, is the comparison.
    """
    report = state.findings("report")
    modelling = state.findings("modelling")
    ran_model = bool(modelling) and not modelling.get("skipped")
    with auditing():
        text = report_text(report) if report else ""
        return {
            "unsupported_numbers": len(unsupported_numbers(text, computed_values(state))),
            "causal_language": len(causal_language(text)),
            "suppressed_negative_result": int(bool(
                ran_model and suppressed_negative_result(
                    text, modelling.get("beats_baseline")))),
            "overridden_signal_claim": int(bool(
                ran_model and modelling.get("beats_baseline") is False
                and overridden_signal_claim(modelling.get("signal_assessment"),
                                            False,
                                            modelling.get("recommended_model")))),
        }


def summary() -> list[dict[str, str]]:
    """The registry as plain data, for documentation and tests.

    The count in the README and the architecture document is derived from this
    rather than maintained by hand, so the two cannot drift apart.
    """
    return [
        {"name": m.name, "stage": m.stage, "checks": m.checks,
         "compares_against": m.compares_against}
        for m in MECHANISMS
    ]
