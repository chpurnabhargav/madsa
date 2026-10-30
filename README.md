# A Collaborative Multi-Agent AI Framework for Intelligent Data Science Workflow Orchestration

Specialised agents, coordinated by an adaptive orchestrator, turn a raw dataset
into a validated analytical report.

**Engineering Capstone Project — Department of CSE, KL University**
Specialisation: Data Science and Big Data Analytics · Guide: Mrs. K. Hema

| University ID | Name |
|---|---|
| 2320030242 | Challagundla Purna Bhargav Chowdary |
| 2320030205 | Narayana Manasa Lahari |
| 2320030275 | Gattineni Sasidhar |

---

## Quickstart

Three commands from a clean checkout to a validated report. No API key is
needed: the default `mock` backend runs the whole pipeline offline and free.

```bash
pip install -r requirements.txt

# Analyse a dataset. Writes Markdown, HTML and a machine-readable run record.
python run.py datasets/heart_disease.csv

# Or use the interface.
streamlit run app.py
```

The report lands in `reports/`. Open the `.html` file to read it.

### If something does not work

```bash
python doctor.py
```

It checks the interpreter, every required and optional package, Java and Spark,
API keys, file permissions, and then runs the pipeline end to end on synthetic
data. Each problem is reported with the exact command that fixes it, and it
exits non-zero if anything is actually broken, so it can gate a script.

Three outcomes: **OK** means verified working; **WARN** means an optional
capability is unavailable and the framework degrades around it; **FAIL** means
something needed is missing.

### Using a real language model

The framework runs offline by default so it can be reproduced with no account
and no cost. To use an actual model, copy `.env.example` to `.env`, add one key,
and name the backend:

```bash
python run.py datasets/heart_disease.csv --backend groq
```

`groq` and `gemini` have free tiers that are sufficient for a full run.
`anthropic` and `grok` are paid. `.env` is gitignored; keys never leave the
machine.

### What to expect

On `heart_disease.csv` (303 rows) the framework identifies the target itself,
repairs the data, derives and tests new features, finds the strongest
associations, fits and cross-validates seven models, and writes a report whose
every number is checked against a computed value. A run takes about 30 seconds
offline.

Exit code `2` means the validation stage judged the analysis unreliable — the
framework refusing to stand behind its own output is a feature, not a crash.

---

## What the framework does

A dataset goes in; a structured, validated analytical report comes out. Nine
specialised agents each own one stage of the data science lifecycle, and a
coordinator decides which of them a given dataset actually needs.

```
                       ┌──────────────────┐
                       │   Coordinator    │  plans which stages this
                       │    (planner)     │  dataset actually needs
                       └────────┬─────────┘
                                │
  ┌──────────┬─────────┬──────────┴──┬─────────────┬──────────────┐
  ▼          ▼         ▼             ▼             ▼              ▼
Profiling  Quality  Features       EDA      Multivariate   Visualization
                                                                  │
                                                             Modelling
                                                                  │
                                        ┌─────────────────────────┘
                                        ▼
                                     Report ──────► Validation
                                        ▲                │
                                        └── revise ──────┘
                                          (bounded retries)
                                                         │
                                                         ▼
                                                  Final report
```

Each agent follows the same three-step contract:

```
compute with deterministic tools  →  interpret with a language model
                                  →  verify the interpretation against the data
```

The third step is the point. A language model asked to analyse data will
occasionally reference a column that does not exist, describe a negligible effect
as important, state a number nothing computed, or present a null result as a
success. Every one of those is detected in code before it reaches the report.

## Why the orchestration is the contribution

Multi-agent LLM systems are not new. What this project contributes is the
orchestration layer:

- **Adaptive planning.** The coordinator inspects a cheap structural profile
  before any analysis and selects the stages the dataset needs. A clean dataset
  skips cleaning; an unsupervised one skips modelling. Plans that could not
  execute are repaired against declared dependencies rather than rejected.
- **Conditional execution.** Some stages only become unnecessary once an earlier
  stage has reported, so that decision is made during the run.
- **Validated revision.** The validation stage can send the report back to be
  re-composed with its objections in hand, rather than delivering it with a
  warning attached.
- **Verification at every agent boundary.** Listed in full below.

## Verification mechanisms

There are **13 verification mechanisms**, and that number is not maintained by
hand. Each is registered in `verification.py`, so the count is derived from the
code and a test fails if this document drifts from it:

```python
>>> import verification
>>> len(verification.MECHANISMS)
13
```

Every one follows the same rule: **compare what the model said against what a
tool computed.** None asks a model whether its own output is correct, because a
confidently wrong claim reviews itself as correct — which is precisely the
weakness in the self-reflection approaches the literature reports.

| Mechanism | What it catches | Where |
|---|---|---|
| Schema checking | A target or feature the model invented | `agents/profiling_agent.py` |
| Action allowlist and guards | Dropping the target, imputing a 77%-missing column, imputing text | `agents/quality_agent.py` |
| Effect-size checking | A negligible effect claimed as important | `agents/eda_agent.py` |
| Causal-language detection | "causes" applied to observational association | `agents/eda_agent.py`, `report_agent.py` |
| Chart specification validation | A chart on a non-existent or wrongly typed column | `tools/visualization.py` |
| Permutation testing | A model that has learned nothing presented as predictive | `tools/modelling.py` |
| Silhouette gating | Weak clustering presented as customer segments | `agents/multivariate_agent.py` |
| Signal override | A null result described as moderate or strong, or a model still recommended despite it | `agents/modelling_agent.py` |
| Importance checking | A feature called a driver that permutation importance cannot tell apart from noise | `agents/modelling_agent.py` |
| Number provenance | A figure in the report that nothing computed | `agents/report_agent.py` |
| Mechanical review | Failed stages, dropped findings, omitted limitations | `agents/validation_agent.py` |

Deterministic checks are authoritative: a model review that disagrees with a
measurement does not overturn it.

## Installation

```bash
pip install -r requirements.txt
```

The framework runs with no API key on the `mock` backend, which derives its
decisions from the statistics the tools compute. That is what the test suite and
offline evaluation use.

For a real analysis, add a key and choose a backend:

```bash
cp .env.example .env        # then add GROQ_API_KEY or GEMINI_API_KEY
```

**If you have keys for more than one provider, use `--backend auto`.** Free
tiers rate-limit hard, and the framework's nine agents exhaust a per-minute
allowance almost at once. The retry logic then does the only sensible thing and
waits the interval the provider names -- which on a measured evaluation sweep
produced stage durations of 62, 150 and 450 seconds on stages whose computation
takes under a second. Almost the entire wall-clock was spent asleep.

Two free tiers from different vendors have separate allowances, so `auto` tries
each in turn and moves on when one reports its limit, setting that provider
aside for a few minutes rather than waiting for it. A second free key roughly
halves the time a full sweep takes and costs nothing.

```bash
python run.py datasets/titanic.csv --backend auto
```

Or run an open-weight model locally with [Ollama](https://ollama.com), which
needs no key but does need the VRAM:

```bash
ollama pull qwen2.5:7b && ollama serve
```

## Usage

**Command line**

```bash
python run.py datasets/titanic.csv
python run.py data.csv --goal "Which customers are likely to churn?"
python run.py data.csv --backend anthropic
python run.py data.csv --fixed-pipeline      # disable adaptive planning
python run.py data.csv --quick               # demo speed: skip the stacked ensemble and tuning
```

Reports are written to `reports/` as Markdown, self-contained HTML, and a JSON
run record. The exit code is `2` when validation judges the analysis unreliable,
so the framework can be used in a pipeline.

**Using the model it recommends**

A report that says "a Random Forest predicts this reliably" is not much use if
the Random Forest no longer exists. The winning model is refitted on every row
and saved to `models/`, with a signature recording the columns it expects, their
types, the target, and the classes it can emit.

```bash
python predict.py --latest new_rows.csv -o scored.csv
python predict.py models/diabetes_20260920_191154.joblib new_rows.csv
```

Preprocessing is inside the saved pipeline, so it takes raw rows in the original
shape — no imputation or encoding to reproduce. That includes everything the
analysis did before modelling: numbers stored as text are converted, category
case is normalised, outliers are clipped to the fences fitted during the run, and
derived features such as a missingness flag or a logarithm are rebuilt from the
columns they came from, with the parameters fitted at the time. The input is
checked against the signature before scoring, so a mismatch is reported as a
named missing column rather than an opaque error about feature counts. Class
probabilities are included where the model provides them.

Only a model that beat the baseline is saved. A baseline winning, or a winner
that fails to clear it, means nothing learned anything usable — a finding
rather than a recommendation, and not something `--latest` should hand out.

**Web interface**

```bash
streamlit run app.py
```

The interface shows the workflow, not only its result. A rail of the nine
agents, drawn in the order they run, explains the system before a run, advances
stage by stage during one, and afterwards records what each agent did, how long
it took and what the checks it carries caught. The Verification tab lists every
registered check — what it compares against, and what it caught in this run —
beside the reviewer's objections. The report is typeset as a paper from the same
code as the file exports, so the two cannot drift, and results computed on one
dataset are never shown under another's name.

**What a report contains**

An at-a-glance card stating the question, the answer and how far to trust it;
the model's interpretation of each stage, each followed by the computed evidence
behind it; what the chosen model does in practice, as out-of-sample counts
rather than a score alone ("of 268 patients with diabetes, it identifies 153");
a full model comparison with fold standard deviations; a ranking of which
features the model actually relies on; charts with a note on what to look for in
each; the validator's unresolved objections; and a glossary defining only the
terms that particular report used.

**Evaluation**

```bash
python -m evaluation.harness
python -m evaluation.harness --case titanic --case pure_noise
python -m evaluation.harness --backend auto --resume runs/20260916_111243
```

A full sweep on a real provider runs for hours, which is longer than most
sessions survive. Each dataset's metrics are written the moment it finishes,
and `--resume` continues in an existing run directory, reusing what is already
there. Without it every interruption restarted at the first dataset, so the
sweep redid the same early cases on each attempt and never reached the later
ones no matter how many times it was launched.

`--no-verification` runs the ablation: every registered check is switched off.
Each run records two figures. `corrections_applied` is how often each mechanism
fired, counted by the mechanism itself; under the ablation it is zero by
construction. `delivered_defects` is measured on the finished report with every
check forced back on — invented figures, causal wording, a buried null result, a
model recommended despite no signal — and is the figure the two arms are
compared on.

**Benchmarks**

```bash
python datasets/fetch_benchmark_data.py      # downloads the larger datasets once
python -m evaluation.benchmark_comparison    # head-to-head with published agents
python -m evaluation.benchmark               # the full workflow at scale
```

Both resume where they stopped and write to `benchmarks/`, which is committed.
See [Benchmarks against published systems](#benchmarks-against-published-systems).

**Tests**

```bash
python -m pytest
```

## How the evaluation works

The project is not a comparative study, so the evaluation measures how well the
framework performs across datasets of differing character rather than whether one
architecture beats another. Twelve datasets are used. Five are real-world data drawn from five
domains: Titanic passenger records (transport), wine chemistry (food science),
Cleveland heart disease and Pima diabetes (healthcare), and US census income
(economics). Seven are synthetic, each constructed to test a behaviour real data
cannot be relied upon to contain: injected faults with known ground truth, a
clean dataset with no relationship, pure noise, a 3% imbalanced target, a
regression target, 15 rows, and 45 columns of which three carry signal.

Five dimensions are reported:

- **Data-quality detection**, scored against injected faults with known ground
  truth rather than judged.
- **Reliability**, measured by how often the framework's own verification caught
  an unsupported claim.
- **Robustness**, measured by whether stress datasets complete and produce
  correct negative results.
- **Adaptivity**, measured by whether the coordinator's plan actually varies
  with the dataset.
- **Cost**, in model calls, tokens and time per stage.

## Benchmarks against published systems

Two surveyed papers report per-dataset results on datasets used here, so the
framework was re-run under each paper's own protocol (figures checked against
the papers' tables). Data Interpreter (Hong et al., ACL Findings 2025, Table 6)
— 80/20 hold-out, ours the mean of five random splits:

| Dataset | AutoGen | OpenInterpreter | TaskWeaver | XAgent | OpenDevin | Data Interpreter | **Ours** (mean ± sd, 5 splits) |
|---|---|---|---|---|---|---|---|
| Wine recognition | 1.00 | 1.00 | 1.00 | 1.00 | 0.96 | 0.96 | **0.994 ± 0.012** |
| Breast cancer Wisconsin | 0.97 | 0.96 | 0.96 | 0.94 | 0.96 | 0.99 | **0.977 ± 0.010** |
| Titanic | 0.82 | 0.81 | 0.43 | 0.00 | 0.81 | 0.82 | **0.847 ± 0.029** |
| House prices (1/(1+RMSLE)) | 0.88 | 0.87 | 0.49 | 0.00 | 0.87 | 0.91 | **0.880 ± 0.007** |
| **Average of the four** | 0.917 | 0.910 | 0.720 | 0.485 | 0.900 | 0.920 | **0.925** |

LAMBDA (Sun et al., 2025, Table 2) — 5-fold cross-validation:

| Dataset | LAMBDA (best of 9 models) | **Ours** (model the framework selected) |
|---|---|---|
| Breast cancer Wisconsin | 0.9807 | **0.9824 ± 0.0111** (stacked ensemble) |
| Wine recognition | 0.9889 | **0.9887 ± 0.0138** (extra trees) — a tie: about 2 of 178 wrong in each |

Behind Data Interpreter on Breast Cancer and House Prices, ahead on Titanic and
on the four-dataset average. Full tables, per-split values and what is not
controlled: [`benchmarks/head_to_head.md`](benchmarks/head_to_head.md).

## Project layout

```
madsa/
├── config.py              all tunable settings
├── state.py               the shared workflow state and agent contract
├── verification.py        the 13 verification mechanisms, in one registry
├── run.py                 command-line entry point
├── predict.py             score new rows with a model a run produced
├── app.py                 Streamlit interface
├── demo.py                signal-versus-noise demonstration in one command
├── tools/                 deterministic computation, no AI
│   ├── profiling.py       structure, types, target candidates
│   ├── quality.py         fault detection and cleaning
│   ├── features.py        derived features and a measurement of their worth
│   ├── eda.py             correlations, group differences, distributions
│   ├── visualization.py   chart generation
│   ├── modelling.py       cross-validation, selection, tuning, permutation tests
│   ├── multivariate.py    PCA, k-means, VIF, mutual information
│   └── spark_engine.py    Spark execution for profiling and correlation
│                          (see "Scale" below for what is not distributed)
├── agents/                nine specialised agents
│   ├── llm.py             the only module that talks to a language model
│   ├── base.py            the contract every agent implements
│   └── *_agent.py         one file per agent
├── orchestrator/
│   ├── graph.py           planning, the LangGraph state machine, revision
│   └── export.py          Markdown, HTML and JSON output
├── evaluation/
│   ├── harness.py         fault injection, metrics, evaluation report
│   ├── benchmark_comparison.py  head-to-head with published agents
│   ├── benchmark.py       the full workflow on the large datasets
│   └── agent_matrix.py    what each agent yielded, dataset by dataset
├── benchmarks/            measured benchmark results (committed)
├── models/                fitted pipelines, each with a signature
├── reports/               generated reports
├── runs/                  full run records and evaluation sweeps
└── tests/
    ├── conftest.py        cheap defaults so the suite stays fast enough to run
    ├── test_framework.py  framework behaviour, on the mock backend
    └── test_hardening.py  edge cases and failure paths
```

## Scale, and its limits

The framework chooses its execution engine from the size of the file before
reading it: pandas below a threshold, Apache Spark above. On a 3,000,000-row,
129 MB dataset the distributed path used **55 MB** of driver memory where pandas
needed **641 MB**, and profiled every row where pandas had to fall back to a
100,000-row sample carrying up to **1.0%** sampling error.

**What runs on Spark:** profiling and the correlation matrix, computed exactly
across every row, and a **full-data validation of the selected model**: its
family is retrained with Spark MLlib on a random 80% of *all* rows and scored on
the other 20% (Covertype: 465,120 training rows, 43 s on a laptop). Quality
assessment, feature engineering, multivariate analysis and the delivered model
run in memory on a bounded sample.

That boundary is deliberate and worth stating rather than glossing. Profiling
and correlation are aggregations, which Spark computes exactly across every row
— verified identical to pandas on all fifteen columns of a 32,561-row dataset,
and to six decimal places on both Pearson and Spearman across seventeen
correlation pairs. Model fitting is not an aggregation. The delivered model and
its reported scores stay scikit-learn's, so a result never depends on which
engine ran — the property that makes automatic engine selection safe. The
MLlib run confirms at full scale rather than replacing: MLlib has a narrower set
of algorithms (no extra-trees, stacking or RBF SVM, which are represented by a
random forest), and its trees are depth-limited to fit one machine's memory, so
its score is a lower bound for the family.

The report says which figures came from the full dataset and which from the
sample, because that changes how each should be read.

## Design decisions

**Agents call fixed tools; they do not write code.** The intelligence lives in
orchestration and interpretation, not in code generation. Every number in a
report is traceable to a function rather than to a model, which is what makes
the verification layer possible at all.

**Agents never call each other.** All communication passes through
`WorkflowState`, so the orchestrator controls information flow completely and an
agent can be added, removed or reordered without touching any other agent.

**Every stage degrades rather than fails.** An agent that cannot reach a model
falls back to its computed results and reports itself as degraded. The framework
produces an analysis with no language model at all, which also makes the
verification layer testable in isolation.

**The backend is swappable in one setting.** `mock`, `groq`, `gemini`,
`anthropic`, `grok` and `ollama` share one interface, so results can be
reproduced without an API key. `auto` is that same interface wrapping several
providers at once, which is how two free tiers become one usable budget.
