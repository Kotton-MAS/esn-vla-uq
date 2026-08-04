# esn-vla-uq

Closed-form, ensemble-free **calibrated prediction intervals** for VLA
(vision-language-action) policies, using an Echo State Network with a ridge read-out and
split conformal prediction.

The coverage guarantee is validated on real `openpi` rollouts. **Failure detection is
not a claim of this release** — see [Scope](#scope).

[日本語版 README](README.ja.md)

> **Every number in this repository comes from bundled *synthetic* rollout data**
> (`source: "synthetic"`). None of it is a real LIBERO evaluation result. The synthetic
> generator is deliberately tuned so that a naive baseline cannot separate success from
> failure perfectly — see [`docs/design.md`](docs/design.md) §7.

## Quickstart

```bash
uv sync
uv run esn-vla-uq calibrate     # coverage / ECE / failure-detection AUROC
uv run esn-vla-uq calibrate --input <openpi-log-dir>   # collected openpi rollouts
uv run esn-vla-uq diagnose      # spectral radius / ESP / memory capacity
```

## Demo

![Uncertainty rises after failure onset](docs/assets/uncertainty_demo.gif)

```bash
uv sync --extra viz
uv run esn-vla-uq demo --output outputs/demo.gif
```

The lower panel is the conformal prediction interval half-width — the per-step
uncertainty score. On the episode shown it is 1.13x higher after failure onset than
before.

**This is synthetic data, and the rise reflects how the generator was built.** The same
relationship does not hold on real openpi rollouts (see [Scope](#scope)). The demo shows
what the output looks like, not that the score predicts failure.

The rise is modest **by design**: the width is bounded to at most a 2x spread so that
it cannot be distorted by the observable's dynamic range (see below). The *ordering* of
the uncertainty score — which is what failure detection uses — is unaffected by that
bound.

> **The uncertainty reacts to the failure; it does not anticipate it.** The rise happens
> **15 steps after** onset, not before. The signal that drives it (action-chunk
> dispersion) is only refreshed at inference steps, which occur every 16 steps, so the
> lag is bounded by the chunk period. Before the failure condition starts, the chunk
> carries no evidence of it. The `demo` command prints the measured
> `detection_lag_steps` on every run.

## How it works

A rollout gives proprioceptive state, the executed action, and the policy's action
chunk. The ESN maps that history to a reservoir state; a ridge read-out predicts the
next action; split conformal turns the residuals into prediction intervals with a
finite-sample coverage guarantee. The interval half-width is the uncertainty score.

The reservoir is fixed and random — only the linear read-out is fitted, in closed form.
There is no ensemble and no gradient training, which is what makes the method a
candidate for porting onto a physical reservoir later.

## Results on the bundled synthetic data

Nominal coverage 90%, averaged over 20 calibration/test splits:

| score        | interval width  | coverage      | mean half-width |
| ------------ | --------------- | ------------- | --------------- |
| `absolute`   | constant        | 0.903 ± 0.027 | 0.0525          |
| `normalized` | varies per step | 0.903 ± 0.026 | **0.0486**      |

`normalized` matches `absolute`'s coverage with a narrower average interval, so it is
the default.

## Results on real openpi rollouts

100 episodes from `libero_spatial` (10 trials per task), collected with
`scripts/collect_openpi_rollouts.py` against a live `pi0_libero` policy server:

| split         | coverage            | ECE    | mean half-width |
| ------------- | ------------------- | ------ | --------------- |
| `within_task` | **0.9033 ± 0.0102** | 0.0029 | 0.250           |
| `across_task` | 0.8977 ± 0.0397     | 0.0020 | 0.297           |

**Coverage holds on real data across four collections.** With 10 episodes it was
0.881 ± 0.049; with 100 the spread shrank about 5x — exactly what the "effective sample
size is the number of episodes" argument predicts.

**`across_task` breaks down on long-horizon tasks**, which is what the exchangeability
argument predicts: calibration and test come from different task distributions, so the
guarantee does not transfer. On `libero_10` it drops to 0.779 with 17x the ECE. This is
why `within_task` is the default.

**Collection is not reproducible.** pi0 samples its actions (flow matching), so the same
`--seed` gives different rollouts. The `--seed` only fixes the LIBERO initial states.
Analysis of an already-collected log *is* reproducible.

## Scope

**What this release establishes:** prediction intervals with a finite-sample coverage
guarantee, validated on real openpi rollouts, plus reservoir diagnostics.

**What it does not:** that the uncertainty score detects failures. `calibrate` still
reports a failure-detection AUROC, but it is an exploratory diagnostic. On real openpi
rollouts it sits at chance (0.457–0.477); the high value on synthetic data (0.87) comes
from the generator having been built with that relationship in it. Alternative
observables were evaluated and none survived a within-task check — with 23 failures
spread over 8 tasks there are only 1–3 failures per task, and per-task AUROC swings
between 0.000 and 1.000.

Revisiting this needs 5–10 failures per task and a setting where failures other than
timeout occur; LIBERO has no early-termination condition, so every observed failure was
the policy running out of steps. See `docs/design.md` §10.14.

## Reservoir diagnostics

The diagnostics are a first-class output, not an afterthought — they are what makes the
reservoir's behaviour auditable before you trust any interval.

```bash
uv run esn-vla-uq diagnose --output-dir outputs/
```

```text
spectral: spectral_radius=0.900000 effective_spectral_radius=0.900000
esp: verdict=esp_holds sufficient(sigma_max<1)=False[1.784905] necessary(rho<1)=True[0.900000] ...
memory_capacity: total_mc=14.8294 mc_per_neuron=0.0741 memory_horizon=20 n_delays=200 ...
```

The echo state property is reported as **three indicators plus a verdict**, never a
single number: a sufficient condition (sigma_max < 1), a necessary one (rho < 1), and an
empirical convergence test. The default configuration satisfies the necessary but not
the sufficient condition, which is normal and why all three are printed.

## Requirements

- Python 3.12+ and [uv](https://docs.astral.sh/uv/)
- Runtime dependency: **numpy only**
- `esn-vla-uq[viz]` adds matplotlib for the reliability diagram and the demo GIF. All
  numbers are computed without it.

## Not implemented yet

- **openpi rollouts have not actually been collected.** `OpenpiLogSource` and the
  collection script exist and were written against openpi's real implementation, but
  no run against a live policy server has been done — that needs a GPU and a LIBERO
  setup. Tests use fixtures shaped like real openpi output (`chunk_horizon=50`,
  replan every 5 steps).
- Real LIBERO footage in the demo (the video panel is a synthetic stand-in).
- VLM feature injection (deferred to v0.2 by the requirements).

## Development

Validation is centralised in the `Makefile`; local runs, the stop hook, and GitHub
Actions all invoke the same target.

```bash
make ci      # lock + gitignore guard + version + secrets + audit + lint + format + type + test
make test    # pytest only
make fmt     # ruff format (modifies files)
```

## Repository layout

```
src/esn_vla_uq/
├── linalg.py       # shared spectral quantities (lowest layer)
├── provenance.py   # DataSource (lowest layer)
├── esn/            # reservoir, ridge read-out, model
├── diagnostics/    # spectral radius / ESP / memory capacity
├── data/           # schema, invariants, sources/, synthetic generation, IO, features
├── uncertainty/    # prediction task, splits, nonconformity, split conformal
├── calibration/    # coverage / ECE / reliability diagram
├── demo/           # demo animation (frame data and rendering are separated)
└── cli/            # argparse entry point
```

## Documentation

- [`docs/design.md`](docs/design.md) — design document, including every measurement that
  changed a decision
- [`docs/plans/`](docs/plans/) — approved implementation specs per sprint
- [`docs/next-pr-candidates.md`](docs/next-pr-candidates.md) — known follow-up items
- [`CHANGELOG.md`](CHANGELOG.md)

## License

Apache-2.0. See [LICENSE](LICENSE). To cite this work, see [CITATION.cff](CITATION.cff).
