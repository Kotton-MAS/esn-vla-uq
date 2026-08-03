# esn-vla-uq

Uncertainty quantification for VLA (vision-language-action) policies via Echo State Networks.

> **Status: Sprint 1 implementation complete (pre-release).**
> The ESN core (reservoir, ridge read-out), reservoir diagnostics (spectral
> radius, ESP, memory capacity), and synthetic sample data generation are
> implemented and covered by tests (`make ci` is green). The `diagnose` and
> `gen-sample-data` subcommands are wired up and runnable — see Quickstart
> below. No version has been released yet.

> **Data disclaimer.** Every number produced by this repository at v0.1 comes from the
> bundled *synthetic* rollout data (`source: "synthetic"`). None of it is a real LIBERO
> evaluation result.

## What this is

A closed-form, ensemble-free uncertainty estimator for VLA policies:

- Reads policy rollout logs (action chunks + proprioceptive state) through an adapter
  boundary, so no VLA runtime (e.g. openpi) is a dependency of this package.
- Feeds them into a self-implemented Echo State Network with a ridge read-out.
- Ships reservoir diagnostics (spectral radius, echo state property, memory capacity)
  as first-class, reportable outputs.

## Requirements

- Python 3.12+
- [uv](https://docs.astral.sh/uv/) for dependency management
- Runtime dependency: numpy only

## Quickstart

```bash
uv sync

# CLI entry point
uv run esn-vla-uq --help
uv run esn-vla-uq --version
```

Both subcommands are implemented and runnable. `diagnose` builds a reservoir
from `ESNConfig` and reports spectral radius, ESP, and memory capacity as JSON
(the numbers below are illustrative, from the default `--seed 0`
configuration on this synthetic-data setup — not a real LIBERO evaluation,
see the data disclaimer above):

```bash
# Reservoir diagnostics (spectral radius / ESP / memory capacity)
uv run esn-vla-uq diagnose --output-dir outputs/
```

```text
diagnostics: schema_version=0.1.0 data_source=synthetic seed=0 n_reservoir=200 n_inputs=1 ...
spectral: spectral_radius=0.900000 effective_spectral_radius=0.900000
esp: verdict=esp_holds sufficient(sigma_max<1)=False[1.784905] necessary(rho<1)=True[0.900000] ...
memory_capacity: total_mc=14.8294 mc_per_neuron=0.0741 memory_horizon=20 n_delays=200 ...
saved diagnostics report: path=outputs/diagnostics/<timestamp>.json
```

```bash
# Regenerate the synthetic sample data
uv run esn-vla-uq gen-sample-data --seed 0 --output outputs/sample.npz
```

```text
saved rollout dataset: source=synthetic n_episodes=40 total_steps=7690 path=outputs/sample.npz
```

## Development

Validation is centralised in the `Makefile`; local runs, the Claude Code stop hook, and
GitHub Actions all invoke the same target.

```bash
make ci      # uv lock --check + ruff check + ruff format --check + mypy + pytest
make test    # pytest only
make fmt     # ruff format (modifies files)
```

## Repository layout

```
.
├── src/esn_vla_uq/       # package (src layout)
│   ├── cli/              # argparse CLI entry point
│   ├── esn/              # ESN core: reservoir, ridge read-out, model
│   ├── diagnostics/      # spectral radius / ESP / memory capacity
│   ├── data/             # rollout schema, synthetic generation, IO
│   └── assets/samples/   # bundled synthetic sample data
├── tests/                # pytest suite
├── docs/                 # requirements, design notes, plans, ADRs
├── Makefile              # single source of truth for validation
└── pyproject.toml        # project metadata and dependencies (uv + hatchling)
```

## Documentation

- `docs/要件_Phase0リポジトリ化_v0.1.md` — requirements (Japanese)
- `docs/plans/` — approved implementation specs
- `docs/design.md` — design document (added in Sprint 1, T2)

A fuller English README and a Japanese `README.ja.md` are planned for Sprint 3.

## License

Apache-2.0. See [LICENSE](LICENSE). To cite this work, see [CITATION.cff](CITATION.cff).
