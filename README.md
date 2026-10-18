# PHT experiment code and complete proofs

Companion to **Exact optimization of data product selection and uniform pricing under access constraints**.

This release contains the experiment implementation and complete proofs. It includes the configuration files, seeds, dependencies and instructions needed to run the code. It does not contain the paper, source measurements, generated instances, archived experiment results or exported figures.

## Contents

| Path | Contents |
| --- | --- |
| `proofs/complete_proofs.pdf` | Complete proofs of the nine lemmas, theorems, proposition and corollaries |
| `proofs/*.tex` | Proof source, notation and theorem environments |
| `proofs/*correspondence*` | Mapping of proof statements and equations to the manuscript |
| `experiments/src/` | PHT-DP, cardinality and fixed-set extensions, G0–G2, MILP, input builders, runner and result analysis |
| `experiments/tests/` | Solver, interface, input-construction and statistical checks |
| `experiments/protocol/` | Frozen configurations, seeds, population matrices, resource budgets and source checksums |
| `experiments/source/` | UCI measurement-construction code and deterministic cost configuration |
| `scripts/` | Experimental figure and table generation, including the predetermined menu example |
| `reproduce.py` | Unified entry point |

The original solver and benchmark modules are unchanged. The entry point has been adapted for a release without archived inputs or results. Generated files are written to `.work/` and `generated/`, both excluded by `.gitignore`.

## Environment

Use Linux or WSL2 for the full benchmark runner, which uses process resource limits and CPU affinity. The recorded Python version is 3.12.14. Install the pinned dependencies in a virtual environment.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python reproduce.py tests
```

The default test command runs without downloading the source dataset. For proof compilation, install a TeX distribution providing `latexmk`, pdfLaTeX, `amsmath`, `amssymb`, `xurl` and `hyperref`. The proof PDF is included for reading without TeX.

## Data and input construction

The source is UCI ElectricityLoadDiagrams20112014, contributed by Artur Trindade. The official dataset page is <https://archive.ics.uci.edu/dataset/321/electricityloaddiagrams20112014>, DOI <https://doi.org/10.24432/C58C86>. The source is provided by UCI under CC BY 4.0. It is downloaded from UCI rather than distributed in this repository. `data_protocol.json` specifies the source URL and expected checksum.

```bash
python reproduce.py raw-data
python reproduce.py inputs
python reproduce.py tests --with-data
```

If the official archive is already available locally, replace the first command with:

```bash
python reproduce.py raw-data --archive /absolute/path/to/electricityloaddiagrams20112014.zip
```

The source archive is approximately 249 MiB. Quality measurements are constructed from the prescribed evaluation months, reference months and delays. Input construction generates the 20,232-scenario pool, the 864-instance comparison subset, the 510 synthetic instances, and the prescribed mapping and budget analyses. All selections and seeds come from the configuration files.

`formal_protocol.json` and `build_formal.py` are retained because the final builder and input tests use this configuration lineage. The commands below execute `final_protocol.json`; they do not execute an earlier benchmark batch.

## Run and analyze the prescribed experiments

```bash
python reproduce.py run --mode bulk --workers 4
python reproduce.py run --mode isolated --workers 4
python reproduce.py summarize
python reproduce.py figures
```

Each method runs with one thread. `--workers` controls the number of concurrent tasks and accepts 1–4. Resource budgets, repetition counts and the full job list are fixed by the protocol. The runner retains terminal states and valid menus returned at resource limits. It can resume an interrupted invocation with the same command and CPU configuration.

The summary uses the specified buyer choice rule to evaluate returned menus, including the MILP baseline. It preserves missing-menu and timeout outcomes. Optimizer completion and realized net profit are evaluated separately.

Results are stored under `.work/experiments/results/`. Summaries, numerical tables and plots are stored under `generated/`. To recompute a summary after additional jobs complete, move or remove `generated/summary` and invoke `summarize` again.

## Experimental figures and tables

| Manuscript item | Generation code | Local output |
| --- | --- | --- |
| Table I, instance parameters | `scripts/export_paper_tables.py` | `generated/tables/table_I_parameters.csv` and panel TeX |
| Table II, solution quality and completion | `scripts/export_paper_tables.py` | `generated/tables/table_II_results.csv` and panel TeX |
| Optimization time, Figure 3 (formerly 6-1) | `plot_publication.efficiency` | `generated/figures/fig6_1.pdf` |
| Scalability, Figure 4 (formerly 6-2) | `plot_publication.scaling` | `generated/figures/fig6_2.pdf` |
| Product cardinality frontier, Figure 5 (formerly 6-3) | `plot_publication.capacity` | `generated/figures/fig6_3.pdf` |
| Menu decisions, Figure 6 (formerly 6-4) | `extract_menu_example.py` and `plot_publication.menu` | `generated/figures/fig6_4.pdf` |
| Additional prescribed analyses | `experiments/src/summarize_final.py` and `plot_final.py` | `generated/summary/` and `generated/analysis_figures/` |

The experimental figures are computed from locally generated run records. Archived measurements and results from the paper are not included, so this release does not provide an immediate redraw of the original recorded timing figures. Runtime and budget-limited results can vary with hardware and execution environment. The paper's conceptual overview and tree illustrations are outside this experiment-code release.

The plotting source uses the manuscript palette and sizes. Matplotlib selects an installed serif font, falling back to its bundled STIX or DejaVu fonts. No font binaries or generated artwork are distributed.

## Complete proofs

Open `proofs/complete_proofs.pdf` or compile the source:

```bash
python reproduce.py proofs
```

The compiled output is `generated/proofs/complete_proofs.pdf`. The proof document covers Lemma 4.1, Theorem 4.1, Proposition 4.1, Theorem 4.2, Lemma 5.1, Theorems 5.1–5.2 and Corollaries 5.1–5.2. Manuscript equation numbers are preserved; S1 and S2 denote proof-only identities.

## Upload scope

Publish the files in this source package. `.work/`, `generated/`, virtual environments and caches are local working files and are excluded from Git. Keep the source configurations, `.gitattributes` and `.gitignore` so the original checksums and upload scope are preserved.
