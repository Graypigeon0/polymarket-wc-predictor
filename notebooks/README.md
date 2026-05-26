# Backtest & calibration notebooks

This directory holds Jupyter notebooks for the validation plan. Each phase
from `docs/BUILD_PLAN.md` gets its own notebook:

| Notebook | Purpose |
|---|---|
| `01_euro2024_backtest.ipynb` | Walk-forward Dixon-Coles backtest on Euro 2024 |
| `02_copa2024_backtest.ipynb` | Same for Copa America 2024 |
| `03_wc_history.ipynb` | Tournament-dynamics test on WC 2014/18/22 |
| `04_calibration_plots.ipynb` | Reliability diagrams, Brier scores, log loss |
| `05_top_scorer_sim.ipynb` | Top scorer model dry-run |

## Standard metrics

- **Brier score** on 1X2 (lower is better, < 0.21 target)
- **Log loss** on 1X2
- **Calibration plot**: bin predicted probabilities, compare to observed frequencies
- **Closing line value** vs. Pinnacle (the gold standard — beating CL = real edge)
- **Tournament outright**: did the realised champion fall inside the model's top-decile probability mass?

Open each notebook in Jupyter (`jupyter lab`) after running `pip install -e ".[dev]"`.
