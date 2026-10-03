# Does access to credit drive economic growth? USA vs. Latin America

Does private credit expansion raise growth, and does the relationship differ
between a mature financial system (the United States) and less financially deep
economies (Argentina, Brazil, Chile, Colombia, Mexico, Peru)?

**Short answer: the data cannot support the hypothesis.** Across seven
specifications the estimated growth effect of credit expansion is
indistinguishable from zero in Latin America. The only coefficient that reaches
significance points the other way — it is the *US* where credit expansion tracks
growth — and it does not survive dropping the 2020-22 pandemic years. Full
discussion in [`results.md`](results.md).

## Contents

| File | What it is |
|---|---|
| [`plan.md`](plan.md) | The research design, written before the data was pulled |
| [`credit_growth.ipynb`](credit_growth.ipynb) | The analysis end to end: API pull, cleaning, charts, conclusion |
| [`analysis.py`](analysis.py) | The pipeline as importable functions — the notebook calls these |
| [`econ_style.py`](econ_style.py) | Chart styling in the house style of *The Economist* |
| [`results.md`](results.md) | Generated regression tables and limitations |
| `figures/` | Generated charts (PNG) |
| `data/` | Generated CSVs — outputs of the API pull, never inputs |

## Data

All figures come from the [World Bank Indicators API
v2](https://datahelpdesk.worldbank.org/knowledgebase/articles/889392), pulled at
run time. Nothing is vendored: delete `data/` and everything regenerates.

| Series | Indicator |
|---|---|
| `FS.AST.PRVT.GD.ZS` | Domestic credit to private sector (% of GDP) |
| `NY.GDP.PCAP.KD.ZG` | Real GDP per capita growth (annual %) |
| `NY.GDP.PCAP.KD` | Real GDP per capita (constant 2015 US$) |
| `SP.POP.TOTL` | Total population (weights for the regional aggregate) |

## Running it

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
cp .env.example .env          # optional: the World Bank API needs no key

.venv/bin/jupyter notebook credit_growth.ipynb   # the notebook, top to bottom
.venv/bin/python analysis.py                     # or the same pipeline as a script
```

## Credentials

The World Bank Indicators API is open, so this project runs with no `.env` file
at all. Configuration is still read from the environment via `python-dotenv`
rather than hardcoded, `.env` is listed in `.gitignore`, and `.env.example`
documents the variables without holding any values.
