"""Does access to credit drive economic growth? USA vs. Latin America.

Empirical implementation of plan.md. Pulls World Bank WDI series, builds a
balanced-as-possible country-year panel, estimates the finance-growth
relationship, and renders the figures in The Economist's house style.

Econometric note on what this design can and cannot deliver. The finance-growth
literature (King & Levine 1993; Levine 2005 for the survey) established a robust
partial correlation between financial depth and growth, but identifying a causal
effect requires either an instrument for credit supply (Rajan & Zingales 1998
use industry-level external-finance dependence; Levine, Loayza & Beck 2000 use
legal origin) or a credible natural experiment. We have neither. What follows is
a conditional-correlation exercise with the endogeneity stated honestly rather
than papered over: credit/GDP is jointly determined with output, and GDP sits in
the denominator of the regressor, which mechanically induces negative
co-movement in recessions. Read the coefficients as descriptive.

Every number in the output is pulled live from the World Bank API at run time —
there is no vendored dataset, so the analysis reproduces from a clean checkout.

Run:  .venv/bin/python analysis.py
Outputs: data/, figures/, results.md
"""
import os
from pathlib import Path
from typing import Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import statsmodels.formula.api as smf
from dotenv import load_dotenv

import econ_style as es

# Configuration comes from the environment, never from source. `.env` is listed
# in .gitignore; `.env.example` documents the variables without holding values.
load_dotenv()

ROOT = Path(__file__).parent
DATA, FIGS = ROOT / "data", ROOT / "figures"
DATA.mkdir(exist_ok=True)
FIGS.mkdir(exist_ok=True)

# The World Bank Indicators API is open: no key, no registration, no rate limit
# worth worrying about at this volume. WORLD_BANK_API_KEY is read anyway so that
# the credential path exists and is exercised — if this analysis were extended to
# a gated source (FRED, IMF, Quandl), the key would arrive the same way and would
# still never touch the repository.
API_BASE = os.getenv("WORLD_BANK_API_BASE", "https://api.worldbank.org/v2")
API_KEY = os.getenv("WORLD_BANK_API_KEY")  # optional; unused by this endpoint

START, END = 1990, 2023
LATAM = ["ARG", "BRA", "CHL", "COL", "MEX", "PER"]
COUNTRIES = ["USA"] + LATAM + ["LCN"]
INDICATORS = {
    "FS.AST.PRVT.GD.ZS": "credit_gdp",      # domestic credit to private sector, % of GDP
    "NY.GDP.PCAP.KD.ZG": "gdp_pc_growth",   # real GDP per capita growth, annual %
    "NY.GDP.PCAP.KD": "gdp_pc",             # real GDP per capita, constant 2015 US$
    "SP.POP.TOTL": "population",            # weights for the regional aggregate
}

# Episodes where credit/GDP is a nominal valuation artifact rather than a measure
# of real intermediation. Under hyperinflation the price level moves far faster
# than the stock of bank credit can be revalued, so the ratio swings by tens of
# percentage points a year with no corresponding change in lending. Brazil's
# 1990-94 series (ending with the 1994 Real Plan) and Argentina's pre-1991
# series (ending with the Convertibility Plan) are the two cases here.
HYPERINFLATION = {"BRA": 1995, "ARG": 1991}


# =============================================================================
# 1. Data acquisition
# =============================================================================
def indicator_url(code: str) -> str:
    """The exact endpoint hit for one indicator. Exposed so a notebook can show it."""
    return f"{API_BASE}/country/{';'.join(COUNTRIES)}/indicator/{code}"


def request_params() -> dict:
    """Query string shared by every indicator request."""
    # Pull one year before START so the t-1 lag and first difference are defined
    # at the start of the estimation window rather than lost to differencing.
    params = {"format": "json", "date": f"{START - 1}:{END}", "per_page": 1000}
    if API_KEY:
        params["api_key"] = API_KEY
    return params


def fetch_json(code: str) -> tuple:
    """Raw (metadata, observations) straight from the API, before any reshaping."""
    response = requests.get(indicator_url(code), params=request_params(), timeout=60)
    response.raise_for_status()
    payload = response.json()
    if isinstance(payload, list) and payload and "message" in (payload[0] or {}):
        raise RuntimeError(f"World Bank API error for {code}: {payload[0]['message']}")
    meta, rows = payload
    # The API paginates at per_page; silently truncating would drop country-years
    # off the end of the panel without any visible symptom, so assert instead.
    assert meta["pages"] == 1, f"{code}: response paginated, raise per_page"
    return meta, rows


def fetch(code: str) -> pd.DataFrame:
    """One World Bank indicator for all countries, as a tidy long frame."""
    _, rows = fetch_json(code)
    return pd.DataFrame(
        {"iso3": r["countryiso3code"], "year": int(r["date"]), INDICATORS[code]: r["value"]}
        for r in rows
    )


def load_raw(save: bool = True) -> pd.DataFrame:
    """Fetch all four indicators from the API and join them into one frame.

    There is deliberately no read-through cache. A cached CSV would let the
    analysis run without ever contacting the API, which quietly breaks the
    reproducibility claim: a reader cloning the repo would re-use whatever
    snapshot happened to be committed. The CSV written here is an output for
    inspection, never an input.
    """
    frames = [fetch(code).set_index(["iso3", "year"]) for code in INDICATORS]
    raw = pd.concat(frames, axis=1).reset_index().sort_values(["iso3", "year"])
    if save:
        raw.to_csv(DATA / "raw_worldbank.csv", index=False)
    return raw
