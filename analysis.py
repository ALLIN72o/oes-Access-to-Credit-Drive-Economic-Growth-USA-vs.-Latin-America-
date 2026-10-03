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


# =============================================================================
# 2. Panel construction and transformations
# =============================================================================
def build_panel(raw: pd.DataFrame) -> pd.DataFrame:
    """Country-year panel with the regressors specified in plan.md §3."""
    df = raw.sort_values(["iso3", "year"]).copy()
    cols = list(INDICATORS.values())

    # Carry forward short reporting gaps only. A two-year cap is a judgement
    # call: long enough to bridge a missed WDI vintage, short enough that we are
    # not inventing a decade of synthetic variation. Interpolation would be the
    # alternative, but it leaks future information into the lag structure.
    df[cols] = df.groupby("iso3")[cols].ffill(limit=2)

    g = df.groupby("iso3")
    df["d_credit"] = g["credit_gdp"].diff()       # ΔCredit/GDP_t, the flow measure
    df["credit_lag"] = g["credit_gdp"].shift(1)   # Credit/GDP_{t-1}, the stock measure
    # Predetermining the stock does not buy exogeneity — growth is serially
    # correlated, so last year's depth is still correlated with this year's
    # shock — but it removes the purely mechanical same-period simultaneity.
    df["latam"] = (df["iso3"] != "USA").astype(int)
    df["hyper"] = [
        iso in HYPERINFLATION and yr < HYPERINFLATION[iso]
        for iso, yr in zip(df["iso3"], df["year"])
    ]

    df = df[df["year"].between(START, END)]
    return df.dropna(subset=["credit_gdp", "gdp_pc_growth"])


def country_panel(panel: pd.DataFrame) -> pd.DataFrame:
    """Estimation sample: the seven individual economies.

    The LCN regional aggregate is excluded. It is a weighted composite of the
    same Latin American economies, so keeping it would double-count them and
    understate the standard errors.
    """
    return panel[panel["iso3"] != "LCN"].copy()


def regional_series(panel: pd.DataFrame) -> pd.DataFrame:
    """USA vs. population-weighted LatAm-6 aggregate, for the time-series chart.

    Population weights (plan.md §3) stop Chile and Peru from counting as much as
    Brazil and Mexico in the regional average. Note the resulting series is a
    weighted mean of ratios, not the ratio of regional totals — the two differ,
    and the latter would be preferable if the WDI published regional credit
    stocks in levels.
    """
    latam = panel[panel["iso3"].isin(LATAM)]
    rows = []
    for year, grp in latam.groupby("year"):
        w = grp["population"]
        rows.append({
            "year": year,
            "credit_gdp": np.average(grp["credit_gdp"], weights=w),
            "gdp_pc_growth": np.average(grp["gdp_pc_growth"], weights=w),
            "n_countries": len(grp),
            "region": "LatAm",
        })
    agg = pd.DataFrame(rows)

    usa = (panel[panel["iso3"] == "USA"][["year", "credit_gdp", "gdp_pc_growth"]]
           .assign(n_countries=1, region="USA"))

    reg = pd.concat([usa, agg], ignore_index=True).sort_values(["region", "year"])
    # Five-year rolling means strip the business cycle so the chart shows the
    # structural trend rather than the 2009 and 2020 troughs.
    for col in ("credit_gdp", "gdp_pc_growth"):
        reg[f"{col}_roll5"] = reg.groupby("region")[col].transform(
            lambda s: s.rolling(5, min_periods=3).mean()
        )
    return reg


def quinquennial(panel: pd.DataFrame) -> pd.DataFrame:
    """Collapse to non-overlapping five-year averages.

    This is the standard unit of observation in the empirical growth literature
    (Levine, Loayza & Beck 2000; Beck & Levine 2004). Annual growth rates are
    dominated by cyclical noise, and the theory being tested is about capital
    accumulation and allocative efficiency, which operate over a horizon of
    years. Averaging also attenuates the recession-denominator artifact, since
    output gaps substantially close within a five-year window.
    """
    d = panel.copy()
    d["period"] = (d["year"] - START) // 5
    rows = []
    for (iso, period), grp in d.groupby(["iso3", "period"]):
        grp = grp.sort_values("year")
        if len(grp) < 3:          # require most of the window to be observed
            continue
        rows.append({
            "iso3": iso,
            "period": period,
            "period_label": f"{grp['year'].iloc[0]}-{str(grp['year'].iloc[-1])[-2:]}",
            "year_mid": grp["year"].mean(),
            "growth": grp["gdp_pc_growth"].mean(),
            # Initial depth, not period-average depth: a predetermined
            # conditioning variable is the convention, and it keeps the
            # regressor from being contemporaneous with the outcome.
            "credit_initial": grp["credit_gdp"].iloc[0],
            "d_credit": grp["credit_gdp"].iloc[-1] - grp["credit_gdp"].iloc[0],
            "latam": int(iso != "USA"),
            "hyper": bool(grp["hyper"].any()),
        })
    return pd.DataFrame(rows)


# =============================================================================
# 4. Figures
# =============================================================================
SOURCE = "World Bank, World Development Indicators"
LATAM_NOTE = "LatAm-6: Argentina, Brazil, Chile, Colombia, Mexico, Peru, weighted by population"


def chart1_dual_axis(reg: pd.DataFrame):
    """Chart 1 as specified in plan.md: credit depth and growth on two scales.

    A caveat worth keeping in mind when reading it: with two y-scales the eye
    reads crossings and relative slopes as meaningful when they are an artifact
    of where the axis limits were set. Solid/dashed encodes which scale a line
    belongs to, and each line is labelled at its right-hand end, but
    `chart1b_panels.png` plots the same data on separate scales and is the
    safer of the two for drawing conclusions.
    """
    fig, ax = plt.subplots(figsize=(9.5, 5.6))
    ax2 = ax.twinx()

    for region, grp in reg.groupby("region"):
        c = es.SERIES[region]
        ax.plot(grp["year"], grp["credit_gdp"], color=c, lw=2.2, solid_capstyle="round",
                zorder=3)
        ax2.plot(grp["year"], grp["gdp_pc_growth_roll5"], color=c, lw=1.6,
                 linestyle=(0, (4, 2)), zorder=3)

    # Direct labels at the right-hand end of each line, in the series color,
    # so the chart is readable without consulting a legend.
    for region, grp in reg.groupby("region"):
        c, last = es.SERIES[region], grp.dropna(subset=["gdp_pc_growth_roll5"]).iloc[-1]
        ax.annotate(f"{region}\ncredit", (last["year"], last["credit_gdp"]),
                    xytext=(7, 0), textcoords="offset points", va="center",
                    fontsize=9, color=c, fontweight="bold", linespacing=1.1)
        ax2.annotate(f"{region}\ngrowth", (last["year"], last["gdp_pc_growth_roll5"]),
                     xytext=(7, 0), textcoords="offset points", va="center",
                     fontsize=9, color=c, linespacing=1.1)

    ax.set_ylim(0, 240)
    ax2.set_ylim(-4, 8)
    ax.set_xlim(START - 0.5, END + 5.5)      # right-hand room for the labels
    ax.set_xticks(range(START, END, 5))      # no tick past the end of the data

    es.axes(ax, units="Credit to private sector, % of GDP (solid)", y_right=False)
    ax2.spines["top"].set_visible(False)
    ax2.spines["left"].set_visible(False)
    ax2.spines["right"].set_visible(False)
    ax2.tick_params(axis="y", length=0, labelsize=9.5)
    # The growth zero line belongs to the right-hand scale only. Drawn in full
    # ink it would read as a left-axis gridline sitting at 80% of GDP, so it is
    # kept faint and the "0" tick on the right axis carries the meaning.
    ax2.axhline(0, color=es.GRID, lw=1.0, linestyle=(0, (2, 2)), zorder=1)
    es.units_right(ax2, "Real GDP per person, 5-yr avg % change (dashed)")

    es.finish(
        fig,
        title="Deep but not dynamic",
        subtitle="Private credit and economic growth, 1990-2023",
        source=SOURCE,
        note=f"{LATAM_NOTE}. Growth series smoothed with a five-year rolling mean.",
        top_in=1.30, right_in=1.05,
    )
    fig.savefig(FIGS / "chart1_dual_axis.png", dpi=200)
    plt.close(fig)


def chart1b_panels(reg: pd.DataFrame):
    """The same data on separate scales — the honest version of chart 1."""
    fig, (top, bot) = plt.subplots(2, 1, figsize=(9.5, 6.6), sharex=True)
    for region, grp in reg.groupby("region"):
        c = es.SERIES[region]
        top.plot(grp["year"], grp["credit_gdp"], color=c, lw=2.2)
        bot.plot(grp["year"], grp["gdp_pc_growth"], color=c, lw=1, alpha=0.35)
        bot.plot(grp["year"], grp["gdp_pc_growth_roll5"], color=c, lw=2.2)
        last = grp.dropna(subset=["gdp_pc_growth_roll5"]).iloc[-1]
        top.annotate(region, (last["year"], last["credit_gdp"]), xytext=(7, 0),
                     textcoords="offset points", va="center", fontsize=9.5,
                     color=c, fontweight="bold")
        bot.annotate(region, (last["year"], last["gdp_pc_growth_roll5"]),
                     xytext=(7, 0), textcoords="offset points", va="center",
                     fontsize=9.5, color=c, fontweight="bold")

    top.set_xlim(START - 0.5, END + 3.5)
    bot.set_xticks(range(START, END, 5))
    es.axes(top, units="Credit to private sector, % of GDP")
    es.axes(bot, units="Real GDP per person, % change (annual faint, 5-yr avg bold)")
    es.finish(
        fig,
        title="Deep but not dynamic",
        subtitle="Private credit and economic growth, on separate scales, 1990-2023",
        source=SOURCE, note=LATAM_NOTE, top_in=1.30,
    )
    fig.savefig(FIGS / "chart1b_panels.png", dpi=200)
    plt.close(fig)


def chart2_scatter(d: pd.DataFrame):
    """Chart 2: ΔCredit/GDP against growth, with OLS fits by region.

    Hyperinflation country-years are drawn hollow. They are kept in the baseline
    regression for fidelity to the plan but, visually, they are exactly the
    high-leverage points that a reader should discount.
    """
    fig, ax = plt.subplots(figsize=(9, 6))
    d = d.assign(region=np.where(d["latam"] == 1, "LatAm", "USA"))

    for region, grp in d.groupby("region"):
        c = es.SERIES[region]
        clean, hyp = grp[~grp["hyper"]], grp[grp["hyper"]]
        ax.scatter(clean["d_credit"], clean["gdp_pc_growth"], s=30, color=c,
                   alpha=0.70, edgecolor=es.SURFACE, linewidth=0.7, zorder=3)
        ax.scatter(hyp["d_credit"], hyp["gdp_pc_growth"], s=34, facecolor="none",
                   edgecolor=c, linewidth=1.3, zorder=3)
        # Fit on the full regional sample, matching the baseline regression.
        slope, intercept = np.polyfit(grp["d_credit"], grp["gdp_pc_growth"], 1)
        xs = np.linspace(grp["d_credit"].min(), grp["d_credit"].max(), 50)
        ax.plot(xs, intercept + slope * xs, color=c, lw=2, zorder=4)
        # Round before formatting, else a slope of -0.0004 prints as "-0.000".
        ax.annotate(f"{region}: slope {round(slope, 3) + 0.0:+.3f}",
                    (xs[-1], intercept + slope * xs[-1]), xytext=(6, 0),
                    textcoords="offset points", va="center", fontsize=9.5,
                    color=c, fontweight="bold")

    ax.axvline(0, color=es.GRID, lw=1, zorder=1)
    ax.set_xlim(-95, 75)
    es.axes(ax, units="Real GDP per person, % change")
    ax.set_xlabel("Annual change in credit to private sector, percentage points of GDP",
                  fontsize=9.5, labelpad=6)
    es.legend(ax, [("USA", es.SERIES["USA"]), ("LatAm", es.SERIES["LatAm"])],
              loc="lower left")
    es.finish(
        fig,
        title="No visible payoff",
        subtitle="Annual credit expansion against growth, country-years 1990-2023",
        source=SOURCE,
        note="Hollow markers: hyperinflation years (Brazil pre-1995, Argentina pre-1991), "
             "where credit/GDP moves with prices rather than lending.",
        top_in=1.30, right_in=1.15,
    )
    fig.savefig(FIGS / "chart2_scatter.png", dpi=200)
    plt.close(fig)


def chart3_too_much_finance(quin: pd.DataFrame, res_quad):
    """Chart 3: five-year growth against the level of financial depth.

    Tests the plan's diminishing-returns prediction directly. If credit deepening
    pays off mainly at low levels of development, the cloud should slope down.
    """
    fig, ax = plt.subplots(figsize=(9, 6))
    q = quin.assign(region=np.where(quin["latam"] == 1, "LatAm", "USA"))

    for region, grp in q.groupby("region"):
        c = es.SERIES[region]
        ax.scatter(grp["credit_initial"], grp["growth"], s=70, color=c, alpha=0.75,
                   edgecolor=es.SURFACE, linewidth=1.0, zorder=3)

    # Quadratic fit over the pooled five-year observations.
    b = np.polyfit(q["credit_initial"], q["growth"], 2)
    xs = np.linspace(q["credit_initial"].min(), q["credit_initial"].max(), 200)
    ax.plot(xs, np.polyval(b, xs), color=es.INK, lw=1.8, linestyle=(0, (5, 2)),
            zorder=4)
    vertex = -b[1] / (2 * b[0])
    if q["credit_initial"].min() < vertex < q["credit_initial"].max():
        ax.axvline(vertex, color=es.GREY, lw=1, linestyle=":", zorder=2)
        ax.annotate(f"fitted peak\n{vertex:.0f}% of GDP", (vertex, ax.get_ylim()[1]),
                    xytext=(6, -14), textcoords="offset points", fontsize=9,
                    color=es.INK_SOFT, va="top", linespacing=1.2)

    es.axes(ax, units="Real GDP per person, average % change over the period")
    ax.set_xlabel("Credit to private sector at the start of the period, % of GDP",
                  fontsize=9.5, labelpad=6)
    es.legend(ax, [("USA", es.SERIES["USA"]), ("LatAm", es.SERIES["LatAm"])],
              loc="upper right")
    es.finish(
        fig,
        title="Too much finance?",
        subtitle="Financial depth and subsequent growth, non-overlapping five-year periods",
        source=SOURCE,
        note="Dashed line: quadratic fit. One point per country per period; "
             f"n = {len(q)}. Periods with fewer than three observed years are dropped.",
        top_in=1.30,
    )
    fig.savefig(FIGS / "chart3_too_much_finance.png", dpi=200)
    plt.close(fig)
