# Results: Does Access to Credit Drive Economic Growth?

**Sample.** 7 economies (USA, ARG, BRA, CHL, COL, MEX, PER), 1990–2023, N = 217 country-years. Five-year panel: 46 country-period observations.
Produced by `analysis.py` from World Bank WDI. Figures in `figures/`, data in `data/`.

## Headline

**Inconclusive.** Neither criterion in plan.md §5 is met: the estimates are too imprecise to separate the two hypotheses.

The USA effect is significant at 5% in 2 of 7 specifications: Two-way fixed effects (+0.073); Two-way FE, excluding 2020-22 (+0.053). It is not significant in the rest. The LatAm effect is not significant at 5% in any of the 7 specifications.

The baseline model specified in the plan explains 0.3% of the variation in growth. β₄, the Latin America differential, carries the predicted positive sign in that baseline but is nowhere near conventional significance, and it flips sign once hyperinflation years or the pandemic years are excluded. The one specification that does produce a significant coefficient — two-way fixed effects — points the *opposite* way to the plan's prediction: it is the USA, not Latin America, where credit expansion co-moves with growth, and the Latin America differential is then negative. That estimate is not robust (see below), but it is the only signal in the set and it does not favour the hypothesis.

## Specification summary

β₂ is the growth effect of credit expansion in the USA; β₂+β₄ is the implied effect in Latin America, with the covariance between the two terms accounted for.

| Specification | β₂ USA | 95% CI | β₂+β₄ LatAm | 95% CI | N | R² |
|---|---:|:--:|---:|:--:|---:|---:|
| Pooled OLS (plan baseline) | -0.002 | [-0.064, 0.059] | +0.022 | [-0.047, 0.090] | 217 | 0.003 |
| Pooled OLS, clustered by country | -0.002 | [-0.012, 0.007] | +0.022 | [-0.010, 0.053] | 217 | 0.003 |
| Two-way fixed effects | +0.073 | [0.050, 0.097] | -0.007 | [-0.031, 0.017] | 217 | 0.619 |
| Two-way FE, excluding 2020-22 | +0.053 | [0.003, 0.102] | +0.002 | [-0.024, 0.028] | 196 | 0.487 |
| Excluding hyperinflation years | -0.002 | [-0.064, 0.061] | -0.050 | [-0.249, 0.150] | 211 | 0.002 |
| Five-year averages | +0.017 | [-0.019, 0.054] | +0.053 | [-0.008, 0.114] | 46 | 0.097 |
| Quadratic in depth | +0.019 | [-0.041, 0.079] | — | — | 217 | 0.004 |

## Baseline model (plan.md §4)

GDP Growth = α + β₁·Credit/GDP(t−1) + β₂·ΔCredit/GDP + β₃·LatAm + β₄·(ΔCredit/GDP × LatAm)

Pooled OLS, heteroskedasticity-robust (HC1) standard errors.

| Term | Coef. | Std. err. | t / z | p-value |
|---|---:|---:|---:|---:|
| `Intercept` | 1.9374 | 1.6435 | 1.18 | 0.2385 |
| `credit_lag` | -0.0022 | 0.0096 | -0.23 | 0.8212 |
| `d_credit` | -0.0022 | 0.0314 | -0.07 | 0.9450 |
| `latam` | -0.1914 | 1.2701 | -0.15 | 0.8802 |
| `d_credit:latam` | 0.0238 | 0.0468 | 0.51 | 0.6114 |

R² = 0.003, N = 217

* **β₂ = -0.002** (p = 0.945). A one-point rise in credit/GDP is associated with 0.002 pp less growth in the USA — effectively nothing.
* **β₄ = +0.024** (p = 0.611). The LatAm differential is positive, as predicted, but imprecise.
* **β₂+β₄ = +0.022** (p = 0.536), 95% CI [-0.047, 0.090].

## Robustness

### Two-way fixed effects

Country and year dummies absorb fixed institutional differences and common global shocks. The LatAm main effect is collinear with the country dummies and drops.

| Term | Coef. | Std. err. | t / z | p-value |
|---|---:|---:|---:|---:|
| `Intercept` | -4.0401 | 2.1823 | -1.85 | 0.0641 |
| `credit_lag` | -0.0147 | 0.0083 | -1.76 | 0.0778 |
| `d_credit` | 0.0731 | 0.0120 | 6.07 | 0.0000 |
| `d_credit:latam` | -0.0798 | 0.0177 | -4.50 | 0.0000 |

This is the only specification in which anything reaches significance, so it deserves scepticism rather than a victory lap. Three reasons to discount it:

1. **Few clusters.** The p-value comes from a cluster-robust variance estimator with seven clusters. That estimator is consistent in the number of clusters; at seven it is known to be severely downward-biased, so the reported significance is not credible at face value.
2. **Pandemic leverage.** US credit/GDP rose 22 points in 2020 and fell 32 points by 2022 as firms drew down and then repaid credit lines, against the largest output swing in the sample. Dropping 2020-22 roughly halves β₂ and pushes β₄ out of significance:

| Term | Coef. | Std. err. | t / z | p-value |
|---|---:|---:|---:|---:|
| `Intercept` | -3.8789 | 2.1977 | -1.76 | 0.0776 |
| `credit_lag` | -0.0141 | 0.0097 | -1.45 | 0.1474 |
| `d_credit` | 0.0527 | 0.0253 | 2.09 | 0.0370 |
| `d_credit:latam` | -0.0510 | 0.0310 | -1.65 | 0.0998 |

3. **Wrong sign for the hypothesis.** Even taken at face value, it says credit expansion tracks growth in the *mature* financial system, which is the reverse of the mechanism in plan.md §1.

### Excluding hyperinflation years

| Term | Coef. | Std. err. | t / z | p-value |
|---|---:|---:|---:|---:|
| `Intercept` | 1.8050 | 1.6993 | 1.06 | 0.2881 |
| `credit_lag` | -0.0014 | 0.0099 | -0.14 | 0.8883 |
| `d_credit` | -0.0015 | 0.0317 | -0.05 | 0.9615 |
| `latam` | 0.0255 | 1.3293 | 0.02 | 0.9847 |
| `d_credit:latam` | -0.0480 | 0.1080 | -0.44 | 0.6566 |

**Inconclusive.** Neither criterion in plan.md §5 is met: the estimates are too imprecise to separate the two hypotheses.

### Five-year averages

Non-overlapping five-year periods, the standard unit in the growth literature. Depth is measured at the start of each period.

| Term | Coef. | Std. err. | t / z | p-value |
|---|---:|---:|---:|---:|
| `Intercept` | 1.5939 | 1.4651 | 1.09 | 0.2766 |
| `credit_initial` | -0.0009 | 0.0081 | -0.11 | 0.9085 |
| `d_credit` | 0.0172 | 0.0186 | 0.93 | 0.3537 |
| `latam` | -0.0179 | 1.1382 | -0.02 | 0.9874 |
| `d_credit:latam` | 0.0358 | 0.0361 | 0.99 | 0.3217 |

### Quadratic in depth ("too much finance")

| Term | Coef. | Std. err. | t / z | p-value |
|---|---:|---:|---:|---:|
| `Intercept` | 1.9034 | 0.6930 | 2.75 | 0.0060 |
| `credit_lag` | -0.0085 | 0.0194 | -0.44 | 0.6609 |
| `I(credit_lag ** 2)` | 0.0000 | 0.0001 | 0.41 | 0.6814 |
| `d_credit` | 0.0187 | 0.0305 | 0.61 | 0.5391 |

The vertex sits at **115% of GDP**, inside the observed range, which is the shape the 'too much finance' literature predicts — though the quadratic term itself is not significant (p = 0.681), so the hump is not firmly established.

## Financial depth, 2023

| | Credit to private sector, % of GDP |
|---|---:|
| USA | 191.1 |
| LatAm-6, population-weighted | 52.4 |

The level gap is the striking fact in the data: US financial depth is roughly 3.6× the LatAm-6 average, and has been for three decades. What the regressions say is that *changes* in that depth do not track growth in either place.

## What this design cannot settle

* **Composition is unobserved.** WDI private credit pools household and corporate lending. The mechanism in plan.md §1 — productive business credit in LatAm versus consumption credit in the USA — is precisely a composition claim, and this series cannot test it. The BIS credit-to-the-non-financial-sector database splits household from corporate borrowing and would be the natural next dataset.
* **Simultaneity.** ΔCredit/GDP is contemporaneous with growth and GDP is its own denominator, so a recession raises the ratio mechanically. This biases β₂ toward zero or below. Lagging the stock helps; it does not identify anything.
* **Power.** With 7 countries and ~217 observations, the 95% interval on β₂ spans roughly 0.12 points of growth. Effects of a plausible size could not be detected even if present, so "no significant effect" here is weak evidence of absence.
* **Few clusters.** Seven clusters is well below the ~40 rule of thumb for cluster-robust inference; a wild cluster bootstrap would be the appropriate fix.
* **Unbalanced panel.** Chile enters in 2001 and Mexico in 1997, so the LatAm-6 aggregate changes composition mid-series. Brazil holds roughly a third of the group's population and dominates the weighted average.

## Figures

| File | Content |
|---|---|
| `chart1_dual_axis.png` | Credit depth and growth over time, two scales, as specified in the plan |
| `chart1b_panels.png` | The same series on separate scales — preferred for reading levels |
| `chart2_scatter.png` | ΔCredit/GDP against growth with per-region OLS fits |
| `chart3_too_much_finance.png` | Depth against subsequent five-year growth, quadratic fit |
| `chart4_coefficients.png` | Marginal effects with 95% intervals across all specifications |
