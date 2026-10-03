# Plan: Does Access to Credit Drive Economic Growth? (USA vs. Latin America)

## 1. Research Question & Expected Findings

* **Primary Question:** Does access to private credit drive overall economic growth, and does this growth relationship differ between the United States and Latin America (LatAm)?
* **Expected Findings:**
  * Credit growth does not unconditionally generate long-term economic wealth.
  * In Latin America, where financial deepening is lower, marginal expansion of business credit is expected to yield higher growth by financing productive capital investment.
  * In mature financial systems like the USA, credit expansion is increasingly skewed toward consumer and personal debt, which yields lower long-term productivity returns.
* **Classification Challenge & Nuance:** Standard macro data aggregates all private credit together. However, personal vs. business credit carries different wealth-generating potential—and dual-use assets complicate this further (for example, a vehicle purchased on credit can be a personal consumption asset or a business capital asset such as a taxi or delivery vehicle).

---

## 2. Data Sources & API Endpoint Specifications

* **API Provider:** World Bank Indicators V2 API (<https://api.worldbank.org/v2/>)
* **Python Package:** `requests` and `pandas` (no API key required for World Bank API).
* **Geographic Coverage:**
  * **USA:** United States (`USA`)
  * **LatAm Representative Economies:** Argentina (`ARG`), Brazil (`BRA`), Chile (`CHL`), Colombia (`COL`), Mexico (`MEX`), Peru (`PER`), plus the regional aggregate Latin America & Caribbean (`LCN`).
* **World Bank Series Codes:**
  1. `FS.AST.PRVT.GD.ZS`: Domestic Credit to Private Sector (% of GDP)
  2. `NY.GDP.PCAP.KD.ZG`: Real GDP Per Capita Growth (Annual %)
  3. `NY.GDP.PCAP.KD`: Real GDP Per Capita (Constant 2015 US$)
  4. `SP.POP.TOTL`: Total Population (used for population-weighted regional aggregation)

---

## 3. Data Cleaning & Transformation Pipeline

1. **Time Horizon:** Filter observations for the period `1990` through `2023` to evaluate multi-decade credit deepening.
2. **Missing Data Handling:** Drop country-year pairs with missing credit or GDP indicators; forward-fill brief gaps ($\le 2$ years) where appropriate.
3. **Population-Weighted Aggregation for LatAm:**
   * Calculate population-weighted averages for Latin American metrics to ensure small countries do not disproportionately bias regional comparisons against the USA:

   $$\text{LatAm Credit Ratio}_t = \frac{\sum_{i} \left(\text{Credit/GDP}_{i,t} \times \text{Population}_{i,t}\right)}{\sum_{i} \text{Population}_{i,t}}$$

4. **Calculations & Transformations:**
   * **Credit Expansion Rate ($\Delta \text{Credit/GDP}_t$):** First-difference of private credit as a percentage of GDP ($\text{Credit/GDP}_t - \text{Credit/GDP}_{t-1}$).
   * **Lagged Credit Ratio ($\text{Credit/GDP}_{t-1}$):** Lagged by 1 year to reduce endogeneity and reverse causality (where high GDP growth generates credit demand).
   * **5-Year Rolling Averages:** Smooth out short-term business cycle volatility for structural comparison.

---

## 4. Visualizations & Empirical Economic Models

### Comparative Visualizations

* **Chart 1 (Time-Series Trajectory):** Dual-axis line plot comparing Private Credit (% of GDP) and Real GDP Per Capita Growth from 1990 to 2023 between the USA and LatAm.
* **Chart 2 (Growth vs. Credit Scatter & Trendlines):** Scatter plot of Annual Change in Credit-to-GDP ($\Delta \text{Credit/GDP}$) vs. Real GDP Per Capita Growth, with separate Ordinary Least Squares (OLS) regression trendlines for the USA vs. LatAm.

### Statistical Test & Economic Model

Run an OLS panel/time-series regression model to estimate the elasticity of GDP per capita growth with respect to credit depth and credit expansion:

$$\text{GDP Growth}_{i,t} = \alpha + \beta_1 (\text{Credit/GDP}_{i,t-1}) + \beta_2 (\Delta \text{Credit/GDP}_{i,t}) + \beta_3 (\text{LatAm Dummy}) + \beta_4 (\Delta \text{Credit/GDP}_{i,t} \times \text{LatAm Dummy}) + \epsilon_{i,t}$$

* **$\beta_2$:** Measures the growth effect of credit expansion in the baseline economy (USA).
* **$\beta_4$ (Interaction Term):** Tests whether the marginal impact of credit expansion on growth is significantly higher or lower in Latin America compared to the USA.

---

## 5. Evaluation Criteria (Support vs. Contradiction)

* **Supports Expectation:**
  * $\beta_2$ is near zero or statistically insignificant for the USA, while $\beta_4$ is positive and statistically significant ($p < 0.05$), showing that credit expansion drives greater growth in developing LatAm markets than in mature US markets.
  * Scatter plot reveals diminishing returns to credit growth at higher baseline debt levels.
* **Contradicts Expectation:**
  * $\beta_2$ is strongly positive and equal across both regions, showing that debt unconditionally drives GDP growth regardless of the country's development stage or debt composition.
