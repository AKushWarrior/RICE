# RICE: Regularized Indicator of Competitive Efficiency

## Overview & Motivation

**RICE** establishes a strictly forward-predictive model for NFL team ability, isolated into Offensive, Defensive, and Special Teams unit ratings. Standard Expected Points Added (EPA) models treat all plays equally and suffer from severe sample-size noise early in the season.

This architecture utilizes `nflreadpy` to ingest raw play-by-play data and applies regularized regression, situational leverage weighting, and temporal decay to generate efficient, opponent-adjusted unit priors. These priors are then combined with game-state context (rest, weather, market lines) to calculate an optimized win probability $P(Win)$ for Week $N+1$.

-----

## Methodology: Foundations

The following procedures represent the core rating derivation process for each unit, utilizing the most efficient deterministic methodologies established in sports modeling literature.

### 1\. Base Data & Leverage-Weighted EPA (wEPA)

> **Explicit Assumption:** Raw EPA natively overvalues high-variance, structurally noisy events (e.g., 80-yard broken coverages in the 4th quarter of a blowout).

**Derivation:** We extract base EPA per play from `nflreadpy` (backed by Polars for efficient memory mapping). To isolate predictive signal, we apply a continuous decay function to the play's weight based on the pre-snap Win Probability ($WP$). Plays occurring outside the $10\% \le WP \le 90\%$ boundary are exponentially discounted to prevent "garbage time" from skewing a unit's true performance baseline. Let $k$ be a specific play:

$$WEPA_k = EPA_k \cdot w(WP_k)$$

where $w$ is the leverage weight function.

### 2\. Opponent Adjustment via Tikhonov Regularization

**Derivation:** A unit's raw WEPA is a function of its own strength, the opponent's strength, and environment. We isolate true unit strength by solving a large-scale system of linear equations across all plays.

We define the sparse design matrix $X$, where each row represents a play, and columns represent dummy variables for the offense, defense, and Home Field Advantage (HFA). We use Ridge Regression ($L_2$ regularization) via `scipy.sparse` to prevent overfitting in small sample sizes (Weeks 1-4). The objective is to find the vector of unit ratings $\beta$ that minimizes the loss function:

$$L(\beta) = \|y - X\beta\|^2_2 + \lambda \|\beta\|^2_2$$

where $y$ is the vector of $WEPA_k$ values, and $\lambda$ is the penalty parameter optimized via cross-validation. This yields arbitrary-scale unit ratings in Expected Points per Play.

### 3\. Temporal Updating via Exponential Decay

**Derivation:** NFL team quality is non-stationary. When solving the Ridge Regression, we apply a time-weighting matrix $W$ to the observations. Based on standard time-series heuristics for NFL schedules, we apply an exponential decay function to historical games:

$$\phi(t) = e^{-\xi t}$$

where $t$ is the days elapsed since the game, and $\xi$ is the decay rate tuned to optimize out-of-sample log-loss.

-----

## Methodology: Theses

The following are ML assertions (hypotheses) that propose to optimize predictive accuracy by addressing structural flaws in the foundation architecture.

### Thesis A: Bivariate Count Modeling for Discrete Scoring (Dixon-Coles)

  * **Assertion:** Predicting point margin via continuous regression (RMSE) fails to capture the discrete reality of NFL scoring (chunks of 3 and 7). Standard ML models treat a predicted margin of 4.5 as mathematically valid, which distorts moneyline conversions.
  * **Implementation Case:** Map the regularized unit EPA ratings to expected scoring events (Touchdowns and Field Goals per drive). Adapt the Dixon-Coles (1997) framework, which models the exact scoreline of a game using a Bivariate Poisson distribution. We model the Home ($X$) and Away ($Y$) scores using independent Poisson rates ($\lambda, \mu$) derived from the unit ratings, but apply a non-linear dependence parameter $\tau(x,y)$ to account for the correlation between scores in tied, late-game states:

$$P(X=x, Y=y) = \tau(x,y) \frac{\lambda^x e^{-\lambda}}{x!} \frac{\mu^y e^{-\mu}}{y!}$$

  * **Why it Optimizes:** It natively outputs a probability matrix for *exact* scorelines. This naturally accounts for key numbers (3, 7, 10) without relying on a secondary, historically mapped normal distribution.

### Thesis B: Latent State-Space Tracking via Kalman Filtering

  * **Assertion:** The static exponential decay function ($\phi(t)$) in the Foundation model is a blunt instrument. It over-reacts to noisy outcomes for inherently stable teams and under-reacts to true paradigm shifts.
  * **Implementation Case:** Replace the Weighted Ridge Regression with a Gaussian Linear State-Space Model. Treat true unit strength $\theta$ as a hidden Markov state that evolves over time. The state evolution equation updates week-to-week via a Kalman Filter:

$$\theta_{t} \sim \mathcal{N}(\gamma \theta_{t-1}, \sigma_{evol}^2)$$

  * **Why it Optimizes:** The Kalman Filter natively outputs both the point estimate of a unit's strength and its *variance* ($\sigma^2$). If a team is highly erratic, the model mathematically widens the variance and trusts the historical prior less, reacting dynamically to instability.

### Thesis C: Bottom-Up VORP Aggregation for Personnel Variance

  * **Assertion:** Macro-level unit ratings fail completely when key personnel (specifically Quarterbacks) change abruptly due to injury. Top-down opponent adjustment is purely backwards-looking.
  * **Implementation Case:** Maintain the macro-unit prior as a baseline, but integrate a bottom-up micro-adjustment. Engineer a feature that calculates the rolling Value Over Replacement Player (VORP) in EPA/play for all starting Quarterbacks and their backups. When predicting Week $N+1$, strictly override the Offensive prior by the historical delta in VORP between the starter and the backup.
  * **Why it Optimizes:** It injects forward-looking, deterministic constraints into an autoregressive time-series model. Tree-based models cannot easily infer the impact of a Sunday morning inactive list from tabular historical data without explicit feature engineering.