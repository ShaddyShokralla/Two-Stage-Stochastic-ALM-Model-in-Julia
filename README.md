A fully worked two‑stage stochastic Asset‑Liability Management (ALM) model implemented in Julia using JuMP and HiGHS.  
This project demonstrates how to build a realistic pension‑fund ALM engine with scenario‑dependent recourse, CVaR risk control, and transaction‑cost‑aware rebalancing — all within a linear optimization framework.

Key Features
Two‑stage stochastic optimization  
Stage 1 selects initial portfolio weights; Stage 2 performs scenario‑specific rebalancing (recourse).

Scenario generation  
Correlated equity and bond returns via multivariate normal distribution, plus lognormal liability shocks.

CVaR risk measure  
Implements the Rockafellar–Uryasev linear CVaR formulation to control tail risk.

Transaction costs  
Linear turnover decomposition (pos/neg variables) models realistic rebalancing costs while keeping the model linear.

Visualization tools  
Includes plots for initial vs. rebalanced weights, scenario‑dependent rebalancing patterns, and turnover distribution.

Fast and fully linear  
Solves quickly using HiGHS, making it suitable for experimentation, teaching, and extension to multi‑period ALM.

What You Can Do With This Model
Explore how optimal allocations change under liability uncertainty

Study the effect of CVaR levels on portfolio robustness

Analyze turnover behavior and recourse intensity

Extend the model to multi‑period ALM, stochastic interest rates, or richer asset universes
