using JuMP
using HiGHS
using Random
using Distributions
using Statistics
using Plots
using StatsPlots

gr()

############################################################
# SECTION 1 — MODEL PARAMETERS
# Defines assets, liabilities, risk parameters, and constants.
############################################################

seed = 1234
Random.seed!(seed)

asset_names = ["Equities","Bonds","Cash"]
n_assets = length(asset_names)
n_scen = 200

A0 = 1.0   # initial portfolio value

# Liability distribution parameters (lognormal)
liability_mean = 1.0
liability_vol  = 0.10

# Asset return parameters (single-period)
mu_equity = 0.06;  sigma_equity = 0.18
mu_bond   = 0.02;  sigma_bond   = 0.06
mu_cash   = 0.01;  sigma_cash   = 0.005
rho_eq_bond = 0.2

# Transaction cost rate applied to turnover in weights
tc_rate = 0.002

# CVaR parameters
alpha = 0.95
lambda_risk = 1.0

############################################################
# SECTION 2 — SCENARIO GENERATION
# Generates correlated asset returns and stochastic liabilities.
############################################################

# Covariance matrix for equities and bonds
Σ = [sigma_equity^2          rho_eq_bond * sigma_equity * sigma_bond;
     rho_eq_bond * sigma_equity * sigma_bond   sigma_bond^2]

μ = [mu_equity, mu_bond]
mv_normal = MvNormal(μ, Σ)

# Scenario arrays
equity_ret = zeros(n_scen)
bond_ret   = zeros(n_scen)
cash_ret   = zeros(n_scen)
liab_T     = zeros(n_scen)

# Draw asset returns and liabilities for each scenario
for s in 1:n_scen
    r = rand(mv_normal)
    equity_ret[s] = r[1]
    bond_ret[s]   = r[2]
    cash_ret[s]   = rand(Normal(mu_cash, sigma_cash))

    # Lognormal liability
    liab_T[s] = rand(LogNormal(
        log(liability_mean) - 0.5 * liability_vol^2,
        liability_vol
    ))
end

# Scenario probabilities (uniform)
p = fill(1.0/n_scen, n_scen)

############################################################
# SECTION 3 — TWO-STAGE STOCHASTIC MODEL WITH RECOURSE
# Stage 1: choose initial weights w
# Stage 2: choose scenario-dependent rebalanced weights w2[s,*]
# Recourse cost = transaction costs from turnover in weights.
############################################################

model = Model(HiGHS.Optimizer)

# ---------- Stage 1 decision: initial portfolio weights ----------
@variable(model, w[1:n_assets] >= 0)
@constraint(model, sum(w) == 1)   # fully invested

# ---------- Stage 2 decision: scenario-specific rebalanced weights ----------
@variable(model, w2[1:n_scen, 1:n_assets] >= 0)

# Turnover decomposition:
# w2[s,i] - w[i] = pos - neg ensures linear absolute value
@variable(model, pos[1:n_scen, 1:n_assets] >= 0)
@variable(model, neg[1:n_scen, 1:n_assets] >= 0)

@constraint(model, [s=1:n_scen, i=1:n_assets],
    w2[s,i] - w[i] == pos[s,i] - neg[s,i]
)

# Rebalanced weights must also sum to 1 in each scenario
@constraint(model, [s=1:n_scen],
    sum(w2[s,i] for i in 1:n_assets) == 1
)

# Final asset value after rebalancing and transaction costs
@expression(model, A_final[s=1:n_scen],
    A0 * (
        sum(w2[s,i] * (1 + (i==1 ? equity_ret[s] :
                            i==2 ? bond_ret[s]   :
                                    cash_ret[s])) for i in 1:n_assets)
        - tc_rate * sum(pos[s,i] + neg[s,i] for i in 1:n_assets)
    )
)

# Surplus = final assets − liability
@variable(model, surplus[1:n_scen])
@constraint(model, [s=1:n_scen],
    surplus[s] == A_final[s] - liab_T[s]
)

############################################################
# SECTION 4 — CVaR RISK MEASURE
# Implements Rockafellar–Uryasev linear CVaR formulation.
############################################################

# Loss = −surplus
@variable(model, loss[1:n_scen])
@constraint(model, [s=1:n_scen], loss[s] == -surplus[s])

# CVaR auxiliary variables
@variable(model, eta)               # VaR threshold
@variable(model, z[1:n_scen] >= 0)  # excess losses

# z[s] ≥ loss[s] − eta
@constraint(model, [s=1:n_scen], z[s] >= loss[s] - eta)

# Expected surplus and CVaR expression
@expression(model, mean_surplus, sum(p[s]*surplus[s] for s in 1:n_scen))
@expression(model, cvar, eta + (1/(1-alpha))*sum(p[s]*z[s] for s in 1:n_scen))

# Objective: maximize mean surplus minus risk penalty
@objective(model, Max, mean_surplus - lambda_risk * cvar)

optimize!(model)

############################################################
# SECTION 5 — SOLUTION EXTRACTION
############################################################

status = termination_status(model)
println("Status: ", status)

if status != MOI.OPTIMAL && status != MOI.LOCALLY_SOLVED
    error("Model not solved to optimality; status = $status")
end

println("Objective: ", objective_value(model))

w_opt  = value.(w)     # stage 1 weights
w2_opt = value.(w2)    # stage 2 weights
pos_opt = value.(pos)
neg_opt = value.(neg)

println("Initial weights (stage 1): ", w_opt)

############################################################
# SECTION 6 — REPORTING & SUMMARY STATISTICS
############################################################

# Average rebalanced weights across scenarios
avg_w2 = vec(mean(w2_opt, dims=1))
data = hcat(w_opt[:], avg_w2[:])

println("Average rebalanced weights (stage 2): ", avg_w2)
println("Difference (avg rebalanced - initial): ", avg_w2 .- w_opt)

# Turnover per scenario (sum of absolute weight changes)
turnover = [sum(pos_opt[s,i] + neg_opt[s,i] for i in 1:n_assets)
            for s in 1:n_scen]

############################################################
# SECTION 7 — VISUALIZATION
# Includes:
#   • Side-by-side bar chart of initial vs rebalanced weights
#   • Heatmap of scenario-specific rebalanced weights
#   • Histogram of turnover
############################################################

# --- Side-by-side bar plot ---
weights = hcat(w_opt, avg_w2)  # matrix: assets × {initial, rebalanced}

display(
    groupedbar(
        asset_names,
        weights,
        bar_position = :dodge,
        title = "Initial vs Rebalanced Weights",
        xlabel = "Asset Class",
        ylabel = "Weight",
        label = ["Initial" "Rebalanced"],
        ylim = (0, 1)
    )
)

# --- Heatmap of rebalanced weights by scenario ---
display(
    heatmap(
        1:n_scen,
        asset_names,
        w2_opt',
        title="Rebalanced Weights by Scenario",
        xlabel="Scenario",
        ylabel="Asset"
    )
)

# --- Turnover distribution ---
display(
    histogram(
        turnover,
        bins = 10,
        title = "Turnover Distribution (in Weights)",
        xlabel = "Total Turnover (sum |Δweight|)",
        label = "Turnover"
    )
)