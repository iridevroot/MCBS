"""
MCBS (Monte Carlo-Inspired Business Strategy)

"""

import numpy as np

rng_master = np.random.default_rng(42)

# -----------------------------------
# ASSUMED NUMERIC PARAMETERS 
# -----------------------------
ALPHA, BETA, GAMMA, DELTA, LAMBDA = 2.0, 0.3, 0.3, 0.2, 0.2
ETA = 8.0             
EPSILON = 0.10        
N_OPP = 10            
T_HORIZON = 1000      
R_RUNS = 500          

# ------------------------------
# OPPORTUNITY PROFILE GENERATORS

# ----------------------------

def high_potential_low_sigma(rng, t):
    P = rng.normal(80, 5)
    C = rng.normal(20, 3); R = rng.normal(15, 3)
    T = rng.normal(10, 2); S = rng.normal(10, 2)
    return P, C, R, T, S

def moderate_performance(rng, t):
    P = rng.normal(40, 4)
    C = rng.normal(20, 3); R = rng.normal(15, 3)
    T = rng.normal(10, 2); S = rng.normal(10, 2)
    return P, C, R, T, S

def high_potential_high_risk(rng, t):
    P = rng.normal(85, 25)          
    C = rng.normal(25, 5); R = rng.normal(30, 8)
    T = rng.normal(12, 3); S = rng.normal(12, 3)
    return P, C, R, T, S

def red_herring(rng, t, decay_tau=60, bias0=45):
    true_mean_P = 25.0                      
    bias = bias0 * np.exp(-t / decay_tau)     
    P = rng.normal(true_mean_P + bias, 6)
    C = rng.normal(15, 3); R = rng.normal(10, 3)
    T = rng.normal(8, 2); S = rng.normal(8, 2)
    return P, C, R, T, S

def efficiency_based(rng, t):
    P = rng.normal(35, 4)          
    C = rng.normal(5, 1); R = rng.normal(8, 2)   
    T = rng.normal(3, 1); S = rng.normal(6, 1)
    return P, C, R, T, S

def masked_high_potential(rng, t):
   
    P = rng.normal(90, 35)
    C = rng.normal(25, 5); R = rng.normal(28, 9)
    T = rng.normal(12, 3); S = rng.normal(12, 3)
    return P, C, R, T, S

# -------------------------
# SCENARIO DEFINITIONS 
# ----------------------
SCENARIOS = {
    "A_Stable": (
        [high_potential_low_sigma] + [moderate_performance] * 9
    ),
    "B_Noisy": (
        [high_potential_high_risk] + [red_herring] * 9
    ),
    "C_Rare": (
        [masked_high_potential] + [efficiency_based] * 5 + [red_herring] * 4
    ),
}
assert all(len(v) == N_OPP for v in SCENARIOS.values())


# -----------------------
# UTILITY FUNCTION 
# --------------------------
class RunningMinMax:
    def __init__(self, n_attrs=5):
        self.min = np.full(n_attrs, np.inf)
        self.max = np.full(n_attrs, -np.inf)

    def update_and_normalize(self, x):
        x = np.asarray(x, dtype=float)
        self.min = np.minimum(self.min, x)
        self.max = np.maximum(self.max, x)
        rng_ = self.max - self.min
        rng_[rng_ == 0] = 1.0  
        return (x - self.min) / rng_

def compute_utility(x_norm):
    P, C, R, T, S = x_norm
    return ALPHA * P - BETA * C - GAMMA * R - DELTA * T - LAMBDA * S

# ------------------------
# ALGORITHM 1: MCBS 
# -------------------------
def run_mcbs(generators, true_means, rng, T=T_HORIZON, epsilon=EPSILON, eta=ETA):
    n = len(generators)
    N = np.zeros(n)
    Uhat = np.zeros(n)
    pi = np.full(n, 1.0 / n)          
    normalizer = RunningMinMax()
    cum_utility = 0.0
    for t in range(1, T + 1):
        
        pi_tilde = (1 - epsilon) * pi + epsilon * (1.0 / n)
        pi_tilde = pi_tilde / pi_tilde.sum()
        o_t = rng.choice(n, p=pi_tilde)

       
        raw = generators[o_t](rng, t)
        x_norm = normalizer.update_and_normalize(raw)
        U_t = compute_utility(x_norm)
        cum_utility += U_t

       
        N[o_t] += 1
        Uhat[o_t] = Uhat[o_t] + (1.0 / N[o_t]) * (U_t - Uhat[o_t])

        
        exp_vals = np.exp(eta * Uhat)
        pi = exp_vals / exp_vals.sum()

    best_true = int(np.argmax(true_means))
    identified_best = int(np.argmax(Uhat)) == best_true
    return cum_utility, identified_best

# ----------------------
# BASELINES
# ----------------------
def run_random(generators, true_means, rng, T=T_HORIZON):
    n = len(generators)
    Uhat = np.zeros(n); N = np.zeros(n)
    normalizer = RunningMinMax()
    cum_utility = 0.0
    for t in range(1, T + 1):
        o_t = rng.integers(0, n)
        raw = generators[o_t](rng, t)
        x_norm = normalizer.update_and_normalize(raw)
        U_t = compute_utility(x_norm)
        cum_utility += U_t
        N[o_t] += 1
        Uhat[o_t] += (1.0 / N[o_t]) * (U_t - Uhat[o_t])
    best_true = int(np.argmax(true_means))
    identified_best = int(np.argmax(Uhat)) == best_true
    return cum_utility, identified_best

def run_uniform(generators, true_means, rng, T=T_HORIZON):
    
    n = len(generators)
    Uhat = np.zeros(n); N = np.zeros(n)
    normalizer = RunningMinMax()
    cum_utility = 0.0
    for t in range(1, T + 1):
        o_t = (t - 1) % n
        raw = generators[o_t](rng, t)
        x_norm = normalizer.update_and_normalize(raw)
        U_t = compute_utility(x_norm)
        cum_utility += U_t
        N[o_t] += 1
        Uhat[o_t] += (1.0 / N[o_t]) * (U_t - Uhat[o_t])
    best_true = int(np.argmax(true_means))
    identified_best = int(np.argmax(Uhat)) == best_true
    return cum_utility, identified_best

def run_greedy(generators, true_means, rng, T=T_HORIZON):
    n = len(generators)
    Uhat = np.zeros(n); N = np.zeros(n)
    normalizer = RunningMinMax()
    cum_utility = 0.0
 
    for t in range(1, n + 1):
        o_t = t - 1
        raw = generators[o_t](rng, t)
        x_norm = normalizer.update_and_normalize(raw)
        U_t = compute_utility(x_norm)
        cum_utility += U_t
        N[o_t] += 1
        Uhat[o_t] += (1.0 / N[o_t]) * (U_t - Uhat[o_t])
    for t in range(n + 1, T + 1):
        o_t = int(np.argmax(Uhat))  
        raw = generators[o_t](rng, t)
        x_norm = normalizer.update_and_normalize(raw)
        U_t = compute_utility(x_norm)
        cum_utility += U_t
        N[o_t] += 1
        Uhat[o_t] += (1.0 / N[o_t]) * (U_t - Uhat[o_t])
    best_true = int(np.argmax(true_means))
    identified_best = int(np.argmax(Uhat)) == best_true
    return cum_utility, identified_best


# ------------------------
# EXPECTED UTILITY PER OPPORTUNITY (for computing OIA)

# -----------------------
PROFILE_TRUE_RAW_MEANS = {
    high_potential_low_sigma:  (80, 20, 15, 10, 10),
    moderate_performance:      (40, 20, 15, 10, 10),
    high_potential_high_risk:  (85, 25, 30, 12, 12),
    red_herring:               (25, 15, 10, 8, 8),   
    efficiency_based:          (35, 5, 8, 3, 6),
    masked_high_potential:     (90, 25, 28, 12, 12),
}

def true_expected_utility(gen):
    P, C, R, T, S = PROFILE_TRUE_RAW_MEANS[gen]
    return ALPHA * P - BETA * C - GAMMA * R - DELTA * T - LAMBDA * S

# ----------------------------
# MAIN EXPERIMENT LOOP
# ----------------------------
def run_experiment():
    results = {}
    for scen_name, generators in SCENARIOS.items():
        true_means = np.array([true_expected_utility(g) for g in generators])
        for strat_name, strat_fn in [("random", run_random),
                                       ("uniform", run_uniform),
                                       ("greedy", run_greedy),
                                       ("MCBS", run_mcbs)]:
            cu_list, oia_list = [], []
            for run_idx in range(R_RUNS):
                rng = np.random.default_rng(1000 * hash(scen_name + strat_name) % (2**31) + run_idx)
                cu, identified = strat_fn(generators, true_means, rng)
                cu_list.append(cu)
                oia_list.append(identified)
            cu_arr = np.array(cu_list)
            results[(scen_name, strat_name)] = {
                "avg_CU": cu_arr.mean(),
                "sd_CU": cu_arr.std(ddof=1),
                "OIA": np.mean(oia_list),
            }
    return results

if __name__ == "__main__":
    res = run_experiment()
    print(f"{'Scenario':<10} {'Strategy':<8} {'Avg CU':>10} {'SD CU':>10} {'OIA':>6}")
    for (scen, strat), vals in res.items():
        print(f"{scen:<10} {strat:<8} {vals['avg_CU']:>10.2f} {vals['sd_CU']:>10.2f} {vals['OIA']:>6.3f}")
