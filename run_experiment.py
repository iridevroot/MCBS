"""
run_experiment.py
==================
Runs the full MCBS experiment 

"""
import numpy as np
import mcbs_simulation as m
from scipy import stats
import math

import hashlib

def _deterministic_seed(*parts):
    
    s = "|".join(parts).encode("utf-8")
    return int(hashlib.sha256(s).hexdigest(), 16) % (2**31)

def run_experiment_raw():
    raw = {}
    for scen_name, generators in m.SCENARIOS.items():
        true_means = np.array([m.true_expected_utility(g) for g in generators])
        for strat_name, strat_fn in [("random", m.run_random),
                                       ("uniform", m.run_uniform),
                                       ("greedy", m.run_greedy),
                                       ("MCBS", m.run_mcbs)]:
            cu_list, oia_list = [], []
            for run_idx in range(m.R_RUNS):
                seed = _deterministic_seed(scen_name, strat_name, str(run_idx))
                rng = np.random.default_rng(seed)
                cu, identified = strat_fn(generators, true_means, rng)
                cu_list.append(cu)
                oia_list.append(identified)
            raw[(scen_name, strat_name)] = (np.array(cu_list), np.array(oia_list, dtype=float))
    return raw

raw = run_experiment_raw()

print("===  (Descriptive) ===")
print(f"{'Scenario':<10} {'Strategy':<8} {'Avg CU':>10} {'SD CU':>10} {'OIA':>6}")
for (scen, strat), (cu, oia) in raw.items():
    print(f"{scen:<10} {strat:<8} {cu.mean():>10.2f} {cu.std(ddof=1):>10.2f} {oia.mean():>6.3f}")

print("\n=== (CU: MCBS vs Greedy, Welch's t) ===")
for scen in m.SCENARIOS:
    cu_m, _ = raw[(scen, "MCBS")]
    cu_g, _ = raw[(scen, "greedy")]
    n1, n2 = len(cu_m), len(cu_g)
    m1, m2 = cu_m.mean(), cu_g.mean()
    s1, s2 = cu_m.std(ddof=1), cu_g.std(ddof=1)
    se = math.sqrt(s1**2/n1 + s2**2/n2)
    t = (m1-m2)/se
    df = (s1**2/n1+s2**2/n2)**2 / ((s1**2/n1)**2/(n1-1)+(s2**2/n2)**2/(n2-1))
    p = 2*(1-stats.t.cdf(abs(t), df))
    tcrit = stats.t.ppf(0.975, df)
    ci = ((m1-m2)-tcrit*se, (m1-m2)+tcrit*se)
    d = (m1-m2)/math.sqrt((s1**2+s2**2)/2)
    print(f"{scen}: diff={m1-m2:.2f}, t({df:.1f})={t:.2f}, p={p:.4g}, CI=[{ci[0]:.2f},{ci[1]:.2f}], d={d:.3f}")

print("\n=== (OIA: MCBS vs Greedy, two-proportion z) ===")
for scen in m.SCENARIOS:
    _, oia_m = raw[(scen, "MCBS")]
    _, oia_g = raw[(scen, "greedy")]
    n1=n2=500
    p1, p2 = oia_m.mean(), oia_g.mean()
    p_pool = (p1*n1+p2*n2)/(n1+n2)
    se = math.sqrt(p_pool*(1-p_pool)*(1/n1+1/n2)) if p_pool not in (0,1) else 1e-9
    z = (p1-p2)/se if se>0 else float('nan')
    p = 2*(1-stats.norm.cdf(abs(z))) if se>0 else float('nan')
    se_unp = math.sqrt(p1*(1-p1)/n1+p2*(1-p2)/n2)
    ci = ((p1-p2)-1.96*se_unp, (p1-p2)+1.96*se_unp)
    def phi(p): return 2*math.asin(math.sqrt(p))
    h = phi(p1)-phi(p2)
    print(f"{scen}: diff={p1-p2:.3f}, z={z:.3f}, p={p:.4g}, CI=[{ci[0]:.3f},{ci[1]:.3f}], h={h:.3f}")
