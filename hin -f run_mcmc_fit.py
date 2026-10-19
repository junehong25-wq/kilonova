[1mdiff --git a/scripts/run_mcmc_fit.py b/scripts/run_mcmc_fit.py[m
[1mindex 8eb7d22..6fc77b4 100644[m
[1m--- a/scripts/run_mcmc_fit.py[m
[1m+++ b/scripts/run_mcmc_fit.py[m
[36m@@ -69,7 +69,7 @@[m [mdef run_single_case(target_case):[m
 [m
     nwalkers = 48[m
     ncpu = 48[m
[31m-    mcmc_steps = 10000[m
[32m+[m[32m    mcmc_steps = 20000[m
     burn_in = 4000      # 3일차 이중 모드 수렴을 위해 4,000으로 상향[m
     thin_step = 2[m
 [m
[36m@@ -185,7 +185,7 @@[m [mdef run_single_case(target_case):[m
                     p_cand = cand[m
                     break[m
             if p_cand is None:[m
[31m-                for _ in range(10000):[m
[32m+[m[32m                for _ in range(20000):[m
                     cand = np.random.uniform(low_b, high_b)[m
                     if prob_wrapper.log_prior(cand) > -1e10:[m
                         p_cand = cand[m
