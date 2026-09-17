import numpy as np
from src.models import planck_with_mod_full_relativistic

class MCMCProbabilityWrapper(object):
    def __init__(self, x_fit, y_fit, err_fit, time_s, bounds, use_nlte=True, use_he=True, days=0.0):
        self.x_fit = x_fit
        self.y_fit = y_fit
        self.err_fit = err_fit
        self.time_s = time_s
        self.bounds = bounds
        self.use_nlte = use_nlte
        self.use_he = use_he
        self.days = days

    def log_prior(self, theta):
        if np.any(np.isnan(theta)) or np.any(np.isinf(theta)):
            return -np.inf

        if self.use_he:
            T_prime, N_29, vmax, vphot, tau_sr, tau_he, trans = theta
        else:
            T_prime, N_29, vmax, vphot, tau_sr, trans = theta
            tau_he = 0.0

        for val, (low, high) in zip(theta, self.bounds):
            if not (low <= val <= high):
                return -np.inf

        # Phase-dependent priors and hydrodynamic constraint from Physics Rules Memory
        if vphot >= vmax - 0.03 or tau_sr <= 0.001 or N_29 <= 0.0:
            return -np.inf

        if self.days < 2.5:
            if not (tau_sr >= tau_he):
                return -np.inf

        return 0.0

    def log_likelihood(self, theta):
        if self.use_he:
            T_prime, N_29, vmax, vphot, tau_sr, tau_he, trans = theta
        else:
            T_prime, N_29, vmax, vphot, tau_sr, trans = theta
            tau_he = 0.0

        try:
            model = planck_with_mod_full_relativistic(
                wav=self.x_fit, T_prime=T_prime, N_29=N_29, vmax=vmax, vphot=vphot,
                tau_sr=tau_sr, tau_he=tau_he, trans=trans,
                t0=self.time_s, use_nlte=self.use_nlte, use_he=self.use_he
            )
            if np.any(np.isnan(model)) or np.any(np.isinf(model)):
                return -np.inf
            total_chi2 = np.sum(((self.y_fit - model) / self.err_fit) ** 2)
            return -0.5 * total_chi2
        except Exception:
            return -np.inf

    def __call__(self, theta):
        lp = self.log_prior(theta)
        if not np.isfinite(lp):
            return -np.inf
        ll = self.log_likelihood(theta)
        return lp + ll if np.isfinite(ll) else -np.inf

    def chi2_for_minimizer(self, theta):
        lp = self.log_prior(theta)
        if not np.isfinite(lp):
            return 1e12
        ll = self.log_likelihood(theta)
        return -2.0 * ll if np.isfinite(ll) else 1e12
