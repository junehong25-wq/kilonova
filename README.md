# AT2017gfo Pure NLTE Spectrum Fitting

A 7D Pure NLTE radiative transfer and MCMC fitting pipeline for kilonova AT2017gfo spectra, considering relativistic beaming, light travel time (LTT), and non-thermal helium/strontium excitations.

## Structure
- `src/radiation_engine.py`: Sobolev radiative transfer engine and Numba-accelerated optical depth functions.
- `src/probability.py`: MCMC likelihood and phase-dependent physical priors wrapper.
- `scripts/run_mcmc_fit.py`: Main execution script for emcee MCMC sampling and plotting.
- `.github/AGENT_INSTRUCTIONS.md`: Rules and constraints for AI coding agents (Jules/Copilot).
