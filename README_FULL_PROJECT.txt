# Applied Energy Q1 Project (full) + COVID sensitivity

## Quick start (Windows CMD)
1) Create/activate your environment, then from the project root run:

    python run_analysis.py --config configs\applied_energy_q1_fast.yaml --output outputs_main

## COVID sensitivity (direct / proxy / none)
Run three experiments (sequential) from the project root:

    for %m in (direct proxy none) do python run_analysis.py --config configs\covid_%m.yaml --output outputs_covid_%m

Then compile a manuscript-ready summary table:

    python scripts\compile_covid_sensitivity.py --direct outputs_covid_direct --proxy outputs_covid_proxy --none outputs_covid_none --out outputs_covid_sensitivity

Output:
- outputs_covid_sensitivity\tables\table_covid_sensitivity_summary.csv

## Notes
- All CSV writers use UTF-8-SIG so Windows/Excel can open files with unicode characters (e.g., Δ).
- Nested CV now passes a fold-local eval_set where supported so early stopping works and runtime is reduced.
