COVID sensitivity (Applied Energy reviewer-facing)

1) Run three times (same random seed; only covid_control_mode changes):

   python run_analysis.py --config configs/covid_direct.yaml --output outputs_covid_direct
   python run_analysis.py --config configs/covid_proxy.yaml  --output outputs_covid_proxy
   python run_analysis.py --config configs/covid_none.yaml   --output outputs_covid_none

2) Compile a manuscript-ready summary table:

   python scripts/compile_covid_sensitivity.py \
       --direct outputs_covid_direct \
       --proxy  outputs_covid_proxy \
       --none   outputs_covid_none \
       --out    outputs_covid_sensitivity

Output:
   outputs_covid_sensitivity/tables/table_covid_sensitivity_summary.csv

This table shows point estimates + 95% CI for each mode and flags CI overlap vs 'direct'.
