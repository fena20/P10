#!/usr/bin/env python3
"""Compile COVID sensitivity outputs into a manuscript-ready table.

Usage:
  python scripts/compile_covid_sensitivity.py \
      --direct outputs_covid_direct \
      --proxy  outputs_covid_proxy \
      --none   outputs_covid_none \
      --out    outputs_covid_sensitivity

Reads:
  - tables/table3_uncertainty.csv
  - tables/table_policy_metrics.csv

Writes:
  - out/tables/table_covid_sensitivity_summary.csv
"""

import argparse
import os
import re
import pandas as pd

def _read_csv(path: str) -> pd.DataFrame:
    return pd.read_csv(path, comment='#')

def _parse_ci(ci_str: str):
    # expects formats like "[18,247, 19,275]" or "[-18.6, -15.9]"
    if not isinstance(ci_str, str):
        return (None, None)
    m = re.findall(r'-?\d+(?:,\d{3})*(?:\.\d+)?', ci_str)
    if len(m) >= 2:
        def to_num(s):
            return float(s.replace(',', ''))
        return to_num(m[0]), to_num(m[1])
    return (None, None)

def _parse_value(v):
    if isinstance(v, (int, float)):
        return float(v)
    if not isinstance(v, str):
        return None
    # handle e.g. "5.9×"
    v2 = v.replace('×','').strip()
    # percentages like "-17.2%"
    v2 = v2.replace('%','').strip()
    # numbers with commas
    if re.match(r'^-?\d+(?:,\d{3})*(?:\.\d+)?$', v2):
        return float(v2.replace(',', ''))
    return None

def _fmt_value_ci(value, ci):
    if ci and isinstance(ci, str) and ci.strip():
        return f"{value} {ci}"
    return str(value)

def load_metrics(run_dir: str) -> dict:
    tables_dir = os.path.join(run_dir, 'tables')
    unc_path = os.path.join(tables_dir, 'table3_uncertainty.csv')
    pol_path = os.path.join(tables_dir, 'table_policy_metrics.csv')

    if not os.path.exists(unc_path):
        raise FileNotFoundError(unc_path)
    if not os.path.exists(pol_path):
        raise FileNotFoundError(pol_path)

    unc = _read_csv(unc_path)
    pol = _read_csv(pol_path)

    out = {}
    # Predictive uncertainty metrics
    for _, r in unc.iterrows():
        metric = str(r['Metric']).strip()
        out[metric] = {
            'value_raw': r.get('Estimate', ''),
            'ci_raw': r.get('95% CI', ''),
            'value_num': _parse_value(r.get('Estimate', '')),
            'ci_low': _parse_ci(r.get('95% CI', ''))[0],
            'ci_high': _parse_ci(r.get('95% CI', ''))[1],
        }

    # Policy metrics
    for _, r in pol.iterrows():
        metric = str(r['Metric']).strip()
        out[metric] = {
            'value_raw': r.get('Value', ''),
            'ci_raw': r.get('95% CI', ''),
            'value_num': _parse_value(r.get('Value', '')),
            'ci_low': _parse_ci(r.get('95% CI', ''))[0],
            'ci_high': _parse_ci(r.get('95% CI', ''))[1],
        }

    return out

def ci_overlap(a_low, a_high, b_low, b_high):
    if any(x is None for x in [a_low, a_high, b_low, b_high]):
        return None
    return max(a_low, b_low) <= min(a_high, b_high)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--direct', required=True, help='Output dir for covid_control_mode=direct')
    ap.add_argument('--proxy', required=True, help='Output dir for covid_control_mode=proxy')
    ap.add_argument('--none', required=True, help='Output dir for covid_control_mode=none')
    ap.add_argument('--out', required=True, help='Output directory for compiled table')
    args = ap.parse_args()

    runs = {
        'direct': load_metrics(args.direct),
        'proxy': load_metrics(args.proxy),
        'none': load_metrics(args.none),
    }

    # Choose the core metrics reviewers care about (edit freely)
    metric_order = [
        'wRMSE', 'wR²', 'wBias',
        'Precision@10%', 'Recall@10%', 'F1@10%',
        'Lift@10%', 'NDCG', 'Top-10% Underprediction'
    ]

    rows = []
    base = runs['direct']
    for m in metric_order:
        if m not in base:
            continue

        r = {'Metric': m}

        for mode in ['direct','proxy','none']:
            v = runs[mode].get(m, {})
            r[mode] = _fmt_value_ci(v.get('value_raw',''), v.get('ci_raw',''))

        # Delta vs direct (numeric when parseable)
        for mode in ['proxy','none']:
            v0 = base[m].get('value_num', None)
            v1 = runs[mode].get(m, {}).get('value_num', None)
            if v0 is not None and v1 is not None:
                r[f'Δ({mode}-direct)'] = v1 - v0
            else:
                r[f'Δ({mode}-direct)'] = ''

            # CI overlap flag
            ov = ci_overlap(
                base[m].get('ci_low'), base[m].get('ci_high'),
                runs[mode].get(m, {}).get('ci_low'), runs[mode].get(m, {}).get('ci_high')
            )
            r[f'CI_overlap({mode})'] = ov

        rows.append(r)

    out_df = pd.DataFrame(rows)

    out_tables = os.path.join(args.out, 'tables')
    os.makedirs(out_tables, exist_ok=True)
    out_path = os.path.join(out_tables, 'table_covid_sensitivity_summary.csv')
    out_df.to_csv(out_path, index=False, encoding='utf-8-sig')

    print(f"Wrote: {out_path}")

if __name__ == '__main__':
    main()
