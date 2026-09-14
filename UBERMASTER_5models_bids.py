#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Uber-master pipeline for running FRL, CaFRL, and SaFRL over all session files in a folder.
For each CSV session and each model, fits the model to real data, simulates with fitted parameters,
per-session parameter recovery, plots trajectories into session-specific subfolders, and aggregates
summary (including recovery stats) into one Excel per model.

INPUT CHANGE (BIDS):
    Data now come from the BIDS dataset you deposit on Zenodo (one .tsv per
    session under sub-<id>/ses-<date>/beh/). The model helpers
    (modelling_Mars0429, IED_Mars0429) still read plain CSV, so each TSV is
    STAGED to a temporary CSV up front -- tab->comma, 'n/a'->empty, and the
    BIDS-only 'feature_set' column dropped so the columns match the old inputs.
    Nothing in the per-model modules or the helpers needs to change.
    Process ONE subject and ONE feature set (acq-) per run, as before.
"""
import os
import re
import shutil
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import modelling_Mars0429
import FRL_Mars0429_FINAL as FRL
import CaFRL_Mars0429
import SaFRL_Mars0429
import PFRL_3params           # [PFRL +] perseveration control, 3-param last-trial (count-matched to CaFRL)
import PFRL_4params           # [PFRL +] perseveration control, 4-param decaying choice kernel
import IED_Mars0429
from scipy.stats import pearsonr


def stage_bids_sessions(bids_root, subject, feature_set, task, staging_dir):
    """Find one subject's acq-<feature_set> TSV sessions in a BIDS tree and
    write each as the plain CSV the model helpers expect.

    Returns a list of (session_label, csv_path). session_label is the 8-digit
    session date parsed from the filename (falls back to the file stem).
    """
    subj_dir = Path(bids_root) / f"sub-{subject}"
    pattern  = f"ses-*/beh/sub-{subject}_ses-*_task-{task}_acq-{feature_set}_beh.tsv"
    tsv_files = sorted(subj_dir.glob(pattern))

    staged, date_re = [], re.compile(r"ses-(\d{8})")
    for tsv in tsv_files:
        df = pd.read_csv(tsv, sep="\t", na_values="n/a")
        df = df.drop(columns=[c for c in ("feature_set",) if c in df.columns])
        m = date_re.search(tsv.name)
        session = m.group(1) if m else tsv.stem
        csv_path = Path(staging_dir) / f"{session}.csv"
        df.to_csv(csv_path, index=False)     # comma-separated, NaN -> empty
        staged.append((session, str(csv_path)))
    return staged


def main():
    # --------------------------
    # 1. User-defined settings
    # --------------------------
    # Point at the BIDS dataset root, and pick ONE subject + ONE feature set.
    BIDS_ROOT   = r"C:\YOUR_PATH\data"
    SUBJECT     = "k"          # BIDS sub- label (no 'sub-' prefix)
    FEATURE_SET = "original"   # BIDS acq- label: 'original' or 'new'
    TASK        = "frst"
    OUTPUT_ROOT = r"C:\YOUR_PATH\results"
    os.makedirs(OUTPUT_ROOT, exist_ok=True)

    SUBJECT_ID   = f"Marmoset_{SUBJECT}"
    SHAPES       = ['heart', 'star']
    COLOURS      = ['blue', 'yellow']
    #RULE_MAPPING = {1:'pacman',2:'stapler',3:'pentagon',4:'purple',5:'orange',6:'cyan'}
    #RULE_MAPPING = {1:'yellow',2:'red',3:'blue',4:'star',5:'square',6:'heart'}
    RULE_MAPPING = {1:'star',2:'square',3:'heart ',4:'yellow',5:'red',6:'blue'}
    MODEL_LIST    = ['FRL','CaFRL','SaFRL','PFRL3','PFRL4']   # [PFRL +] both PFRL variants

    # ---- Discover + stage this subject's TSV sessions (once, reused by all models) ----
    staging_dir = Path(tempfile.mkdtemp(prefix="bids_tsv2csv_"))
    staged_sessions = stage_bids_sessions(BIDS_ROOT, SUBJECT, FEATURE_SET, TASK, staging_dir)
    if not staged_sessions:
        shutil.rmtree(staging_dir, ignore_errors=True)
        raise SystemExit(
            f"No TSVs found for sub-{SUBJECT} / acq-{FEATURE_SET} / task-{TASK} "
            f"under {BIDS_ROOT}. Check SUBJECT, FEATURE_SET, and BIDS_ROOT.")
    print(f"Staged {len(staged_sessions)} session(s) for sub-{SUBJECT} / acq-{FEATURE_SET}")

    try:
        for MODEL_NAME in MODEL_LIST:
            print(f"Running model: {MODEL_NAME}")
            # select functions and params
            if MODEL_NAME == 'FRL':
                simulate_func = FRL.simulate
                fit_func      = FRL.fit
                ll_func       = FRL.llIED
                k_params      = 2
                transforms    = ['sigmoid','exp']
            elif MODEL_NAME == 'CaFRL':
                simulate_func = CaFRL_Mars0429.simulate
                fit_func      = CaFRL_Mars0429.fit
                ll_func       = CaFRL_Mars0429.llIED
                k_params      = 3
                transforms    = ['sigmoid','exp',None]
            elif MODEL_NAME == 'PFRL3':  # [PFRL +] FRL + perseveration, last-trial trace (3 params, count-matched to CaFRL)
                simulate_func = PFRL_3params.simulate
                fit_func      = PFRL_3params.fit
                ll_func       = PFRL_3params.llIED
                k_params      = 3
                transforms    = ['sigmoid','exp',None]
            elif MODEL_NAME == 'PFRL4':  # [PFRL +] FRL + perseveration, decaying choice kernel (4 params)
                simulate_func = PFRL_4params.simulate
                fit_func      = PFRL_4params.fit
                ll_func       = PFRL_4params.llIED
                k_params      = 4
                transforms    = ['sigmoid','exp',None,'sigmoid']
            else:  # SaFRL
                simulate_func = SaFRL_Mars0429.simulate
                fit_func      = SaFRL_Mars0429.fit
                ll_func       = SaFRL_Mars0429.llIED
                k_params      = 4
                transforms    = ['sigmoid','exp',None,'sigmoid']
            fit_args = ['reward','choice','stimuli','rule']

            model_out = os.path.join(OUTPUT_ROOT, MODEL_NAME)
            os.makedirs(model_out, exist_ok=True)

            # summary rows
            rows = []
            # [PFRL +] per-model parameter names so each model's columns are labelled
            #          correctly (PFRL's 3rd/4th params are phi and alpha_p, not theta0/eps)
            param_names_by_model = {
                'FRL':   ['alpha', 'beta'],
                'CaFRL': ['alpha', 'beta', 'theta0'],
                'SaFRL': ['alpha', 'beta', 'theta0', 'eps'],
                'PFRL3': ['alpha', 'beta', 'phi'],
                'PFRL4': ['alpha', 'beta', 'phi', 'alpha_p'],
            }
            names = param_names_by_model[MODEL_NAME]

            for session, trials_csv in staged_sessions:
                print(f"  Session: {session}")
                sess_out = os.path.join(model_out, f"Plots_{session}")
                os.makedirs(sess_out, exist_ok=True)

                # trials_csv already points at the staged CSV for this session
                # 1) per-session recovery
                dist = (np.ones(k_params), np.diag(np.ones(k_params)*2))
                sim_res = modelling_Mars0429.simulate(
                    model=simulate_func,
                    trials_csv=trials_csv,
                    rule_mapping=RULE_MAPPING,
                    shapes=SHAPES,
                    colours=COLOURS,
                    dist=dist,
                    transforms=transforms,
                    N=10,
                    seed=42
                )
                est_sim = modelling_Mars0429.fit(
                    data=sim_res['data'], nP=k_params,
                    fit_func=fit_func, fit_args=fit_args,
                    transforms=transforms, fit='ML', n_jobs=1, seed=42
                )
                # compute correlations
                corr_vals, p_vals = [], []
                for i in range(k_params):
                    sim_i = sim_res['params'][:, i]
                    est_i = est_sim['m'][:, i]
                    mask = np.isfinite(sim_i) & np.isfinite(est_i)
                    if np.sum(mask) > 1:
                        c, p = pearsonr(sim_i[mask], est_i[mask])
                    else:
                        c, p = np.nan, np.nan
                    corr_vals.append(c)
                    p_vals.append(p)

                # 2) load & fit real data
                df_real = IED_Mars0429.load_trialdata(trials_csv)
                df_real['subject'] = SUBJECT_ID
                real_data = IED_Mars0429.trials2dict(df_real)
                real_data = IED_Mars0429.flip_stimuli_in_data(real_data, COLOURS, SHAPES)

                est = modelling_Mars0429.fit(
                    data=real_data, nP=k_params,
                    fit_func=fit_func, fit_args=fit_args,
                    transforms=transforms, fit='ML', n_jobs=1, seed=42
                )
                params_est = est['m'][0]

                # 3) simulate with fitted params
                sim = modelling_Mars0429.simulate(
                    model=simulate_func,
                    trials_csv=trials_csv,
                    rule_mapping=RULE_MAPPING,
                    shapes=SHAPES,
                    colours=COLOURS,
                    subjects=[SUBJECT_ID],
                    params=est['m'], transforms=transforms,
                    rng=None, seed=42
                )
                sim_data = sim['data'][0]
                sim_data = IED_Mars0429.flip_stimuli_in_data([sim_data], COLOURS, SHAPES)[0]

                # 4) plot
                plt_show_orig = plt.show
                plt.show      = lambda *args, **kwargs: None
                IED_Mars0429.plot_idata(sim_data, colours=COLOURS, shapes=SHAPES)
                plt.show = plt_show_orig
                for i, num in enumerate(plt.get_fignums()):
                    plt.figure(num)
                    plt.savefig(os.path.join(sess_out, f"plot_{i}_TESTT.png"), bbox_inches='tight')
                plt.close('all')

                # 5) stats for real fit
                args_ll = [real_data[0]['reward'], real_data[0]['choice'], real_data[0]['stimuli'], real_data[0]['rule'], False]
                nll     = ll_func(params_est, *args_ll)
                logL    = -nll
                Ntr     = len(real_data[0]['choice'])
                BIC     = -2*logL + k_params*np.log(Ntr)
                alpt    = modelling_Mars0429.alpt(data=real_data, like_func=ll_func, fit_args=fit_args, params=est['m'], n_jobs=1)
                avg_like   = alpt['alpt']
                total_like = alpt['subject_alpts'][0] * Ntr

                # build summary row
                row = {'session': session}
                for nm, val, c, p in zip(names, params_est, corr_vals, p_vals):
                    row[f'est_{nm}']  = val
                    row[f'corr_{nm}'] = c
                    row[f'p_{nm}']    = p
                row.update({'BIC': BIC, 'avg_like': avg_like, 'total_like': total_like})
                rows.append(row)

            # save summary
            summary_df   = pd.DataFrame(rows)
            summary_file = os.path.join(model_out, f"summary_{SUBJECT_ID}_TESTCHANGE.xlsx")
            summary_df.to_excel(summary_file, index=False)
            print(f"  Saved summary to {summary_file}")
    finally:
        # clean up the staged CSVs
        shutil.rmtree(staging_dir, ignore_errors=True)


if __name__ == '__main__':
    main()
