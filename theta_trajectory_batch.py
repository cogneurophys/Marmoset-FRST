#!/usr/bin/env python3
"""
Batch attention-trajectory analysis for the CaFRL marmoset sessions.--Liya Ma

For every session CSV under RAW_ROOT it:
  1. reconstructs the within-session attention trajectory sigma(theta_t) by replaying
     the CaFRL forward model on the animal's real choices (validated to reproduce
     CaFRL_Mars0429.llIED's NLL exactly),
  2. aligns attention-to-the-relevant-dimension to each DIMENSIONAL switch and
     accumulates a switch-triggered average (overall and per animal x feature set),
  3. runs the key model test: does letting theta UPDATE within a session fit better
     than holding it at a fixed bias?  It ML-fits both a dynamic-theta and a
     static-theta CaFRL per session and compares BIC.

Outputs (written next to this script / to OUTDIR):
  - per_session_results.csv        one row per session (params, NLLs, dBIC, switch metrics)
  - switch_triggered_average.csv   attention-to-relevant vs trials-since-switch (overall + per cell)
  - switch_triggered_attention.png overall switch-triggered average figure

CONVENTIONS (from CaFRL_Mars0429.py):
  - value V = sigmoid(theta)*colour_value + (1-sigmoid(theta))*shape_value
    so sigmoid(theta) is the weight on COLOUR (dimension 1 = first word of each stimulus).
  - rule codes 1-3 => shape is the relevant dimension; 4-6 => colour.
  - reward is mapped 0/1 -> -1/+1 (as in IED_Mars0429.load_trialdata).
  - ML parameter transforms match llIED: alpha=sigmoid(raw), beta=exp(raw), theta0=raw.

This script is self-contained (it re-implements the forward pass faithfully) so it does
NOT import the model modules, but if CaFRL_Mars0429 is importable it will cross-check the
dynamic NLL on the first session and warn if they differ.
"""

import os, re, glob, warnings, sys
import numpy as np
import pandas as pd
from scipy.optimize import minimize

# ============================== CONFIG ==============================
RAW_ROOT      = r"C:\YOUR_PATH\CleanedCSVs"    # <-- folder containing the 8 session folders
PERSESSION_CSV= r"CaFRL_EM_persession.csv"        # <-- EM per-session fits (for the reported trajectory)
MODEL_DIR     = r"C:\YOUR_PATH\RL_codes"               # <-- folder with CaFRL_Mars0429.py etc. (for the optional faithfulness check); "" = don't add
OUTDIR        = r"C:\YOUR_OUTPUT_PATH\"                        # where to write outputs

# make the model modules importable for the optional faithfulness cross-check
for _p in (MODEL_DIR, RAW_ROOT, os.getcwd()):
    if _p and os.path.isdir(_p) and _p not in sys.path:
        sys.path.insert(0, os.path.abspath(_p))
WIN_PRE, WIN_POST = 5, 20                        # switch-triggered window (trials before / after)
N_RESTARTS    = 6                                # random restarts for each ML fit
RNG           = np.random.default_rng(42)
# ===================================================================

sig = lambda x: 1.0/(1.0+np.exp(-x))

# ----------------------------- parsing -----------------------------
def parse_meta(path):
    name = os.path.basename(path)
    stem = name[:-4] if name.lower().endswith('.csv') else name
    a  = re.search(r'[_ ]([A-Za-z])[_ ]WCST', stem, re.I)
    animal = a.group(1).upper() if a else '??'
    fs = 'New' if re.search(r'Step8New', stem, re.I) else ('Original' if re.search(r'Step8', stem, re.I) else '??')
    ts = re.search(r'\d{8}_\d{6}', stem)
    session = ts.group(0) if ts else stem
    return animal, fs, session

def load_session(csv):
    """Return R(+/-1), choice(str), S(list of [s1,s2]), rule(list of int). Assumes 'colour shape' order."""
    df = pd.read_csv(csv)
    need = {'stimulus_1','stimulus_2','choice','reward','rule'}
    if not need.issubset({c.lower() for c in df.columns}):
        raise ValueError(f"missing columns in {os.path.basename(csv)}")
    col = {c.lower(): c for c in df.columns}
    R      = df[col['reward']].map({0:-1, 1:1}).tolist()
    choice = df[col['choice']].astype(str).tolist()
    S      = [[str(a), str(b)] for a, b in zip(df[col['stimulus_1']], df[col['stimulus_2']])]
    rule   = [int(r) for r in df[col['rule']].tolist()]
    return R, choice, S, rule

# --------------------- faithful forward replay ---------------------
def forward(params_native, R, choice, S, static=False, return_traj=False):
    """
    Replay CaFRL on real choices with NATIVE params [alpha, beta, theta0].
    static=True freezes theta at theta0 (feature weights still learn).
    Returns NLL, or (NLL, theta_trajectory) if return_traj.
    Matches CaFRL_Mars0429.llIED exactly for static=False.
    """
    alpha, beta, theta0 = params_native
    cols   = sorted({s.split()[0] for pr in S for s in pr})
    shapes = sorted({s.split()[1] for pr in S for s in pr})
    ce = {c: np.eye(len(cols))[i].reshape(1,-1)   for i,c in enumerate(cols)}
    se = {s: np.eye(len(shapes))[j].reshape(1,-1) for j,s in enumerate(shapes)}
    wh_col = np.zeros((1,len(cols))); wh_shp = np.zeros((1,len(shapes)))
    theta = float(theta0)
    ll = 0.0; traj = []
    for t in range(len(choice)):
        s1, s2 = S[t]; f1 = s1.split(); f2 = s2.split()
        col1, shp1, col2, shp2 = ce[f1[0]], se[f1[1]], ce[f2[0]], se[f2[1]]
        ah1 = (wh_col@col1.T).item(); ah2 = (wh_shp@shp1.T).item()
        ah3 = (wh_col@col2.T).item(); ah4 = (wh_shp@shp2.T).item()
        st = sig(theta)
        V1 = st*ah1 + (1-st)*ah2; V2 = st*ah3 + (1-st)*ah4
        Vt = np.array([V1, V2]); vmax = beta*np.amax(Vt)
        if choice[t] not in (s1, s2):      # choice must be one of the two stimuli
            return (np.inf, np.array(traj)) if return_traj else np.inf
        idx = [s1, s2].index(choice[t])
        ll += beta*(Vt[idx]-vmax) - np.log(np.sum(np.exp(beta*(Vt-vmax))))
        if return_traj: traj.append(theta)
        Rt = [None, None]; Rt[idx] = R[t]; Rt[1-idx] = -R[t]
        dV1 = V1-Rt[0]; dV2 = V2-Rt[1]
        wh_col = wh_col - alpha*(st*dV1*col1 + st*dV2*col2)
        wh_shp = wh_shp - alpha*((1-st)*dV1*shp1 + (1-st)*dV2*shp2)
        if not static:
            ahc = [[ah1,ah2],[ah3,ah4]][idx]
            dcostDV = Vt[idx]-Rt[idx]
            dV_dth  = st*(1-st)*(ahc[0]-ahc[1])
            theta   = theta - alpha*(dcostDV*dV_dth)
    nll = -float(ll)
    return (nll, np.array(traj)) if return_traj else nll

# --------------------------- ML fitting ----------------------------
def _unpack(raw):
    return np.array([sig(raw[0]), np.exp(raw[1]), raw[2]])   # native alpha,beta,theta0

def ml_fit(R, choice, S, static=False, init_native=None):
    """ML-fit [alpha,beta,theta0] (llIED transforms). Returns (native_params, nll)."""
    def obj(raw):
        return forward(_unpack(raw), R, choice, S, static=static)
    best = None
    seeds = []
    if init_native is not None:
        a,b,t = init_native
        a = min(max(a,1e-4),1-1e-4)
        seeds.append([np.log(a/(1-a)), np.log(max(b,1e-4)), t])
    for _ in range(N_RESTARTS):
        seeds.append([RNG.normal(2,1), RNG.normal(-1,1), RNG.normal(0,2)])  # raw space
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        for x0 in seeds:
            try:
                r = minimize(obj, x0, method='Nelder-Mead',
                             options={'xatol':1e-4,'fatol':1e-4,'maxiter':4000})
                if np.isfinite(r.fun) and (best is None or r.fun < best.fun):
                    best = r
            except Exception:
                continue
    if best is None:
        return None, np.inf
    return _unpack(best.x), float(best.fun)

# -------------------- switch-triggered extraction ------------------
def relevant_dim(rule):
    return np.array(['shape' if r in (1,2,3) else 'colour' for r in rule])

def switch_aligned(attn_rel, reldim):
    """Return list of (offset, value) around each DIMENSIONAL switch."""
    sw = [i for i in range(1, len(reldim)) if reldim[i] != reldim[i-1]]
    out = []
    for s in sw:
        for off in range(-WIN_PRE, WIN_POST+1):
            j = s + off
            if 0 <= j < len(attn_rel):
                out.append((off, attn_rel[j]))
    return out, len(sw)

# ------------------------------ main -------------------------------
def main():
    files = sorted(glob.glob(os.path.join(RAW_ROOT, '**', '*.csv'), recursive=True))
    # drop obvious non-session csvs (results files)
    files = [f for f in files if 'WCST' in os.path.basename(f)]
    if not files:
        raise SystemExit(f"No session CSVs (…WCST…) found under {RAW_ROOT}")

    # optional EM fits for the reported trajectory
    emfit = {}
    if os.path.exists(os.path.join(RAW_ROOT, PERSESSION_CSV)) or os.path.exists(PERSESSION_CSV):
        p = PERSESSION_CSV if os.path.exists(PERSESSION_CSV) else os.path.join(RAW_ROOT, PERSESSION_CSV)
        em = pd.read_csv(p)
        for _, r in em.iterrows():
            key = re.search(r'\d{8}_\d{6}', str(r['session']))
            if key:
                emfit[(str(r['animal']).upper(), str(r['featureset']), key.group(0))] = (r['alpha'], r['beta'], r['theta0'])

    # cross-check faithfulness on first loadable session
    checked = False

    rows = []
    swpts_all = []                       # (offset, value)
    swpts_cell = {}                      # (animal,fs) -> list
    for f in files:
        animal, fs, session = parse_meta(f)
        try:
            R, choice, S, rule = load_session(f)
        except Exception as e:
            print(f"  skip {os.path.basename(f)}: {e}"); continue
        if len(choice) < 10:
            continue

        # ---- reported trajectory: use EM fit if available, else ML dynamic ----
        native = emfit.get((animal, fs, session))
        if native is None:
            native, _ = ml_fit(R, choice, S, static=False)
        if native is None:
            continue
        nll_dyn_report, traj = forward(native, R, choice, S, static=False, return_traj=True)

        if not checked:
            try:
                import CaFRL_Mars0429 as C
                a,b,t = native
                raw = [np.log(a/(1-a)), np.log(b), t]
                ref = float(np.asarray(C.llIED(raw, R, choice, S, rule=rule, likelihood=False)).ravel()[0])
                print(f"[faithfulness check] llIED={ref:.4f}  replay={nll_dyn_report:.4f}  diff={abs(ref-nll_dyn_report):.4f}")
            except Exception as e:
                print(f"[faithfulness check skipped: {e}]")
            checked = True

        reldim = relevant_dim(rule)
        s_col  = sig(traj)
        attn_rel = np.where(reldim == 'colour', s_col, 1-s_col)
        pts, nsw = switch_aligned(attn_rel, reldim)
        swpts_all += pts
        swpts_cell.setdefault((animal, fs), []).extend(pts)

        # post-switch recovery metric (mean attn_rel 1..10 trials after each dim switch)
        sw = [i for i in range(1, len(reldim)) if reldim[i] != reldim[i-1]]
        rec = np.mean([np.mean(attn_rel[s:s+10]) for s in sw]) if sw else np.nan

        # ---- static vs dynamic BIC (ML-fit BOTH for a fair comparison) ----
        pd_dyn, nll_dyn = ml_fit(R, choice, S, static=False, init_native=native)
        pd_sta, nll_sta = ml_fit(R, choice, S, static=True,  init_native=native)
        T = len(choice); k = 3
        bic_dyn = 2*nll_dyn + k*np.log(T)
        bic_sta = 2*nll_sta + k*np.log(T)
        dBIC = bic_sta - bic_dyn        # >0 => dynamic (updating theta) preferred

        rows.append(dict(animal=animal, featureset=fs, session=session, n_trials=T,
                         n_dim_switches=nsw,
                         alpha=native[0], beta=native[1], theta0=native[2],
                         mean_attn_rel=float(np.mean(attn_rel)),
                         post_switch_recovery=rec,
                         nll_dynamic=nll_dyn, nll_static=nll_sta,
                         bic_dynamic=bic_dyn, bic_static=bic_sta, dBIC=dBIC,
                         favours_dynamic=bool(dBIC > 0)))

    res = pd.DataFrame(rows)
    res.to_csv(os.path.join(OUTDIR, 'per_session_results.csv'), index=False)

    # ---- switch-triggered average table ----
    def sta_table(pts, label):
        d = pd.DataFrame(pts, columns=['offset','val'])
        g = d.groupby('offset')['val']
        return pd.DataFrame({'group':label,'offset':g.mean().index,
                             'mean':g.mean().values,'sem':(g.std()/np.sqrt(g.count())).values,
                             'n':g.count().values})
    tabs = [sta_table(swpts_all, 'ALL')]
    for cell, pts in sorted(swpts_cell.items()):
        if pts: tabs.append(sta_table(pts, f'{cell[0]}_{cell[1]}'))
    sta = pd.concat(tabs, ignore_index=True)
    sta.to_csv(os.path.join(OUTDIR, 'switch_triggered_average.csv'), index=False)

    # ---- figure: overall switch-triggered average ----
    try:
        import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
        a = sta[sta.group=='ALL'].sort_values('offset')
        fig, ax = plt.subplots(figsize=(8,4.5))
        ax.axvspan(-WIN_PRE-0.5, -0.5, color='#eee', alpha=.6)
        ax.axhline(0.5, color='k', ls=':', lw=.8)
        ax.axvline(0, color='k', ls='--', lw=1.2)
        ax.plot(a.offset, a['mean'], color='#2b7a4b', lw=2)
        ax.fill_between(a.offset, a['mean']-a['sem'], a['mean']+a['sem'], color='#2b7a4b', alpha=.25)
        ax.set_xlabel('trials since dimensional switch'); ax.set_ylabel('attention to the relevant dimension')
        ax.set_title('Switch-triggered attention realignment (all sessions)')
        ax.set_ylim(0,1)
        plt.tight_layout(); plt.savefig(os.path.join(OUTDIR,'switch_triggered_attention.png'), dpi=140)
        print("saved switch_triggered_attention.png")
    except Exception as e:
        print(f"[figure skipped: {e}]")

    # ---- console summary ----
    print(f"\nsessions analysed: {len(res)}")
    if len(res):
        print("\nStatic-theta vs dynamic-theta (dBIC>0 favours DYNAMIC updating):")
        print(f"  overall: {res.favours_dynamic.mean()*100:.0f}% of sessions favour dynamic | "
              f"median dBIC = {res.dBIC.median():.1f}")
        cell = res.groupby(['animal','featureset']).agg(
                 n=('session','size'), pct_dynamic=('favours_dynamic','mean'),
                 median_dBIC=('dBIC','median')).reset_index()
        cell['pct_dynamic'] = (cell['pct_dynamic']*100).round(0)
        print(cell.to_string(index=False))
        a = sta[sta.group=='ALL'].sort_values('offset')
        pre = a[a.offset<0]['mean'].mean(); at0 = a[a.offset==0]['mean'].values
        post = a[(a.offset>0)&(a.offset<=10)]['mean'].mean()
        print(f"\nswitch-triggered attention-to-relevant: pre={pre:.2f} | at switch={at0[0]:.2f} | 1-10 after={post:.2f}")
    print("\nwrote per_session_results.csv and switch_triggered_average.csv")

if __name__ == '__main__':
    main()
