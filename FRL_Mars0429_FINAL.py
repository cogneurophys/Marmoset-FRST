# -*- coding: utf-8 -*-
"""
Created on YYYY-MM-DD by Marium

Continuous Feature RL (FRL) implementation updated to draw stimuli exactly as CaFRL/SaFRL:
— on each trial, pick one correct and one incorrect compound stimulus from all colour×shape combos,
  based solely on the current rule label.
Parameters per subject: [alpha (feature LR), beta (inverse temperature)].
Functions:
  - simulate: generates synthetic two-option trials, updates feature weights via FRL.
  - llIED: computes negative log-likelihood for ML fitting on that same FRL.
  - fit: maximises likelihood by minimising llIED.
"""

import numpy as np
import pandas as pd


def simulate(params, trials_csv, rule_mapping, subject, shapes, colours, rng=None, seed=42):
    """
    Simulate stage-like continuous FRL using model-driven rule changes.
    Each rule (feature) runs until criterion (8/10 correct), then moves to next.

    params : [alpha, beta]
    trials_csv : CSV with 'rule' column indicating the experimental rule order (with repeats)
    rule_mapping : dict mapping rule codes to feature strings
    subject, shapes, colours : metadata
    rng, seed : RNG controls

    Returns dict with keys 'stimuli','choice','reward','rule','WH1','WH2','PE','alpha','beta','wh0','subject'
    """
    import numpy as _np, pandas as _pd
    from IED_Mars0429 import flip_one_stim
    # RNG
    rng = rng if rng else _np.random.default_rng(seed)
    # Read experimental rule order
    df = _pd.read_csv(trials_csv)
    raw_rules = df['rule'].tolist()
    # Derive rule sequence (unique changes)
    rule_sequence = []
    prev = None
    for code in raw_rules:
        if code != prev:
            rule_sequence.append(code)
            prev = code
    # All possible stimuli
    stimuli_all = [f"{c} {s}" for c in colours for s in shapes]
    # Unpack parameters
    alpha, beta = params
    # Initialize weights
    wh_col = _np.zeros(len(colours)); wh_shp = _np.zeros(len(shapes))
    wh0 = [wh_col.copy(), wh_shp.copy()]
    # Storage
    stimuli, choice, reward, rule_list = [], [], [], []
    WH1, WH2, PE = [], [], []
    # Encodings
    colour_enc = {c: _np.eye(len(colours))[i] for i,c in enumerate(colours)}
    shape_enc = {s: _np.eye(len(shapes))[j] for j,s in enumerate(shapes)}
    # Run through each rule stage
    for feat_code in rule_sequence:
        feat = rule_mapping[feat_code]
        # track correct count history
        outcomes = []
        while True:
            # generate one correct & one incorrect stimulus
            corr = rng.choice([st for st in stimuli_all if feat in st])
            inc = rng.choice([st for st in stimuli_all if feat not in st])
            pair = [corr, inc] if rng.random() < 0.5 else [inc, corr]
            stimuli.append(pair)
            rule_list.append(feat_code)
            # encode features
            col1, shp1 = pair[0].split(); col2, shp2 = pair[1].split()
            c1, s1 = colour_enc[col1], shape_enc[shp1]
            c2, s2 = colour_enc[col2], shape_enc[shp2]
            # values
            V1 = wh_col @ c1 + wh_shp @ s1; V2 = wh_col @ c2 + wh_shp @ s2
            V = _np.array([V1, V2])
            # softmax
            expV = _np.exp(beta*(V - _np.max(V)))
            probs = expV/expV.sum()
            idx = 0 if rng.random() < probs[0] else 1
            ch = pair[idx]; choice.append(ch)
            # reward
            rew = 1 if feat in ch else -1; reward.append(rew)
            # record outcome for criterion
            outcomes.append(1 if rew>0 else 0)
            # PE
            pe = rew - V[idx]; PE.append(pe)
            # FRL update
            R_trial = [None, None]; R_trial[idx] = rew; R_trial[1-idx] = -rew
            dV1 = R_trial[0] - V[0]; dV2 = R_trial[1] - V[1]
            wh_col += alpha*(dV1*c1 + dV2*c2)
            wh_shp += alpha*(dV1*s1 + dV2*s2)
            WH1.append(wh_col.copy()); WH2.append(wh_shp.copy())
            # check criterion: 8 correct in last 10
            if len(outcomes) >= 10 and sum(outcomes[-10:]) >= 8:
                break
    return {
        'stimuli': stimuli,
        'choice': choice,
        'reward': reward,
        'rule': rule_list,
        'WH1': WH1,
        'WH2': WH2,
        'PE': PE,
        'alpha': alpha,
        'beta': beta,
        'wh0': wh0,
        'subject': subject
    }


def llIED(params, R, choice, S, rule=None, likelihood=False, u=None, v2=None):
    """Negative log-likelihood for FRL on synthetic FRL stream."""
    import numpy as _np
    from scipy.special import logsumexp
    # Transform raw params
    a = _np.asarray(params).reshape(2,1)
    alpha = 1/(1+_np.exp(-a[0,0]))
    beta = _np.exp(a[1,0])

    # Prepare encodings
    cols = sorted({st.split()[0] for pair in S for st in pair})
    shps = sorted({st.split()[1] for pair in S for st in pair})
    colour_enc = {c: _np.eye(len(cols))[i] for i, c in enumerate(cols)}
    shape_enc = {s: _np.eye(len(shps))[j] for j, s in enumerate(shps)}

    # Initialize weights
    wh_col = _np.zeros(len(cols))
    wh_shp = _np.zeros(len(shps))

    ll = 0.0
    like_vals = []

    for t in range(len(choice)):
        # encode features
        col1, shp1 = S[t][0].split()
        col2, shp2 = S[t][1].split()
        c1, s1 = colour_enc[col1], shape_enc[shp1]
        c2, s2 = colour_enc[col2], shape_enc[shp2]

        # values
        v1 = wh_col @ c1 + wh_shp @ s1
        v2 = wh_col @ c2 + wh_shp @ s2
        V = _np.array([v1, v2])
        idx = [S[t][0], S[t][1]].index(choice[t])

        # log-softmax for log-likelihood
        logp = beta * V[idx] - logsumexp(beta * V)
        ll += logp
        if likelihood:
            like_vals.append(_np.exp(logp))

        # reward vector
        rew = R[t]
        R_trial = [None, None]
        R_trial[idx] = rew
        R_trial[1-idx] = -rew

        # compute prediction errors
        dV1 = R_trial[0] - V[0]
        dV2 = R_trial[1] - V[1]

        # update weights
        wh_col += alpha * (dV1 * c1 + dV2 * c2)
        wh_shp += alpha * (dV1 * s1 + dV2 * s2)

    if likelihood:
        return (_np.prod(like_vals), _np.mean(like_vals))
    # negative sum of log probs
    return -ll


def fit(params):
    """Use ML to fit FRL by minimizing llIED."""
    import scipy.optimize, warnings
    warnings.simplefilter('ignore', RuntimeWarning)
    args = params[:-1]
    m0 = params[-1]
    res = scipy.optimize.minimize(fun=llIED, x0=list(m0), args=tuple(args))
    warnings.resetwarnings()
    return list(res.x)
