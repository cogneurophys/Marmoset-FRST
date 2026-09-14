# -*- coding: utf-8 -*-
"""
PFRL_4params.py  —  Feature-RL + feature-level CHOICE PERSEVERATION
                     (decaying choice-kernel form; 4 parameters)
====================================================================

Control model for CaFRL. This file is FRL_Mars0429_FINAL.py with ONE mechanism
added: a feature-level choice-perseveration term that enters the DECISION
VARIABLE (never the value update). Everything tagged

    # [PFRL +]

is new relative to FRL. Everything else is copied verbatim from FRL, so the
value-learning core is byte-for-byte identical to the feature-only model.

Perseveration as a decaying choice kernel
-----------------------------------------
A feature-level trace p_t(f) is updated by a delta rule with its OWN rate
alpha_p (this is the defining feature of perseveration: it tracks CHOICES, not
rewards, and it never touches the value weights):

        p_{t+1}(f) = p_t(f) + alpha_p * ( 1[f in chosen_t] - p_t(f) )

The trace enters the softmax as a separate additive term weighted by phi:

        P_t(o) ∝ exp(  beta * sum_{f in o} V_t(f)  +  phi * sum_{f in o} p_t(f)  )

Parameters and transforms
-------------------------
    params     : [alpha, beta, phi, alpha_p]            (4 params)
    transforms : ['sigmoid', 'exp', None, 'sigmoid']
        alpha   = sigmoid(raw[0])      feature learning rate (value update)
        beta    = exp(raw[1])          inverse temperature on value
        phi     = raw[2]               perseveration weight   (used raw, like CaFRL theta0)
        alpha_p = sigmoid(raw[3])      choice-kernel update rate (0..1)

Note on model comparison
------------------------
CaFRL has 3 parameters; this decaying-kernel PFRL has 4, so BIC penalises it by
one extra parameter (k*ln N). It is therefore NOT exactly count-matched to
CaFRL. Interpretation still holds, just asymmetrically: if CaFRL wins on BIC it
does so despite having fewer parameters AND facing a more flexible competitor
(strong evidence for the dimensional account); if this PFRL wins, it does so
despite the heavier penalty (strong evidence for perseveration).

For the exactly count-matched test (3 vs 3), use the LAST-TRIAL trace variant,
provided separately as PFRL_3params.py (params [alpha, beta, phi]). Run both in
UBERMASTER to compare all five models (FRL, CaFRL, SaFRL, PFRL_3params,
PFRL_4params) by BIC.

Two specification points that keep the test fair:
  1. The trace is FEATURE-LEVEL (one weight per colour, one per shape), not
     stimulus- or response-level. Only the feature-level form can mimic the
     within- vs cross-dimension signature, because that signature depends on
     features being shared across options.
  2. phi is a SEPARATE additive weight, NOT folded under beta. A shared
     temperature would force value and perseveration to the same scale and
     quietly handicap PFRL, biasing the comparison toward CaFRL.
"""

import numpy as np
import pandas as pd


def simulate(params, trials_csv, rule_mapping, subject, shapes, colours, rng=None, seed=42):
    """
    Simulate FRL + decaying-kernel perseveration for one subject (criterion-based
    stages, identical to FRL.simulate).

    params : [alpha, beta, phi, alpha_p]     # [PFRL +] phi, alpha_p added
    Returns dict with FRL's keys plus 'phi', 'alpha_p', and 'PERS'.
    """
    import numpy as _np, pandas as _pd
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

    # [PFRL +] unpack FOUR native parameters (FRL had: alpha, beta = params)
    alpha, beta, phi, alpha_p = params

    # Initialize weights
    wh_col = _np.zeros(len(colours)); wh_shp = _np.zeros(len(shapes))
    wh0 = [wh_col.copy(), wh_shp.copy()]

    # [PFRL +] feature-level perseveration (choice-kernel) traces: one weight per
    #          colour, one per shape. Initialised to zero -> no bias on trial 1.
    pers_col = _np.zeros(len(colours)); pers_shp = _np.zeros(len(shapes))

    # Storage
    stimuli, choice, reward, rule_list = [], [], [], []
    WH1, WH2, PE = [], [], []
    PERS = []  # [PFRL +] perseveration-trace history (extra key; ignored by plotting)
    # Encodings
    colour_enc = {c: _np.eye(len(colours))[i] for i,c in enumerate(colours)}
    shape_enc = {s: _np.eye(len(shapes))[j] for j,s in enumerate(shapes)}
    # Run through each rule stage
    for feat_code in rule_sequence:
        feat = rule_mapping[feat_code]
        # track correct count history
        outcomes = []
        guard = 0  # [PFRL +] safety counter to prevent an infinite criterion loop
        while True:
            guard += 1
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

            # [PFRL +] feature-level perseveration value of each option
            P1 = pers_col @ c1 + pers_shp @ s1
            P2 = pers_col @ c2 + pers_shp @ s2
            # [PFRL +] decision variable: beta scales VALUE, phi is a SEPARATE additive
            #          weight on PERSEVERATION (NOT folded under beta).
            DV = _np.array([beta*V1 + phi*P1, beta*V2 + phi*P2])
            # softmax over the decision variable (beta is already inside DV)
            expV = _np.exp(DV - _np.max(DV))
            probs = expV/expV.sum()
            idx = 0 if rng.random() < probs[0] else 1
            ch = pair[idx]; choice.append(ch)
            # reward
            rew = 1 if feat in ch else -1; reward.append(rew)
            # record outcome for criterion
            outcomes.append(1 if rew>0 else 0)
            # PE
            pe = rew - V[idx]; PE.append(pe)
            # ---- FRL value-learning core: UNCHANGED (rewards only, never the trace) ----
            R_trial = [None, None]; R_trial[idx] = rew; R_trial[1-idx] = -rew
            dV1 = R_trial[0] - V[0]; dV2 = R_trial[1] - V[1]
            wh_col += alpha*(dV1*c1 + dV2*c2)
            wh_shp += alpha*(dV1*s1 + dV2*s2)
            WH1.append(wh_col.copy()); WH2.append(wh_shp.copy())

            # [PFRL +] DECAYING CHOICE-KERNEL update of the perseveration trace:
            #          p <- p + alpha_p * ( 1[f in chosen] - p ), feature-wise.
            #          The chosen option's colour/shape entries move toward 1; all
            #          others decay toward 0. Tracks CHOICES, not rewards.
            ch_c = c1 if idx == 0 else c2
            ch_s = s1 if idx == 0 else s2
            pers_col = pers_col + alpha_p * (ch_c - pers_col)
            pers_shp = pers_shp + alpha_p * (ch_s - pers_shp)
            PERS.append([pers_col.copy(), pers_shp.copy()])

            # check criterion: 8 correct in last 10
            if len(outcomes) >= 10 and sum(outcomes[-10:]) >= 8:
                break
            if guard >= 1000:  # [PFRL +] safety break (criterion not reached)
                break
    return {
        'stimuli': stimuli,
        'choice': choice,
        'reward': reward,
        'rule': rule_list,
        'WH1': WH1,
        'WH2': WH2,
        'PE': PE,
        'PERS': PERS,         # [PFRL +]
        'alpha': alpha,
        'beta': beta,
        'phi': phi,           # [PFRL +]
        'alpha_p': alpha_p,   # [PFRL +]
        'wh0': wh0,
        'subject': subject
    }


def llIED(params, R, choice, S, rule=None, likelihood=False, u=None, v2=None):
    """Negative log-likelihood for FRL + decaying-kernel perseveration.

    Identical to FRL.llIED except: (i) two extra raw parameters (phi used raw,
    alpha_p via sigmoid); (ii) a feature-level choice-kernel trace maintained
    from the observed choices; (iii) phi*P added to the decision variable. The
    value update is unchanged from FRL.
    """
    import numpy as _np
    from scipy.special import logsumexp
    # [PFRL +] Transform FOUR raw params (FRL reshaped to (2,1)). Matches
    #          transforms = ['sigmoid','exp',None,'sigmoid']. Done unconditionally
    #          (as in FRL) so both the BIC path (likelihood=False) and the
    #          avg-likelihood path (likelihood=True) receive native parameters.
    a = _np.asarray(params).reshape(4,1)
    alpha   = 1/(1+_np.exp(-a[0,0]))
    beta    = _np.exp(a[1,0])
    phi     = a[2,0]                    # [PFRL +] perseveration weight, used raw
    alpha_p = 1/(1+_np.exp(-a[3,0]))    # [PFRL +] choice-kernel update rate (0..1)

    # Prepare encodings
    cols = sorted({st.split()[0] for pair in S for st in pair})
    shps = sorted({st.split()[1] for pair in S for st in pair})
    colour_enc = {c: _np.eye(len(cols))[i] for i, c in enumerate(cols)}
    shape_enc = {s: _np.eye(len(shps))[j] for j, s in enumerate(shps)}

    # Initialize weights
    wh_col = _np.zeros(len(cols))
    wh_shp = _np.zeros(len(shps))

    # [PFRL +] feature-level perseveration (choice-kernel) traces (zero on trial 1)
    pers_col = _np.zeros(len(cols))
    pers_shp = _np.zeros(len(shps))

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

        # [PFRL +] feature-level perseveration value and combined decision variable:
        #          beta on value, phi on perseveration.
        P = _np.array([pers_col @ c1 + pers_shp @ s1,
                       pers_col @ c2 + pers_shp @ s2])
        DV = beta * V + phi * P

        # log-softmax over DV (beta is already inside DV)
        logp = DV[idx] - logsumexp(DV)
        ll += logp
        if likelihood:
            like_vals.append(_np.exp(logp))

        # ---- FRL value-learning core: UNCHANGED ----
        rew = R[t]
        R_trial = [None, None]
        R_trial[idx] = rew
        R_trial[1-idx] = -rew
        dV1 = R_trial[0] - V[0]
        dV2 = R_trial[1] - V[1]
        wh_col += alpha * (dV1 * c1 + dV2 * c2)
        wh_shp += alpha * (dV1 * s1 + dV2 * s2)

        # [PFRL +] DECAYING CHOICE-KERNEL update from the OBSERVED choice:
        #          p <- p + alpha_p * ( 1[f in chosen] - p )
        ch_c = c1 if idx == 0 else c2
        ch_s = s1 if idx == 0 else s2
        pers_col = pers_col + alpha_p * (ch_c - pers_col)
        pers_shp = pers_shp + alpha_p * (ch_s - pers_shp)

    if likelihood:
        return (_np.prod(like_vals), _np.mean(like_vals))
    # negative sum of log probs
    return -ll


def fit(params):
    """Use ML to fit PFRL by minimizing llIED.

    IDENTICAL to FRL.fit — the optimiser is parameter-count-agnostic; it simply
    receives a length-4 initial vector (x0) instead of length-2.
    """
    import scipy.optimize, warnings
    warnings.simplefilter('ignore', RuntimeWarning)
    args = params[:-1]
    m0 = params[-1]
    res = scipy.optimize.minimize(fun=llIED, x0=list(m0), args=tuple(args))
    warnings.resetwarnings()
    return list(res.x)
