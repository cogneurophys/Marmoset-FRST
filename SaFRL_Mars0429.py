# -*- coding: utf-8 -*-
"""
Created on YYYY-MM-DD by Marium

Continuous SaFRL implementation integrated into the CaFRL pipeline.
Provides a standalone simulate, llIED, and fit interface for ML fitting.

Parameters per subject: [alpha (feature LR), beta (inv. temperature), theta0 (initial bias), epsilon (dim learning rate)].
Functions:
  - simulate: trial-by-trial continuous SaFRL using real IED data.
  - llIED: computes negative log-likelihood for ML.
  - fit: performs ML fitting by minimizing llIED.
"""

import numpy as np
import pandas as pd


def simulate(params, trials_csv, rule_mapping, subject, shapes, colours, rng=None, seed=42):
    """
    Simulate continuous SaFRL data for one subject.
    """
    # RNG
    rng = rng if rng else np.random.default_rng(seed)
    # Load data
    df = pd.read_csv(trials_csv)
    if 'reward' in df.columns:
        df['reward'] = df['reward'].map({0: -1, 1: 1})
    rule_data = df['rule'].tolist()
    # Prepare all possible stimuli
    stimuli_all = [f"{c} {s}" for c in colours for s in shapes]
    # Unpack params
    alpha, beta, theta0, eps = params
    theta = float(theta0)
    # Encodings
    colour_encoding = {c: np.eye(len(colours))[i] for i, c in enumerate(colours)}
    shape_encoding = {s: np.eye(len(shapes))[j] for j, s in enumerate(shapes)}
    # Initialize weights
    wh_col = np.zeros(len(colours))
    wh_shp = np.zeros(len(shapes))
    wh0 = [wh_col.copy(), wh_shp.copy()]
    # Storage
    stimuli, choice, reward, rule_list = [], [], [], []
    WH1, WH2, theta_list, PE = [], [], [], []
    def sigmoid(x): return 1/(1+np.exp(-x))
    # Loop
    for feat_code, feat_rule in enumerate(rule_data):
        feat = rule_mapping[feat_rule]
        # sample stimuli
        correct = [st for st in stimuli_all if feat in st]
        incorrect = [st for st in stimuli_all if feat not in st]
        stim_corr = rng.choice(correct)
        stim_inc = rng.choice(incorrect)
        pair = [stim_corr, stim_inc] if rng.random()<0.5 else [stim_inc, stim_corr]
        stimuli.append(pair)
        rule_list.append(feat_rule)
        # encode
        col1, shp1 = pair[0].split()
        col2, shp2 = pair[1].split()
        c1, s1 = colour_encoding[col1], shape_encoding[shp1]
        c2, s2 = colour_encoding[col2], shape_encoding[shp2]
        # activations
        ah1 = wh_col @ c1; ah2 = wh_shp @ s1
        ah3 = wh_col @ c2; ah4 = wh_shp @ s2
        # values
        v1 = sigmoid(theta)*ah1 + (1-sigmoid(theta))*ah2
        v2 = sigmoid(theta)*ah3 + (1-sigmoid(theta))*ah4
        V = np.array([v1, v2])
        # choice
        expV = np.exp(beta*V); probs = expV/expV.sum()
        idx = 0 if rng.random()<probs[0] else 1
        ch = pair[idx]; choice.append(ch)
        # reward
        rew = 1 if feat in ch else -1; reward.append(rew)
        # PE
        R_trial = [None,None]; R_trial[idx]=rew; R_trial[1-idx]=-rew
        pe = rew - V[idx]; PE.append(pe)
        # update features
        dV1 = V[0]-R_trial[0]; dV2 = V[1]-R_trial[1]
        g1 = sigmoid(theta); g2 = 1-g1
        wh_col -= alpha*(g1*dV1*c1 + g1*dV2*c2)
        wh_shp -= alpha*(g2*dV1*s1 + g2*dV2*s2)
        # update theta
        dcost = V[idx]-R_trial[idx]
        dtheta = sigmoid(theta)*(1-sigmoid(theta))*((ah1-ah2) if idx==0 else (ah3-ah4))
        theta -= eps*dcost*dtheta
        # log
        WH1.append(wh_col.copy()); WH2.append(wh_shp.copy()); theta_list.append(theta)
    return {
        'stimuli': stimuli, 'choice': choice, 'reward': reward, 'rule': rule_list,
        'WH1': WH1, 'WH2': WH2, 'theta': theta_list, 'PE': PE,
        'alpha': alpha, 'beta': beta, 'theta0': theta0, 'epsilon': eps,
        'wh0': wh0, 'subject': subject
    }


def llIED(params, R, choice, S, rule=None, likelihood=False, u=None, v2=None):
    """Compute negative log-likelihood for ML."""
    import numpy as np
    a = np.asarray(params).reshape(4,1)
    alpha = 1/(1+np.exp(-a[0,0])); beta = np.exp(a[1,0])
    theta = float(a[2,0]); eps = 1/(1+np.exp(-a[3,0]))
    cols = sorted({st.split()[0] for pair in S for st in pair})
    shps = sorted({st.split()[1] for pair in S for st in pair})
    colour_enc = {c: np.eye(len(cols))[i] for i,c in enumerate(cols)}
    shape_enc = {s: np.eye(len(shps))[j] for j,s in enumerate(shps)}
    def sigmoid(x): return 1/(1+np.exp(-x))
    wh_col = np.zeros(len(cols)); wh_shp = np.zeros(len(shps))
    ll=0.0; like_vals=[]
    for t in range(len(choice)):
        col1, shp1 = S[t][0].split(); col2, shp2 = S[t][1].split()
        c1, s1 = colour_enc[col1], shape_enc[shp1]
        c2, s2 = colour_enc[col2], shape_enc[shp2]
        ah1 = wh_col @ c1; ah2 = wh_shp @ s1
        ah3 = wh_col @ c2; ah4 = wh_shp @ s2
        v1 = sigmoid(theta)*ah1 + (1-sigmoid(theta))*ah2
        v2 = sigmoid(theta)*ah3 + (1-sigmoid(theta))*ah4
        V = np.array([v1, v2])
        vmax = beta*np.max(V)
        idx = [S[t][0], S[t][1]].index(choice[t])
        ll_trial = beta*(V[idx]-vmax) - np.log(np.sum(np.exp(beta*(V-vmax))))
        ll += ll_trial
        if likelihood:
            probs = np.exp(beta*V); probs/=probs.sum(); like_vals.append(probs[idx])
        rew = R[t]; R_trial=[None,None]; R_trial[idx]=rew; R_trial[1-idx]=-rew
        dV1 = V[0]-R_trial[0]; dV2 = V[1]-R_trial[1]
        g1 = sigmoid(theta); g2=1-g1
        wh_col -= alpha*(g1*dV1*c1 + g1*dV2*c2)
        wh_shp -= alpha*(g2*dV1*s1 + g2*dV2*s2)
        dcost = V[idx]-R_trial[idx]
        dtheta = sigmoid(theta)*(1-sigmoid(theta))*((ah1-ah2) if idx==0 else (ah3-ah4))
        theta -= eps*dcost*dtheta
    if likelihood: return (np.prod(like_vals), np.mean(like_vals))
    return -ll


def fit(params):
    """Maximum Likelihood estimation by minimizing llIED."""
    import scipy.optimize, numpy as np, warnings
    warnings.simplefilter('ignore', RuntimeWarning)
    args = params[:-1]
    m0 = params[-1]
    res = scipy.optimize.minimize(fun=llIED, x0=list(m0), args=tuple(args))
    warnings.resetwarnings()
    return list(res.x)
