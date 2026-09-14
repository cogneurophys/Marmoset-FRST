# -*- coding: utf-8 -*-
"""
Created on Fri May  2 13:07:50 2025

@author: mariu
"""

#!/usr/bin/env python
# coding: utf-8

############################################
# IED Code
############################################
def plot_idata(data, colours=None, shapes=None):
    """
    Generate plots for a continuous Colour–Shape RL run.

    Parameters
    ----------
    data : dict
      Must contain keys 'choice', 'WH1', 'WH2', 'PE', and 'stimuli'.
    colours : list of str, optional
      The ordered list of colour labels (e.g. ['purple','cyan']). 
      If None, will infer from the first word of data['stimuli'].
    shapes : list of str, optional
      The ordered list of shape labels (e.g. ['stapler','pentagon']).
      If None, will infer from the second word of data['stimuli'].
    """
    import seaborn as sns
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib import rcParams

    # infer feature sets from the very first trial if not provided
    first_pair = data['stimuli'][0]
    if colours is None:
        colours = sorted({stim.split()[0] for stim in first_pair})
    if shapes is None:
        shapes = sorted({stim.split()[1] for stim in first_pair})

    # get plotting colours from the default cycle
    cycle = rcParams['axes.prop_cycle'].by_key()['color']

    # 1) choices
    plt.figure(figsize=(12,4))
    sns.scatterplot(x=range(len(data['choice'])), y=data['choice'],
                    color=cycle[0], alpha=0.6, s=50)
    plt.xlabel('Trial')
    plt.ylabel('Choice')
    plt.title('Choices over Trials')
    plt.show()

    # 2) colour‐weight trajectories
    if 'WH1' in data:
        wh1 = np.vstack(data['WH1'])   # (n_trials, n_colours)
        plt.figure(figsize=(12,4))
        for i, col in enumerate(colours):
            plt.plot(wh1[:,i], color=cycle[i], label=col.capitalize())
        plt.legend()
        plt.xlabel('Trial')
        plt.ylabel('Colour feature weight')
        plt.title('Colour Weights')
        plt.show()

    # 3) shape‐weight trajectories
    if 'WH2' in data:
        wh2 = np.vstack(data['WH2'])   # (n_trials, n_shapes)
        plt.figure(figsize=(12,4))
        for i, shp in enumerate(shapes):
            plt.plot(wh2[:,i], color=cycle[i], label=shp.capitalize())
        plt.legend()
        plt.xlabel('Trial')
        plt.ylabel('Shape feature weight')
        plt.title('Shape Weights')
        plt.show()

    # 4) prediction error
    if 'PE' in data:
        pe = np.array(data['PE'])
        plt.figure(figsize=(12,4))
        plt.plot(pe, color=cycle[1], alpha=0.7)
        plt.axhline(0, color='k', ls=':')
        plt.xlabel('Trial')
        plt.ylabel('Prediction error')
        plt.title('Prediction Error')
        plt.show()

    # For a continuous task, stage-level plots (errors per stage, trials per stage) are not applicable.
    # You can include additional plots as needed.

    return


def load_trialdata(csv, subjects=None):
    """
    Load trial-by-trial data from CSV.

    Parameters:
      csv (str): Directory (file path) of the CSV containing trial-by-trial IED data.
      subjects (list): List of subjects to filter dataframe by (if applicable).

    Returns:
      DataFrame: Cleaned DataFrame.
    """
    import pandas as pd
    import numpy as np

    df = pd.read_csv(csv)  # Adjust delimiter if necessary

     # Map raw 0/1 → –1/+1 and then duplicate (-1/+1 stabilizes learning) 
    if 'reward' in df.columns:
        # overwrite `reward` in place:
        df['reward'] = df['reward'].map({0: -1, 1: 1})
        # keep a record of your original 0/1 if you want:
        df['trials_IsAnswerCorrect'] = df['reward']
    else:
        print("Warning: 'reward' column not found in CSV!")
        
    if subjects is not None:
        df = df[df.subject.isin(subjects)]
    df = df.drop_duplicates()
    # df[['trials_IsAnswerCorrect']] = df[['trials_IsAnswerCorrect']].astype(int).replace({0: -1})
    return df
    
    # subjects_list = subjects


def trials2dict(dftrials):
    """
    Convert trial-by-trial DataFrame to a list of dictionaries for each subject.

    Parameters:
      dftrials (DataFrame): must have columns 
          'trial', 'stimulus_1', 'stimulus_2', 'choice', 'reward', 'rule', and 'subject'.

    Returns:
      list of dict: One dict per subject, each with keys:
            'trial', 'choice', 'reward', 'rule', 'stimuli', 'subject'.
    """
    dicts = []
    # fallback if somehow subject wasn't added upstream
    if 'subject' not in dftrials.columns:
        dftrials['subject'] = 'subj_1'

    for subj in dftrials['subject'].unique():
        dft = dftrials[dftrials['subject'] == subj].reset_index(drop=True)
        dicts.append({
            'trial':   dft['trial'].tolist(),
            'choice':  dft['choice'].tolist(),
            'reward':  dft['reward'].tolist(),
            'rule':    dft['rule'].tolist(),
            'stimuli': [[s1, s2] for s1, s2 in zip(dft['stimulus_1'], dft['stimulus_2'])],
            'subject': subj
        })
    return dicts

#-----------------------------------------------i added--------------------------------------------------
def flip_one_stim(item, colours, shapes):
    """
    Ensure that a single stimulus string is always 'colour shape'.
    If it's 'shape colour', flip it; otherwise leave it.
    """
    a, b = item.split()
    if a in colours and b in shapes:
        return item
    if a in shapes and b in colours:
        return f"{b} {a}"
    raise ValueError(f"Unrecognized feature in '{item}'")

def flip_stimuli_pair(pair, colours, shapes):
    """Flip each of the two stimuli in a pair to 'colour shape' order."""
    return [
        flip_one_stim(pair[0], colours, shapes),
        flip_one_stim(pair[1], colours, shapes),
    ]

def flip_stimuli_in_data(data, colours, shapes):
    """
    For each subject in data, ensure that both the 'stimuli' pairs and the 'choice'
    strings are in 'colour shape' order.
    
    Parameters:
      data: list of dicts, each with keys 'stimuli' (list of [s1,s2]) and 'choice' (list of strings)
      colours: the two colour labels (e.g. ['purple','cyan'])
      shapes:  the two shape  labels (e.g. ['stapler','pentagon'])
    """
    for subj_data in data:
        # flip each trial’s stimuli pair
        subj_data['stimuli'] = [
            flip_stimuli_pair(pair, colours, shapes)
            for pair in subj_data['stimuli']
        ]
        # flip each choice
        subj_data['choice'] = [
            flip_one_stim(ch, colours, shapes)
            for ch in subj_data['choice']
        ]
    return data

#-----------------------------------------------i added--------------------------------------------------


def assess_fit(trials_csv, subjects_pickle,subject_id,shapes, colours, rule_mapping, nP,
               transforms, fit_args, fit='ML', n_jobs=1, seed=42, Nsamples=5000, EM_iter=100):
    """
    Master Function. Loads trial data, fits the model, simulates data, and compares
    simulated with real data.

    Parameters:
      trials_csv (str): Path to CSV containing trial-by-trial data.
      subjects (str): Path to a pickled list of subject identifiers.
      nP (int): Number of parameters to be estimated per subject.
      transforms (list): List of parameter transformations.
      n_jobs (int): Number of parallel jobs.
      seed (int): Seed for the RNG.
      fit (str): 'ML' or 'EM'.
      rng: Optional RNG.
      Nsamples (int): Number of samples for iBIC.
      EM_iter (int): Number of EM iterations.

    Returns:
      dict: Contains:
            'est'   - Output from parameter estimation,
            'df_all' - Combined trial-level data Marmoset and Model),
            'iBIC'  - iBIC metric.
    """
    import numpy as np
    import pandas as pd
    import seaborn as sns
    import matplotlib.pyplot as plt

    rng = np.random.default_rng(seed)

    #1) Load & Prepare
    
    print('**Loading Trial-by-Trial Data**')
    
    df = IED_Mars0429.load_trialdata(trials_csv, subjects=None)
    #force subject tag 
    df['subject'] = subject_id
    data = IED_Mars0429.trials2dict(df)
    # assemble the 2-item stimuli field
    for d in data:
        d['stimuli'] = list(zip(d.pop('stimulus_1'), d.pop('stimulus_2')))
        # flip any “shape colour” back to “colour shape”
    data = IED_Mars0429.flip_stimuli_in_data(data, colours, shapes)
    
    #2) Fit 
    print("Fitting parameters...")
    est = modelling_Mars0429.fit(
        data=data, 
        nP=nP, 
        fit_func = CaFRL_Mars0429.fit, 
        fit_args=fit_args, 
        transforms=transforms, 
        fit=fit,
        n_jobs=n_jobs, 
        seed=seed, 
        EM_iter=EM_iter
    )
    
    #3) Simulate with Fits 
    print("Simulating with fitted parameters....")
    sim = modelling_Mars0429.simulate(
        model=CaFRL_Mars0429.simulate, 
        trials_csv=trials_csv, 
        subject=subject_id, 
        shapes=shapes, 
        colours=colours, 
        rule_mapping=rule_mapping, 
        params=est['m'], 
        transforms = transforms, 
        rng = rng, 
        seed = seed
    )
    
    #4) Plot Real vs Model Rewads 
    df_real = pd.DataFrame({
        'trial': sim['data'][0]['trial'],
        'reward': sim['data'][0]['reward'],
        'Data': 'Model'
    })
    df_all = pd.concat([df_real, df_mod], ignore_index=True)
    sns.lineplot(x='trial', y='reward', hue='Data', data = df_all)
    plt.title('Reward: Marmoset vs Model')
    plt.show()
    
    #5) iBIC or BIC 
    print(f"Computing{'BIC' if fit == 'ML' else 'iBIC'}...")
    if fit =='ML': 
        #standard BIC 
        total_logL = 0
        total_trials = 0
        for sub_idx, d in enumerate(data):
            args_sub = [
                d['reward'],
                d['choice'],
                d['stimuli'],
                d['rule'],
                False  # likelihood=False ⇒ llIED returns NLL
            ]
            nll = CaFRL_Mars0429.llIED(est['m'][sub_idx], args_sub)
            total_logL += (-nll)
            total_trials += len(d['choice'])
        BIC = -2 * total_logL + nP * np.log(total_trials)
        ibic = {'iBIC': BIC, 'sublog': None, 'n': total_trials}
    else:
        ibic = modelling_Mars0429.iBIC(
            u=est['u'], v2=est['v2'], data=data,
            transforms=transforms,
            like_func=CaFRL_Mars0429.llIED,
            fit_args=fit_args,
            rng=rng, seed=seed,
            Nsamples=Nsamples, n_jobs=n_jobs
        )

    return {'est': est, 'sim': sim, 'ibic': ibic, 'df_all': df_all}
    

############################################
# End of IED Code
############################################

