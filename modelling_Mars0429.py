# -*- coding: utf-8 -*-
"""
Created on Thu May  1 14:32:35 2025

@author: mariu
"""

#!/usr/bin/env python
# coding: utf-8

############################################
# 1. Simulation Function
############################################

def simulate(model, model_args=None, rng=None, seed=42, params=None, dist=None, N=100, transforms=None, subjects=None, **model_kwargs):
    """
Simulate data from a given model, drawing or accepting parameters, and forward any extra
    keyword arguments (e.g., trials_csv, rule_mapping, shapes, colours) to the model itself.

    Parameters:
      model (func): Your CAFRL_Mars0429.simulate function.
      model_args: Unused here (preserved for compatibility).
      rng: Random number generator or None.
      seed: Seed for RNG if rng is None.
      params: N x nP array of parameter vectors, or None to draw from dist+transforms.
      dist: Tuple (mean, cov) for drawing parameters if params is None.
      N: Number of simulated "subjects".
      transforms: List of transformations for each parameter (e.g., ['sigmoid','exp',None]).
      subjects: List of subject IDs; if None, defaults to range(N).
      **model_kwargs: Forwarded directly into the model call.
    Returns:
      dict: A dictionary with keys:
            'params' - an array of transformed parameters used for simulation,
            'data'   - a list (length N) where each element is the simulated data for one subject.
    """
    import numpy as np
    
    #RNG set up 
    rng = rng if rng else np.random.default_rng(seed)

    if model is None:
        raise Exception('Essential model argument not received.')

    # If no parameters are provided, draw N parameter vectors from the specified distribution.
    if params is None:
        if dist is None or transforms is None:
            raise Exception('dist or transforms argument not received. Both must be given if params is not provided.')
        params = rng.multivariate_normal(dist[0], dist[1], N)
        nP = params.shape[1]
        for x in range(nP):
            if transforms[x] == 'sigmoid':
                params[:, x] = 1 / (1 + np.exp(-params[:, x]))
            elif transforms[x] == 'exp':
                params[:, x] = np.exp(params[:, x])
    else:
        N = params.shape[0]

    subjects = subjects if subjects else list(range(N))

    # Simulate data for each subject using the provided model function.
    data = [
        model(
            params=params[x, :], 
            subject=subjects[x],
            rng=rng,
            seed=seed + x,
            **model_kwargs
        )
            for x in range(N)
    ]

    results = {'params': params, 'data': data}
    return results


############################################
# 2. Fitting Function
############################################

def fit(data, nP, fit_func, fit_args, transforms, rng=None, seed=42, n_jobs=2, fit='ML', EM_iter=100):
    """
    Estimate parameter values from subject data given a model.

    Parameters:
      data (list): List of subject data dictionaries.
      nP (int): Number of parameters to estimate per subject.
      fit_func (func): Function that fits data for a given subject.
      fit_args (list): List of keys (from each subject's data dictionary) to pass to fit_func.
                   For example, for our task you might use: ['reward', 'choice', 'stimuli', 'rule'].
                   (Note: if your simulation returns 'stimuli' rather than 'S', use that key.)
      transforms (list): List of parameter transformations (e.g., 'exp', 'sigmoid', or None).
      rng: Optional random number generator.
      seed (int): Seed value if rng is not provided.
      n_jobs (int): Number of parallel jobs.
      fit (str): 'ML' for maximum likelihood or 'EM' for expectation maximization.
      EM_iter (int): Number of iterations of the EM algorithm.

    Returns:
      dict: If fit=='EM', returns a dictionary with keys:
            'm'  - array of individual best-fitting parameters,
            's2' - individual parameter covariances,
            'u'  - group-level prior mean,
            'v2' - group-level prior covariance.
            If fit=='ML', returns {'m': array of best-fitting parameters}.
    """
    import numpy as np
    import scipy.linalg
    from multiprocessing import Pool
    import tqdm
    
    rng = rng if rng else np.random.default_rng(seed)

    if (data is None) | (nP is None) | (fit_args is None) | (fit_func is None) | (transforms is None): 
        raise Exception('At least one of essential arguments data, nP, fit_args, fit_func, transforms not received.') 

    N = len(data) # set number of participants as length of data list
    v20 = 0.1 * np.identity(nP)
    m = np.matmul(rng.standard_normal((N, nP)), scipy.linalg.sqrtm(np.linalg.inv(v20))) # initialise individual parameters

    
    if fit == 'ML':
        # Maximum Likelihood estimation (one subject at a time).
        # generate extra arguments required for fit function one participant at a time
        args = ([data[x][fit_args[y]] for y in range(len(fit_args))] + [False,m[x,:]] for x in range(N))
        p = Pool(n_jobs)
        
        # use multiprocessing to fit each participants parameters
        results = list(tqdm.tqdm(p.imap(fit_func, args), desc='Participant'))
        p.close()
        p.join()
        m = np.array(results)
        # transform parameters back to correct range values
        
        for x in range(nP):
            if transforms[x] == 'sigmoid':
                m[:, x] = 1 / (1 + np.exp(-m[:, x]))
            elif transforms[x] == 'exp':
                m[:, x] = np.exp(m[:, x])
        results = {'m':m}
        return results
    else:
        if fit != 'EM':
            raise Exception('fit argument must be set to "ML" or "EM".')
        u0 = np.mean(m, axis=0) # set prior mean to mean of m's
        u, v2 = u0.copy(), v20.copy()
        U, V2 = [u.copy()], [v2.copy()]
        s2 = np.tile(v20.copy(), N)
        M, S2 = [m.copy()], [s2.copy()]

        for t in range(EM_iter):
            args = ([data[x][fit_args[y]] for y in range(len(fit_args))] + [False,u,v2,m[x,:],s2[:,x*nP:(x+1)*nP],rng] for x in range(N))
            p = Pool(n_jobs)
            results = list(tqdm.tqdm(p.imap(fit_func, args), desc='Participant'))
            p.close()
            p.join()
            for x in range(N):
                m[x, :] = results[x][0]['x']
                s2[:, x * nP:(x + 1) * nP] = results[x][0].hess_inv
            M.append(m.copy())
            S2.append(s2.copy())
            u = np.mean(m, axis=0)
            v2 = sum([np.outer(m[x, :], m[x, :]) + s2[:, x * nP:(x + 1) * nP] for x in range(N)]) / N - np.outer(u, u)
            U.append(u.copy())
            V2.append(v2.copy())
            print('Iteration:', t + 1, u, v2)
            if len(U) >= 15 and np.all(abs(U[-1] - U[-2]) < 1e-3) and np.all(abs(V2[-1] - V2[-2]) < 1e-3):
                break
        
        for x in range(nP):
            if transforms[x] == 'sigmoid':
                m[:, x] = 1 / (1 + np.exp(-m[:, x]))
            elif transforms[x] == 'exp':
                m[:, x] = np.exp(m[:, x])
        results = {'m':m,'s2':s2,'u':u, 'v2':v2}
        return results


############################################
# 3. Average Likelihood per Trial
############################################

def alpt(data, like_func, fit_args, params, n_jobs=1):
    """
    Calculate average likelihood per trial given data, a likelihood function, and parameter values.

    Parameters:
      data (list): List of subject data dictionaries.
      like_func (func): Likelihood function.
      fit_args (list): List of keys to extract from each subject's data (e.g., ['reward', 'choice', 'stimuli', 'rule']).
                    (Make sure these keys match those in your simulation output.)
      params (array): An N x nP array of transformed parameter values.
      n_jobs (int): Number of processes to use.

    Returns:
      dict: Contains:
            'alpt' - Average likelihood per trial over all subjects,
            'subject_alpts' - List of average likelihood per trial for each subject.
    """
    import numpy as np
    from tqdm import tqdm
    from multiprocessing import Pool

    N = len(data)
    assert N == params.shape[0]
    
    #Build one tuple per subject: (params_raw, R, choice, stimuli, rule, True) 
    arg_tups = []
    for i in range(N):
        R, choice, stimuli, rule = (data[i][k] for k in fit_args)
        arg_tups.append((params[i, :], R, choice, stimuli, rule, True))

    with Pool(n_jobs) as pool:
          results = list(tqdm(pool.starmap(like_func, arg_tups), total=N, desc="Participants"))

  # each result is (total_like, avg_like)
          subject_alpts = [res[1] for res in results]
          return {'alpt': np.mean(subject_alpts), 'subject_alpts': subject_alpts}



############################################
# 4. Integrated Bayesian Information Criterion (iBIC)
############################################

def alpt(data, like_func, fit_args, params, n_jobs=1):
    import numpy as np
    from tqdm import tqdm
    from multiprocessing import Pool

    N = len(data)
    assert N == params.shape[0]

    # Build argument tuples
    arg_tups = []
    for i in range(N):
        R, choice, stimuli, rule_data = (data[i][k] for k in fit_args)
        arg_tups.append((params[i, :], R, choice, stimuli, rule_data, True))

    if n_jobs == 1:
        results = [like_func(*tup) for tup in tqdm(arg_tups, desc="Participants")]
    else:
        with Pool(n_jobs) as pool:
            results = list(tqdm(pool.starmap(like_func, arg_tups), total=N, desc="Participants"))

    # unpack (total_like, avg_like)
    subject_alpts = [res[1] for res in results]
    return {'alpt': np.mean(subject_alpts), 'subject_alpts': subject_alpts}


def iBIC(u, v2, data, transforms, like_func, fit_args, rng=None, seed=42, Nsamples=5000, n_jobs=1):
    """
    Calculate the integrated Bayesian Information Criterion given a group‐level prior.
    """
    import numpy as np
    from multiprocessing import Pool
    from tqdm import tqdm

    rng = rng or np.random.default_rng(seed)
    nP = v2.shape[0]
    N = len(data)

    # 1) Draw from prior & apply transforms
    samples = rng.multivariate_normal(u, v2, Nsamples)
    for j in range(nP):
        if transforms[j] == 'sigmoid':
            samples[:, j] = 1/(1+np.exp(-samples[:, j]))
        elif transforms[j] == 'exp':
            samples[:, j] = np.exp(samples[:, j])

    sublog = []
    total_trials = 0

    # 2) For each subject, estimate log p(data|θ) under the prior
    for subj in tqdm(range(N), desc="Subjects"):
        subj_data = data[subj]
        R = subj_data['reward']
        choice = subj_data['choice']
        stimuli = subj_data['stimuli']
        rule = subj_data['rule']
        total_trials += len(choice)

        # build Nsamples tuples: (θ_sample, R, choice, stimuli, rule, True)
        arg_tups = [(samples[k, :], R, choice, stimuli, rule, True)
                    for k in range(Nsamples)]
        with Pool(n_jobs) as pool:
            likes = pool.starmap(like_func, arg_tups)

        # each like is (total_like, avg_like); we want the likelihood itself = total_like
        total_likes = np.array([l[0] for l in likes])
        sublog.append(np.log(np.mean(total_likes)))

    # 3) integrate & penalty
    iBIC_val = sum(sublog) - nP * np.log(total_trials)
    return {'iBIC': iBIC_val, 'sublog': sublog, 'n': total_trials}



############################################
# 5. Parameter Recovery Plotting
############################################

def recovery(sim_params, fit_params, names=None, color='lightseagreen', alpha=0.5, s=80):
    """
    Assess and plot parameter recovery given simulated and recovered parameter values.

    Parameters:
      sim_params (array): Array of simulated parameters.
      fit_params (array): Array of recovered parameters.
      names (list): List of parameter names (optional).
      color (str): Color for scatter points.
      alpha (float): Transparency for points.
      s (int): Marker size.
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    import numpy as np
    from scipy import stats
    import matplotlib

    font = {'family': 'Arial', 'size': 22}
    matplotlib.rc('font', **font)

    for x in range(sim_params.shape[1]):
        fig, ax = plt.subplots(figsize=(15,15), dpi=300, facecolor='w', edgecolor='k')
        ax = sns.scatterplot(x=sim_params[:, x], y=fit_params[:, x], color=color, s=s, alpha=alpha, edgecolor=None)
        lims = [np.min([ax.get_xlim(), ax.get_ylim()]), np.max([ax.get_xlim(), ax.get_ylim()])]
        ax.plot(lims, lims, 'k:', alpha=0.7, zorder=0)
        name = names[x] if names else 'parameter ' + str(x+1)
        nas = np.logical_or(np.isnan(fit_params[:, x]), np.isinf(fit_params[:, x]))
        corr, p_val = stats.pearsonr(sim_params[:, x][~nas], fit_params[:, x][~nas])
        print('%s corr: %f, p: %f' % (name, corr, p_val))
        plt.xlabel('Simulated')
        plt.ylabel('Recovered')
        plt.title(name)
        sns.despine()
        plt.show()