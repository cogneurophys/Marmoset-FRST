# -*- coding: utf-8 -*-
"""
Created on Wed Apr 23 13:20:01 2025

@author: mariu
"""

#!/usr/bin/env python
# coding: utf-8

def simulate(params, trials_csv, rule_mapping, subject, shapes, colours, rng=None, seed=42):
    """Simulate IED_assess_fit data for one subject given parameter values.

    Parameters:
    params (list): list of parameters in order [alpha: learning rate, beta: choice determinism].
    subject (str): a subject name.
    rng (func): pass an existing random number generator. Default = None.
    seed (int): if rng is None, this seed will be used to create a random number generator. Default = 42
    shapes (list of str): the two shape labels for this subject
    colours (list of str): the two colour labels for this subject
    rule_mapping (dict): maps every possible code to its feature string

    Returns:
    dict: keys are
    'choice' - list of participants choices on each trial
    'R' - list of
    'r' - list of feedback on each trial
    'S'  - list of stimuli seen on each trial
    'inputs' - list of vectors denoting which features were input into the model on each trial
    'WH1' - list of learnt colour feature values over trials
    'WH2' - list of learnt shape feature values over trials
    'alpha' - learning rate parameter
    'beta' - choice determinism parameter
    'wh0' - initial feature weights
    'dimension1' - first relevant stimulus dimension
    'subject' - subject name
    'stage' - list of stages
    'Trials' - list of trials to criterion for each stage
    'Errors' - list of errors per stages
    'Stages' - list of stage names
    'Passed' - list of booleans, based on whether participant passed or failed each stage.
    """
    import numpy as np
    import pandas as pd
    
    # RNG setup - for reproducibility
    if rng is None:
        rng = np.random.default_rng(seed)
    
    # Acquire rule_data: use passed list or load from CSV (if a list or file pathokay  isnt provided it can't proceed)
    if trials_csv is None:      #if neither a list nor a file path is prvided, function can't proceed 
        raise ValueError("Must provide rule_data or trials_csv to simulate()") 
    df = pd.read_csv(trials_csv)    #reads full trial-by-trial CSV into a dataframe - csv path given in master run file 
    # rule_data = df[df['rule'].isin([2, 3, 4, 6])]['rule']   #Filters to only those rows where your rule column is 2,3,4,6
    rule_data = df['rule'].tolist() #more general line to work with any rule combo 
    
    

#Initializes empty lists that will store evolving values on each trial 
# After each trial, these will be appended to update feature-weight matrices (wh1/wh2) 

    wh1_full   = []   # feature weights for “colour” 
    wh2_full   = []   # feature weights for “shape” 
    PE_full    = []   # prediction errors on the chosen stimulus each trial
    theta_full = []   # the evolving dimension‐primacy weight θ each trial

    #Unpacks our three element parameter vector into named variables 
    #Learning rate - 
    #Beta - choice determinism - inverse temperature for the softmac, alwatys > 0 (bc we exponentiate the raw estimate) 
        #Higher beta makes choices more deterministic - you almost always pick the higher valued option, lower beta makes choices more random 
    #Theta - the intitial dimension-primacy weight - seeds how strongly colour vs. shape is attended to on the very first trial 
    
    alpha, beta, theta0 = params

    # Initializing sigmoid
    def sigmoid(x): return 1 / (1 + np.exp(-x))
     
    theta = np.array(theta0).reshape(1,1)  # start your θ at theta0
    V,AH = [],[]   
    inputs = []  # Initializing inputs

    #task parameters
    
    # generate all 2×2 combos of your passed‐in colours × shapes
    stimuli = [f"{c} {s}" for c in colours for s in shapes]

    #Define one-hot encodings for each feature
    # one-hot encode each passed-in feature
    colour_encoding = {
        c: np.eye(len(colours), dtype=float)[i]
        for i, c in enumerate(colours)
    }
    shape_encoding = {
        s: np.eye(len(shapes), dtype=float)[j]
        for j, s in enumerate(shapes)
    }
        
    #initialize wights for each dimension
    wh1 = np.zeros(len(colours))   # shape (2,)
    wh2 = np.zeros(len(shapes))    # shape (2,)
    #one weight per shape 
    theta = float(theta0) #initializes dimension-bias to the scalar theta - begins as single value(initial dimension primacy, and updates itseld to shift attention between colour vs shape over trials) 


    #Make a dictionary to store simulation results
    results = {
        'stimuli': [],   #stimulus pair on each trial
        'choice': [],    #chosen stimulus on each trial
        'reward': [],    # reward received on each trial
        'rule': []       #rule from data on each trial
    }


    #How many trials we'll actually run/simulate 
    num_trials = len(rule_data)

    
    #get the current rule label from the data to determine the rewarded feature
    for trial in range(num_trials):
        current_rule_label = rule_data[trial]
        current_reward_feature = rule_mapping[current_rule_label]
        results['rule'].append(current_rule_label)

        #Generates a trial such that one of the stimuli contains the rewarded feature and the other does not
        correct_options = [stim for stim in stimuli if current_reward_feature in stim]
        incorrect_options = [stim for stim in stimuli if current_reward_feature not in stim]

        #Randomly choose one stimulus from each category:
        stim_correct = rng.choice(correct_options)
        stim_incorrect = rng.choice(incorrect_options)

        #Randomize the order so the rewarded stimulus can appear on th eleft or right
        if rng.random() < 0.5:
            trial_pair = [stim_correct, stim_incorrect]
        else:
            trial_pair = [stim_incorrect, stim_correct]
        results['stimuli'].append(trial_pair)

        #Split the stimuli into features
        features1 = trial_pair[0].split()
        features2 = trial_pair[1].split()

        #Retrieve one-hot vectors for each feature
        colour1 = colour_encoding[features1[0]]
        shape1 = shape_encoding[features1[1]]
        colour2 = colour_encoding[features2[0]]
        shape2 = shape_encoding[features2[1]]
        
    
        inputs.append([[colour1,shape1],[colour2,shape2]]) # list of lists on each trial (sublist for each stimulus)

        #Compute single-trial activations 
        ah1 = np.dot(wh1, colour1.T)
        ah2 = np.dot(wh2, shape1.T)
        ah3 = np.dot(wh1, colour2.T)
        ah4 = np.dot(wh2, shape2.T)
        
        #Record them 
        AH.append([[ah1,ah2],[ah3,ah4]])
        
        #Computes estimated value for each stimulus
        #The total value is the sum of the contributions from each dimension
      
        # compute two scalar values
        v1 = float(sigmoid(theta)*ah1 + (1-sigmoid(theta))*ah2)
        v2 = float(sigmoid(theta)*ah3 + (1-sigmoid(theta))*ah4)
        # store them as a length-2 vector
        V_trial = np.array([v1, v2])
        V.append(V_trial)


        #Use softmax fxn to convert estimated values into choice probabilities
        exp_vals = np.exp(beta * V_trial)  #how strongly option 'i' is favoured under inverse-temperature beta 
        sev = sum(exp_vals) #sum of exp values - ensures the two probabilities add to 1 
        p = exp_vals / sev  # is now a length-2 array where p[0] + p[1] == 1, so choice can be sampled with rng.choice 9[0,1], p=p) 
        
        #Simulate a choice based on these probabilities
        if rng.random() < p[0]: 
            chosen_index = 0 
        else :
            chosen_index= 1 
        chosen_stim = trial_pair[chosen_index]
        
        results['choice'].append(chosen_stim)

        #Determine the reward
        if current_reward_feature in chosen_stim:
            reward = 1
        else:
            reward = -1
        results['reward'].append(reward)
        

# backpropagation for dimension and feature weight
        
        #find which stimulus was chosen 
        idx = chosen_index 
        
        #Build the two-option reward vector 
        R_trial = [None, None]
        R_trial[idx] = reward    #chosen stimulus got reward r 
        R_trial [1-idx] = -reward  #other stimulus gets opposite
        
        #prediction error for the chosen option
        pe = reward -V_trial[idx]
        PE_full.append(pe)
        
        # 2) Dimension‐weight (θ) gradient
        dcost_dV = V[trial][idx] - R_trial[idx]
        dV_dtheta = (
            sigmoid(theta) * (1 - sigmoid(theta)) * AH[trial][idx][0]
          - sigmoid(theta) * (1 - sigmoid(theta)) * AH[trial][idx][1]
        )
        dcost_theta = np.dot(dcost_dV, dV_dtheta.T)

        # 3) Feature‐weight gradients (phase 1)
        #prediction error with respect to each V component
        #(how much the cost changed when each of the value estimated is off from its target reward)
        dcost_dV1 = V[trial][0] - R_trial[0]
        dcost_dV2 = V[trial][1] - R_trial[1]
        #how each dimension's activation contributes to V 
        dV_dah1 = sigmoid(theta)
        dV_dah2 = 1 - sigmoid(theta)
        #gradient flowing into each raw activation
        dcost_dah1 = dV_dah1 * dcost_dV1
        dcost_dah2 = dV_dah2 * dcost_dV1
        dcost_dah3 = dV_dah1 * dcost_dV2
        dcost_dah4 = dV_dah2 * dcost_dV2


        # 4) Phase 2: back‐prop into your wh updates
        # Phase 2: back‐prop into your wh updates
        col1_vec, shape1_vec = inputs[trial][0]
        col2_vec, shape2_vec = inputs[trial][1]
        
        # use np.dot (or elementwise multiply if the shapes align)
        dcost1_wh1 = dcost_dah1 * col1_vec   # → shape (2,)
        dcost1_wh2 = dcost_dah2 * shape1_vec
        dcost2_wh1 = dcost_dah3 * col2_vec
        dcost2_wh2 = dcost_dah4 * shape2_vec

        # 5) Apply updates exactly as Talwar
        wh1 -= alpha * (dcost1_wh1 + dcost2_wh1)
        wh2 -= alpha * (dcost1_wh2 + dcost2_wh2)

        theta -= alpha * dcost_theta


        wh1_full.append(wh1.copy())
        wh2_full.append(wh2.copy())
        theta_full.append(theta.copy())
        
            # Store results at the end of simulation
    results['WH1'] = wh1_full  # Store learned color weights
    results['WH2'] = wh2_full  # Store learned shape weights
    results['PE'] = PE_full    # Store prediction errors
    results['theta'] = theta_full   # Store prediction errors
    
   
    results['alpha']     = alpha
    results['beta']      = beta
    results['theta0']    = theta0
    results['subject']   = subject
    results['wh0']       = [np.zeros((1,2)), np.zeros((1,2))] #initial weights

    return results

#-----------------------------------------------i added--------------------------------------------------
def flip_stimuli(stimuli):
    """
    Flips the stimulus list by switching the order of shape and color.

    Parameters:
      stimuli (list): List of stimulus strings, where each string contains a shape and a color separated by a space.
      
    Returns:
      list: A new list of stimuli where the shape and color are flipped.
      
    Example:
      Input: ['pentagon purple', 'stapler cyan']
      Output: ['purple pentagon', 'cyan stapler']
    """
    flipped = []
    for item in stimuli:
        # Split the item into two parts
        shape, color = item.split()
        # Reformat the parts and add to the flipped list
        flipped.append(f"{color} {shape}")
    return flipped

#-----------------------------------------------i added--------------------------------------------------

def llIED(params, R, choice, S, rule=None, likelihood=False, u=None, v2=None):
    """
    Negative log-likelihood (or negative log-posterior if using EM) for one subject.

    Parameters
    ----------
    params : array-like of length 3
        [alpha (learning rate), beta (inverse temperature), theta0 (initial dimension bias)].
    R : list of int
        Reward on each trial (internally +1 for correct, –1 for incorrect).
    choice : list of str
        The chosen stimulus (e.g. "purple stapler") each trial.
    S : list of [str, str]
        The two stimuli presented each trial.
    likelihood : bool
        If True, return (total_likelihood, avg_likelihood) instead of negative log-likelihood.
    u : array-like, optional
        Prior mean for EM.
    v2 : 2D array, optional
        Prior covariance for EM.

    Returns
    -------
    float
        Negative log-likelihood (or negative log-posterior when u,v2 given).
    or
    (float, float)
        (total_likelihood, average_likelihood) if `likelihood=True`.
    """
    import numpy as np
    
    #unpack parameters and transform raw parameters 
    nP = len(params)
    ab = np.asarray(params).reshape(nP, 1)
    alpha, beta, theta0 = params

    #Parameter transforms for ML 
    if not likelihood:
        alpha = 1 / (1 + np.exp(-ab[0,0]))
        beta = np.exp(ab[1,0])
        theta0 = np.array((ab[2,0])).reshape(1,1)
        
    # If using EM (with priors), compute the negative log likelihood of drawing these parameters.
    NLP = 0.0 #initializes variable that will hold the negative log-prior 
    if u is not None and v2 is not None: 
       #make sure u is a column vector 
       u = np.asarray(u).reshape(nP,1) 
       PP = ab - u 
       L = 0.5 * (PP.T @ np.linalg.pinv(v2) @ PP)
       LP = -np.log(2 * np.pi) - 0.5 * np.log(np.linalg.det(v2)) - L
       NLP = -LP[0, 0] 

     # before looping over trials, extract the two colour labels and two shape labels
    all_cols   = sorted({stim.split()[0] for pair in S for stim in pair})
    all_shapes = sorted({stim.split()[1] for pair in S for stim in pair})
    
    
    # Define one-hot encodings for our features (2 colours, 2 shapes)
    colour_encoding = {
        c: np.eye(len(all_cols), dtype=float)[i].reshape(1, len(all_cols))
        for i, c in enumerate(all_cols)
    }
    shape_encoding = {
        s: np.eye(len(all_shapes), dtype=float)[j].reshape(1, len(all_shapes))
        for j, s in enumerate(all_shapes)
    }
    # Initialize lists to store predicted values and likelihood contributions.
    AH,V = [], []        # predicted values per trial (a 2-element array each trial)
    inputs = []          # stores the feature inputs for each stimulus per trial 
    l,ll = [0]*2,0       # l is a dummy 2-vector, ll accumulated log-likelihood
    like_vals = []       # list of per-trial choice likelihoods (if likelihood==True)

    # Initialize weights for each dimension (colour and shape)
    wh_col = np.zeros((1, len(all_cols)))
    wh_shp = np.zeros((1, len(all_shapes)))
    theta = theta0.copy()
    
    # Define a sigmoid function (if needed)
    def sigmoid(x):
        return 1/(1 + np.exp(-x))

    # Loop over trials (simulate the learning process and compute likelihood)
    for t in range(len(choice)):
        # decodes stimulus features, e.g., ["purple stapler", "cyan pentagon"]
        stim1, stim2 = S[t]
        # Split each stimulus into its features.
        f1 = stim1.split()  # e.g., ['purple', 'stapler']
        f2 = stim2.split()  # e.g., ['cyan', 'pentagon']
        # Retrieve one-hot vectors for each feature.
        col1 = colour_encoding[f1[0]]
        shp1 = shape_encoding[f1[1]]
        col2 = colour_encoding[f2[0]]
        shp2 = shape_encoding[f2[1]]
        # Record the feature inputs (for potential analysis later).
        inputs.append([[col1, shp1], [col2, shp2]])

        # FEEDFORWARD to estimate stimulus values
        #Compute single-trial activations 
        ah1 = wh_col @ col1.T
        ah2 = wh_shp @ shp1.T
        ah3 = wh_col @ col2.T
        ah4 = wh_shp @ shp2.T
         
        #Record them 
        AH.append([[ah1,ah2],[ah3,ah4]])
        
        #Computes estimated value for each stimulus
        #The total value is the sum of the contributions from each dimension
        
        V1 = sigmoid(theta)*ah1 + (1-sigmoid(theta))*ah2
        V2 = sigmoid(theta)*ah3 + (1-sigmoid(theta))*ah4
        V.append(np.concatenate((V1,V2)))
        
        #Stable softmax log-likelihood
        vmax = beta * np.amax(V[t])
        l = beta*(V[t][S[t].index(choice[t])] - vmax) - np.log(sum((np.exp(beta*(V[t][x]-vmax)) for x in range(len(S[t])))))
        ll += l.copy()

        if likelihood:
            ev = np.exp(beta*V[t])
            sev = sum(ev)
            p = ev/sev
                
            like_vals.append(p[[stim1,stim2].index(choice[t])])   
            
        idx = [stim1, stim2].index(choice[t])
        
        #Build reward vector 
        R_trial = [None, None]
        R_trial[idx] = R[t]
        R_trial[1-idx] = -R[t]        
     
        #Feature weight update 
        dV1 = V[t][0] - R_trial[0]
        dV2 = V[t][1] - R_trial[1]
        dV_dah1 = sigmoid(theta)
        dV_dah2 = 1-sigmoid(theta)
        
        dcost1 = dV_dah1 * dV1
        dcost2 = dV_dah2 * dV1
        dcost3 = dV_dah1 * dV2
        dcost4 = dV_dah2 * dV2
        
        wh_col -= alpha * (dcost1 * col1 + dcost3 * col2)
        wh_shp -= alpha * (dcost2 * shp1 + dcost4 * shp2)
       
        #Dimension-weight update 
        dcostDV = V[t][idx] - R_trial[idx]
        dV_dtheta = sigmoid(theta)*(1-sigmoid(theta)) * (AH[t][idx][0] - AH[t][idx][1])
        dcost_theta = (dcostDV * dV_dtheta).item()
        theta -= alpha * dcost_theta
        
    if likelihood == True:
        return (np.prod(like_vals), np.mean(like_vals))
    
    if u is not None and v2 is not None:
        return -ll + NLP
    else:
        return -ll
          
    
def fit(params):
    """
    Optimise (log) likelihood for subject data given parameter values.

    params is the single list that modelling_Mars0429.fit hands you:
      • For ML: [R, choice, S, rule, False, m0]
      • For EM: [R, choice, S, rule, False, u, v2, m0, s2, rng]
    """
    import scipy.optimize, scipy.linalg
    import numpy as np
    import warnings

    warnings.simplefilter('ignore', RuntimeWarning)

    if len(params) > 6:
        # if we are using EM.
        args = params[:-3]  # [R, choice, S, rule, False, u, v2]
        rng = params[-1]    # last element is the RNG
        var = 1
        while var >= 1:
            for x in range(50):
                if x == 0:
                    m = params[-3]
                else:
                    m = params[-3] + x * 0.1 * np.matmul(rng.standard_normal((1, len(m))),
                                                          scipy.linalg.sqrtm(np.linalg.inv(0.1 * np.identity(len(m)))))
                xopt = scipy.optimize.minimize(fun=llIED,
                               x0=list(m),
                               args=tuple(args))
                if xopt.success:
                    warnings.resetwarnings()
                    return ([xopt, var])
                else:
                    continue
            var += 1
        warnings.resetwarnings()
    
    else:
        # We are using ML.
        args = params[:-1]
        m = params[-1]
        # Minimise negative log likelihood for subject data and given parameters.
        xopt = scipy.optimize.minimize(fun=llIED,
                               x0=list(m),
                               args=tuple(args))
        params = list(xopt['x'])
        return (params)