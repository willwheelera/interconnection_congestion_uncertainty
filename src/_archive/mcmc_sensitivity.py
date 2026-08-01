import numpy as np
from utils import avail_to_id, id_to_avail



def move_pTf(x, w, df):
    # need to load file for sensitivity
    result = df[avail_to_id(x)]
    s = compute_sensitivity(result)
    x_ = propose_move(x, s)
    result_ = df[avail_to_id(x_)]

    a_ = compute_sensitivity(result_)
    rp = p_ratio(x, x_)
    rT = reverse_ratio(x, x_, a, a_)
    A0 = np.minimum(1., rp * rT)
    
    rf = importance_ratio(x, x_)
    A = np.minimum(1., rp * rT * rf)
    # by the time we compute f (rf), we want very low chance of rejecting
    # hopefully rT helps

    # recording list of samples and weights: just record (x, w), (x', w'), even if move is rejected, and keep sampling.
    w1 = w * A0 / A
    w2 = w * (1 - A0) / (1 - A)

    check = np.random.random()
    if A > check:
        x_new = x_
        wnew = w1
    else:
        x_new = x
        wnew = w2
    return x_new, wnew, w1, w2
    
def p_ratio(x, xp):
    # Implement probability function
    pass

def importance_ratio(x, xp):
    # Implement importance function
    pass

def compute_sensitivity(result):
    df = mu_sensitivity_df(result)
    M = np.linalg.norm(df.values)
    s = df.sum(axis=0) / M
    return s

def _new_prob(x, s, p_flip):
    return 0.5 * (p_flip + s) + x * (0.5 - p_flip)

def propose_move(x, s, p_flip=0.1):
    p = _new_prob(x, s, p_flip)
    xnew = np.random.random(len(x)) < p
    return xnew.astype(int)

def reverse_ratio(x, xp, a, ap, p_flip):
    p1 = _new_prob(x, a, p_flip)
    p2 = _new_prob(xp, ap, p_flip)
    return np.prod((1 - xp - p1 + 2*xp*p1) / (1 - x - p2 + 2*x*p2))

