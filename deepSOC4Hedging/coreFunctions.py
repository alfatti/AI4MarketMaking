import numpy as np
import scipy.stats as stats
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

def calculate_stock_prices(S_0,mu,vol,W): #maybe use Euler for stock
    S = torch.zeros(size=(W.shape[0],W.shape[1]))
    t = torch.arange(W.shape[1])
    for i in range(S.shape[0]):
        S[i,:] = S_0*np.exp(vol*W[i,:]+(mu-vol**2 /2)*t)
    return S
	
def calculate_d1(S, K, ir, vol, tau):
    x = np.log(S/K) + (ir + vol**2 / 2) * tau
    return x / (vol * np.sqrt(tau))

def calculate_d2(S, K, ir, vol, tau):        
    x = np.log(S/K) + (ir - vol**2 / 2) * tau
    return x / (vol * np.sqrt(tau))
	
def calculate_deltas(S, K, ir, vol, T):

    dt = T / (S.shape[1]-1)

    # Convert to torch tensors
    K = torch.full_like(S, K)
    ir = torch.full_like(S, ir)
    vol = torch.full_like(S, vol)

    T = torch.full_like(S, T)
    t = torch.zeros_like(S)
    for j in range(t.shape[1]): t[:, j] = j*dt
    tau = T - t
    deltas = torch.zeros_like(S)

    # Calculate prices before expiry
    d1 = calculate_d1(S[:, :-1], K[:, :-1], ir[:, :-1], vol[:, :-1], tau[:, :-1])
    deltas[:, :-1] = torch.from_numpy(stats.norm.cdf(d1))

    # Calculate prices at expiry
    deltas[:, -1] = torch.from_numpy(np.where(S[:, -1] > K[:, -1], 1.0, 0.0))

    return deltas