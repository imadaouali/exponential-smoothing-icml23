import torch
from torch import nn, optim
import pytorch_lightning as pl
import numpy as np
from tqdm import tqdm
from sklearn import datasets, preprocessing
from scipy.optimize import minimize_scalar, minimize
from .utils import *
from .policies import *

probit = torch.distributions.normal.Normal(0., 1.).cdf

#############################################
################################# London et al. generalization bound with softmax policies
#############################################

class LondonSoftmax(ClipSoftmaxPolicy):
    def __init__(self, n_actions, context_dim, tau, N, diag = False, loc_weight=None, delta = 0.05, device = torch.device("cpu")):
        
        super().__init__(n_actions = n_actions, context_dim = context_dim, 
                         tau = tau, N = N, lmbd = 1., diag = diag, loc_weight = loc_weight, device = device)
        
        
        self.delta = delta
        self.unc_kl = np.log((2. * self.N ** 0.5)/self.delta)
    
    def upper_bound(self, x, a, ps, r):
        dist_x_a = self.policy_a(x, a)
        risk_mean = torch.mean(self.compute_risk(dist_x_a, a, ps, r))
        sigma = torch.exp(self.log_scale)
        
        L_sig = torch.exp(-0.5 * sigma ** 2)
        L_2sig = torch.exp(- 2. * sigma ** 2)
        
        biased_risk_mean = L_sig * risk_mean
        kl = (self.normal_kl_div() + self.unc_kl)/self.N
        term = biased_risk_mean + 2. * kl/self.tau + torch.sqrt(2. * (biased_risk_mean + 1./self.tau) * kl / self.tau)
        
        return L_2sig * term
    
    
    def upper_bound_all_dataloader(self, dataloader):
        risk = 0.
        for (x, a, ps, r) in tqdm(dataloader) :
            x, a, ps, r = x.to(self.dev), a.to(self.dev), ps.to(self.dev), r.to(self.dev)
            bsize = x.size(0)
            dist_x_a = self.policy_a(x, a)

            risk += torch.sum(self.compute_risk(dist_x_a, a, ps, r))

        risk_mean = risk/self.N
        
        sigma = torch.exp(self.log_scale)
        
        L_sig = torch.exp(-0.5 * sigma ** 2)
        L_2sig = torch.exp(- 2. * sigma ** 2)
        
        biased_risk_mean = L_sig * risk_mean
        kl = (self.normal_kl_div() + self.unc_kl)/self.N
        term = biased_risk_mean + 2. * kl/self.tau + torch.sqrt((biased_risk_mean + 1./self.tau) * kl / self.tau)
        
        return L_2sig * term
    
        
    def training_step(self, train_batch, batch_idx):
        x, a, ps, r = train_batch
        loss = self.upper_bound(x, a, ps, r)
        return loss    

    
#############################################
################################# London et al. generalization bound with Gaussian policies
#############################################

class LondonGaussian(ClipGaussianPolicy):
    
    def __init__(self, n_actions, context_dim, tau, N, diag = False, loc_weight=None, delta = 0.05, device = torch.device("cpu")):
        
        super().__init__(n_actions = n_actions, context_dim = context_dim, 
                         tau = tau, N = N, lmbd = 1., diag = diag, loc_weight = loc_weight, device = device)
        
        
        self.delta = delta
        self.unc_kl = np.log((2. * self.N ** 0.5)/self.delta)
    
    def upper_bound(self, x, a, ps, r, n_samples = 32):
        dist_x_a = self.policy_a(x, a, n_samples = n_samples)
        risk_mean = torch.mean(self.compute_risk(dist_x_a, a, ps, r))
        biased_risk_mean = risk_mean
        kl = (self.normal_kl_div() + self.unc_kl)/self.N
        term = biased_risk_mean + 2. * kl/self.tau + torch.sqrt(2. * (biased_risk_mean + 1./self.tau) * kl / self.tau)
        
        return term
    
    
    def upper_bound_all_dataloader(self, dataloader, n_samples = 4):
        
        risk = 0.
        for (x, a, ps, r) in tqdm(dataloader) :
            x, a, ps, r = x.to(self.dev), a.to(self.dev), ps.to(self.dev), r.to(self.dev)
            bsize = x.size(0)
            dist_x_a = self.policy_a(x, a, n_samples = n_samples)

            risk += torch.sum(self.compute_risk(dist_x_a, a, ps, r))

        risk_mean = risk/self.N
        
        biased_risk_mean = risk_mean
        kl = (self.normal_kl_div() + self.unc_kl)/self.N
        term = biased_risk_mean + 2. * kl/self.tau + torch.sqrt((biased_risk_mean + 1./self.tau) * kl / self.tau)
        
        return term
    
        
    def training_step(self, train_batch, batch_idx):
        x, a, ps, r = train_batch
        loss = self.upper_bound(x, a, ps, r)
        return loss    
    
#############################################
################################# London et al. generalization bound with MixedLogit policies
#############################################

class LondonMixedLogit(ClipMixedLogitPolicy):
    
    def __init__(self, n_actions, context_dim, tau, N, diag = False, loc_weight=None, delta = 0.05, device = torch.device("cpu")):
        
        super().__init__(n_actions = n_actions, context_dim = context_dim, 
                         tau = tau, N = N, lmbd = 1., diag = diag, loc_weight = loc_weight, device = device)
        
        
        self.delta = delta
        self.unc_kl = np.log((2. * self.N ** 0.5)/self.delta)
    
    def upper_bound(self, x, a, ps, r):
        dist_x_a = self.policy_a(x, a)
        risk_mean = torch.mean(self.compute_risk(dist_x_a, a, ps, r))
        sigma = torch.exp(self.log_scale)
        
        kl = (self.normal_kl_div() + self.unc_kl)/self.N
        term = risk_mean + 2. * kl/self.tau + torch.sqrt(2. * (risk_mean + 1./self.tau) * kl / self.tau)
        
        return term
    
    
    def upper_bound_all_dataloader(self, dataloader):
        
        risk = 0.
        for (x, a, ps, r) in tqdm(dataloader) :
            x, a, ps, r = x.to(self.dev), a.to(self.dev), ps.to(self.dev), r.to(self.dev)
            bsize = x.size(0)
            dist_x_a = self.policy_a(x, a)

            risk += torch.sum(self.compute_risk(dist_x_a, a, ps, r))

        risk_mean = risk/self.N
        
        sigma = torch.exp(self.log_scale)
        
        kl = (self.normal_kl_div() + self.unc_kl)/self.N
        term = risk_mean + 2. * kl/self.tau + torch.sqrt((risk_mean + 1./self.tau) * kl / self.tau)
        
        return term
    
        
    def training_step(self, train_batch, batch_idx):
        x, a, ps, r = train_batch
        loss = self.upper_bound(x, a, ps, r)
        return loss    

#############################################
################################# Sakhi et al. first generalization bound with softmax policies
#############################################

class CatoniSoftmax(ClipSoftmaxPolicy):
    
    def __init__(self, n_actions, context_dim, tau, N, diag = False, loc_weight=None, delta = 0.05, device = torch.device("cpu")):
        
        super().__init__(n_actions = n_actions, context_dim = context_dim, 
                         tau = tau, N = N, lmbd = 1., diag = diag, loc_weight = loc_weight, device = device)
        
        
        self.alpha = torch.nn.Parameter(data = - 4. * torch.ones(()))
        self.delta = delta
        self.unc_kl = np.log((2. * self.N ** 0.5)/self.delta)
    
    def upper_bound(self, x, a, ps, r):
        
        dist_x_a = self.policy_a(x, a)
        risk_mean = torch.mean(self.compute_risk(dist_x_a, a, ps, r))
        
        sigma = torch.exp(self.log_scale)
        L_sig = torch.exp(-0.5 * sigma ** 2)
        L_2sig = torch.exp(- 2. * sigma ** 2)
        
        biased_risk_mean = L_sig * risk_mean
        
        lmbd = torch.nn.functional.softplus(self.alpha)
        kl = (self.normal_kl_div() + self.unc_kl)/self.N
        
        G = self.tau * lmbd * biased_risk_mean + kl
        
        approx_catoni = (1. - torch.exp(-G))/(self.tau * torch.exp(self.alpha))
        
        return L_2sig * approx_catoni
    
    def upper_bound_all_dataloader(self, dataloader):
        
            kl = (self.normal_kl_div() + self.unc_kl)/self.N
            lmbd_0 = torch.nn.functional.softplus(self.alpha).item()

            risk = 0.

            for (x, a, ps, r) in tqdm(dataloader) :

                x, a, ps, r = x.to(self.dev), a.to(self.dev), ps.to(self.dev), r.to(self.dev)
                bsize = x.size(0)
                dist_x_a = self.policy_a(x, a)
                
                risk += torch.sum(self.compute_risk(dist_x_a, a, ps, r))

            risk_mean = risk/self.N
            sigma = torch.exp(self.log_scale)
        
            L_sig = torch.exp(-0.5 * sigma ** 2)
            L_2sig = torch.exp(- 2. * sigma ** 2)

            biased_risk_mean = L_sig * risk_mean

            
            catoni_lmbd = lambda lmbd : L_2sig * (1. - np.exp(- self.tau * lmbd * biased_risk_mean.item() - kl.item()))/(self.tau * (np.exp(lmbd) - 1.))
            catoni_bound = minimize_scalar(catoni_lmbd, lmbd_0, bounds = [1e-10, 6.], method = 'bounded')['fun']

            return catoni_bound
        
    def training_step(self, train_batch, batch_idx):
        
        x, a, ps, r = train_batch
        approx_catoni = self.upper_bound(x, a, ps, r)
        
        return approx_catoni

#############################################
################################# Sakhi et al. first generalization bound with mixed-logit policies
#############################################

class CatoniMixedLogit(ClipMixedLogitPolicy):
    
    def __init__(self, n_actions, context_dim, tau, N, diag = False, loc_weight=None, delta = 0.05, device = torch.device("cpu")):
        
        super().__init__(n_actions = n_actions, context_dim = context_dim, 
                         tau = tau, N = N, lmbd = 1., diag = diag, loc_weight = loc_weight, device = device)
        
        
        self.alpha = torch.nn.Parameter(data = - 4. * torch.ones(()))
        self.delta = delta
        self.unc_kl = np.log((2. * self.N ** 0.5)/self.delta)
    
    def upper_bound(self, x, a, ps, r):
        
        dist_x_a = self.policy_a(x, a)
        risk_mean = torch.mean(self.compute_risk(dist_x_a, a, ps, r))
        
        lmbd = torch.nn.functional.softplus(self.alpha)
        kl = (self.normal_kl_div() + self.unc_kl)/self.N
        
        G = self.tau * lmbd * risk_mean + kl
        
        approx_catoni = (1. - torch.exp(-G))/(self.tau * torch.exp(self.alpha))
        
        return approx_catoni
    
    def upper_bound_all_dataloader(self, dataloader):
        
            kl = (self.normal_kl_div() + self.unc_kl)/self.N
            lmbd_0 = torch.nn.functional.softplus(self.alpha).item()

            risk = 0.

            for (x, a, ps, r) in tqdm(dataloader) :

                x, a, ps, r = x.to(self.dev), a.to(self.dev), ps.to(self.dev), r.to(self.dev)
                bsize = x.size(0)
                dist_x_a = self.policy_a(x, a)
                
                risk += torch.sum(self.compute_risk(dist_x_a, a, ps, r))

            risk_mean = risk/self.N
            
            catoni_lmbd = lambda lmbd : (1. - np.exp(- self.tau * lmbd * risk_mean.item() - kl.item()))/(self.tau * (np.exp(lmbd) - 1.))
            catoni_bound = minimize_scalar(catoni_lmbd, lmbd_0, bounds = [1e-10, 6.], method = 'bounded')['fun']

            return catoni_bound
        
    def training_step(self, train_batch, batch_idx):
        
        x, a, ps, r = train_batch
        approx_catoni = self.upper_bound(x, a, ps, r)
        
        return approx_catoni

#############################################
################################# Sakhi et al. first generalization bound with Gaussian policies
#############################################

class CatoniGaussian(ClipGaussianPolicy):
    
    def __init__(self, n_actions, context_dim, tau, N, diag = False, loc_weight=None, delta = 0.05, device = torch.device("cpu")):
        
        super().__init__(n_actions = n_actions, context_dim = context_dim, 
                         tau = tau, N = N, lmbd = 1., diag = diag, loc_weight = loc_weight, device = device)
        
        
        self.alpha = torch.nn.Parameter(data = - 4. * torch.ones(()))
        self.delta = delta
        self.unc_kl = np.log((2. * self.N ** 0.5)/self.delta)
    
    def upper_bound(self, x, a, ps, r, n_samples = 32):
        
        dist_x_a = self.policy_a(x, a, n_samples = n_samples)
        risk_mean = torch.mean(self.compute_risk(dist_x_a, a, ps, r))
        
        lmbd = torch.nn.functional.softplus(self.alpha)
        kl = (self.normal_kl_div() + self.unc_kl)/self.N
        
        G = self.tau * lmbd * risk_mean + kl
        
        approx_catoni = (1. - torch.exp(-G))/(self.tau * torch.exp(self.alpha))
        
        return approx_catoni
    
    def upper_bound_all_dataloader(self, dataloader, n_samples = 4):
        
            kl = (self.normal_kl_div() + self.unc_kl)/self.N
            lmbd_0 = torch.nn.functional.softplus(self.alpha).item()

            risk = 0.

            for (x, a, ps, r) in tqdm(dataloader) :

                x, a, ps, r = x.to(self.dev), a.to(self.dev), ps.to(self.dev), r.to(self.dev)
                bsize = x.size(0)
                dist_x_a = self.policy_a(x, a, n_samples = n_samples)
                
                risk += torch.sum(self.compute_risk(dist_x_a, a, ps, r))

            risk_mean = risk/self.N
            
            catoni_lmbd = lambda lmbd : (1. - np.exp(- self.tau * lmbd * risk_mean.item() - kl.item()))/(self.tau * (np.exp(lmbd) - 1.))
            catoni_bound = minimize_scalar(catoni_lmbd, lmbd_0, bounds = [1e-10, 6.], method = 'bounded')['fun']

            return catoni_bound
        
    def training_step(self, train_batch, batch_idx):
        
        x, a, ps, r = train_batch
        approx_catoni = self.upper_bound(x, a, ps, r)
        
        return approx_catoni
    
    
#############################################
################################# Sakhi et al. second generalization bound with softmax policies
#############################################

class BernsteinSoftmax(ClipSoftmaxPolicy):
    
    def __init__(self, n_actions, context_dim, tau, N, diag = False, loc_weight=None, delta = 0.05, num_p = 200, xi = 0, rc = 1., 
                 device = torch.device("cpu")):
        
        super().__init__(n_actions = n_actions, context_dim = context_dim, 
                         tau = tau, N = N, lmbd = 1., diag = diag, loc_weight = loc_weight, device = device)
        
        
        self.delta = delta
        self.num_p = num_p
        self.rc = rc
        
        self.unc_kl1 = np.log((4. * self.N ** 0.5)/self.delta)
        self.unc_kl2 = np.log((2. * self.num_p)/self.delta)
        
        self.xi = xi
        self.tau_xi = (1. + xi)/self.tau - xi
        self.xi_coef = np.maximum(self.xi ** 2, (1 + self.xi) ** 2)
        
        lb = (2. * self.tau * np.log(1/self.delta)/(5. * self.rc * self.N * self.xi_coef)) ** 0.5
        ub = 2. * self.rc / self.tau_xi
        print("S_lmbd is defined by these two bounds", lb, ub)
        self.lmbd_cands = np.linspace(lb, ub, num = num_p)
        
        
    def compute_risk(self, dist_x_a, a, ps, r):
        
        bsize = dist_x_a.size(0)
        p_a = ps[torch.arange(bsize), a]
        clipped_p_a = torch.where(p_a < self.tau, self.tau * torch.ones_like(p_a), p_a)
        
        w = dist_x_a/clipped_p_a
        
        risk = (r - self.xi) * w + self.xi
        
        return risk
        
    
    def find_lmbd_from_candidates(self, kl_a, mm):
        
        g_values = g_fun(self.lmbd_cands * self.tau_xi)
        
        values_1 = kl_a/self.lmbd_cands
        values_2 = self.xi_coef * self.lmbd_cands * g_values * mm
        
        to_min = values_1 + values_2
        lmbd_index = np.argmin(to_min)
        
        return self.lmbd_cands[lmbd_index]
    
    
    def upper_bound(self, x, a, ps, r):
        
        bsize = x.size(0)
        dist_x = self.policy(x)
        dist_x_a = dist_x[np.arange(bsize), a]
        
        mm = self.compute_mean_second_moment(dist_x, ps)
        bias = self.compute_bias_term(dist_x, ps)
        risk_mean = torch.mean(self.compute_risk(dist_x_a, a, ps, r))
        
        sigma = torch.exp(self.log_scale)
        L_sig = torch.exp(-0.5 * sigma ** 2)
        L_2sig = torch.exp(- 2. * sigma ** 2)
        
        biased_risk_mean = L_sig * risk_mean

        
        kl_c = ((self.normal_kl_div() + self.unc_kl1)/(2. * self.N)) ** 0.5
        kl_a = (self.normal_kl_div() + self.unc_kl2)/(self.rc * self.N)
        
        lmbd = self.find_lmbd_from_candidates(kl_a.detach().item(), mm.detach().item())
        
        first_part = biased_risk_mean - self.xi * bias + kl_c
        second_part = kl_a/lmbd + self.xi_coef * lmbd * g_fun(lmbd * self.tau_xi / self.rc) * mm / self.rc
        
        bernstein_bound = first_part + second_part
        
        return L_2sig * bernstein_bound
    
    def upper_bound_all_dataloader(self, dataloader):
            
            bandit_size = self.rc * self.N
            kl_c = ((self.normal_kl_div() + self.unc_kl1)/(2. * self.N)) ** 0.5
            kl_a = (self.normal_kl_div() + self.unc_kl2)/(bandit_size)
            
            risk, bias, mm = 0., 0., 0.

            for (x, a, ps, r) in tqdm(dataloader) :

                x, a, ps, r = x.to(self.dev), a.to(self.dev), ps.to(self.dev), r.to(self.dev)
                bsize = x.size(0)
                dist_x = self.policy(x)
                dist_x_a = dist_x[np.arange(bsize), a]

                mm += self.compute_mean_second_moment(dist_x, ps) * bsize
                bias += self.compute_bias_term(dist_x, ps) * bsize
                risk += torch.sum(self.compute_risk(dist_x_a, a, ps, r))


            risk_mean, bias_mean, mm_mean = risk/bandit_size, bias/bandit_size, mm/bandit_size
            print('Risk :', risk_mean)
            print('Bias :', bias_mean)
            print('Second Moment :', mm_mean)
            
            sigma = torch.exp(self.log_scale)
            L_sig = torch.exp(-0.5 * sigma ** 2)
            L_2sig = torch.exp(- 2. * sigma ** 2)

            biased_risk_mean = L_sig * risk_mean


            lmbd = self.find_lmbd_from_candidates(kl_a.item(), mm_mean.item())

            first_part = biased_risk_mean - self.xi * bias_mean + kl_c
            second_part = kl_a/lmbd + self.xi_coef * lmbd * g_fun(lmbd * self.tau_xi / self.rc) * mm_mean / self.rc

            bernstein_bound = first_part + second_part

            return L_2sig * bernstein_bound
        
        
    def training_step(self, train_batch, batch_idx):
        
        x, a, ps, r = train_batch
        bern_bound = self.upper_bound(x, a, ps, r)
        
        return bern_bound 

#############################################
################################# Sakhi et al. second generalization bound with Gaussian policies
#############################################

class BernsteinGaussian(ClipGaussianPolicy):
    
    def __init__(self, n_actions, context_dim, tau, N, diag = False, loc_weight=None, delta = 0.05, num_p = 200, xi = 0, rc = 1., 
                 device = torch.device("cpu")):
        
        super().__init__(n_actions = n_actions, context_dim = context_dim, 
                         tau = tau, N = N, lmbd = 1., diag = diag, loc_weight = loc_weight, device = device)
        
        
        self.delta = delta
        self.num_p = num_p
        self.rc = rc
        
        self.unc_kl1 = np.log((4. * self.N ** 0.5)/self.delta)
        self.unc_kl2 = np.log((2. * self.num_p)/self.delta)
        
        self.xi = xi
        self.tau_xi = (1. + xi)/self.tau - xi
        self.xi_coef = np.maximum(self.xi ** 2, (1 + self.xi) ** 2)
        
        lb = (2. * self.tau * np.log(1/self.delta)/(5. * self.rc * self.N * self.xi_coef)) ** 0.5
        ub = 2. * self.rc / self.tau_xi
        print("S_lmbd is defined by these two bounds", lb, ub)
        self.lmbd_cands = np.linspace(lb, ub, num = num_p)
        
        
    def compute_risk(self, dist_x_a, a, ps, r):
        
        bsize = dist_x_a.size(0)
        p_a = ps[torch.arange(bsize), a]
        clipped_p_a = torch.where(p_a < self.tau, self.tau * torch.ones_like(p_a), p_a)
        
        w = dist_x_a/clipped_p_a
        
        risk = (r - self.xi) * w + self.xi
        
        return risk
        
    
    def find_lmbd_from_candidates(self, kl_a, mm):
        
        g_values = g_fun(self.lmbd_cands * self.tau_xi)
        
        values_1 = kl_a/self.lmbd_cands
        values_2 = self.xi_coef * self.lmbd_cands * g_values * mm
        
        to_min = values_1 + values_2
        lmbd_index = np.argmin(to_min)
        
        return self.lmbd_cands[lmbd_index]
    
    
    def upper_bound(self, x, a, ps, r, n_samples = 32):
        
        bsize = x.size(0)
        dist_x = self.policy(x, n_samples = n_samples)
        dist_x_a = dist_x[np.arange(bsize), a]
        
        mm = self.compute_mean_second_moment(dist_x, ps)
        bias = self.compute_bias_term(dist_x, ps)
        risk_mean = torch.mean(self.compute_risk(dist_x_a, a, ps, r))
        
        kl_c = ((self.normal_kl_div() + self.unc_kl1)/(2. * self.N)) ** 0.5
        kl_a = (self.normal_kl_div() + self.unc_kl2)/(self.rc * self.N)
        
        lmbd = self.find_lmbd_from_candidates(kl_a.detach().item(), mm.detach().item())
        
        first_part = risk_mean - self.xi * bias + kl_c
        second_part = kl_a/lmbd + self.xi_coef * lmbd * g_fun(lmbd * self.tau_xi / self.rc) * mm / self.rc
        
        bernstein_bound = first_part + second_part
        
        return bernstein_bound
    
    def upper_bound_all_dataloader(self, dataloader, n_samples = 4):
            
            bandit_size = self.rc * self.N
            kl_c = ((self.normal_kl_div() + self.unc_kl1)/(2. * self.N)) ** 0.5
            kl_a = (self.normal_kl_div() + self.unc_kl2)/(bandit_size)
            
            risk, bias, mm = 0., 0., 0.

            for (x, a, ps, r) in tqdm(dataloader) :

                x, a, ps, r = x.to(self.dev), a.to(self.dev), ps.to(self.dev), r.to(self.dev)
                bsize = x.size(0)
                dist_x = self.policy(x, n_samples = n_samples)
                dist_x_a = dist_x[np.arange(bsize), a]

                mm += self.compute_mean_second_moment(dist_x, ps) * bsize
                bias += self.compute_bias_term(dist_x, ps) * bsize
                risk += torch.sum(self.compute_risk(dist_x_a, a, ps, r))


            risk_mean, bias_mean, mm_mean = risk/bandit_size, bias/bandit_size, mm/bandit_size
            print('Risk :', risk_mean)
            print('Bias :', bias_mean)
            print('Second Moment :', mm_mean)

            lmbd = self.find_lmbd_from_candidates(kl_a.item(), mm_mean.item())

            first_part = risk_mean - self.xi * bias_mean + kl_c
            second_part = kl_a/lmbd + self.xi_coef * lmbd * g_fun(lmbd * self.tau_xi / self.rc) * mm_mean / self.rc

            bernstein_bound = first_part + second_part

            return bernstein_bound
        
        
    def training_step(self, train_batch, batch_idx):
        
        x, a, ps, r = train_batch
        bern_bound = self.upper_bound(x, a, ps, r)
        
        return bern_bound
    
    
#############################################
################################# Sakhi et al. second generalization bound with mixed-logit policies
#############################################

class BernsteinMixedLogit(ClipMixedLogitPolicy):
    
    def __init__(self, n_actions, context_dim, tau, N, diag = False, loc_weight=None, delta = 0.05, num_p = 200, xi = 0, rc = 1., 
                 device = torch.device("cpu")):
        
        super().__init__(n_actions = n_actions, context_dim = context_dim, 
                         tau = tau, N = N, lmbd = 1., diag = diag, loc_weight = loc_weight, device = device)
        
        
        self.delta = delta
        self.num_p = num_p
        self.rc = rc
        
        self.unc_kl1 = np.log((4. * self.N ** 0.5)/self.delta)
        self.unc_kl2 = np.log((2. * self.num_p)/self.delta)
        
        self.xi = xi
        self.tau_xi = (1. + xi)/self.tau - xi
        self.xi_coef = np.maximum(self.xi ** 2, (1 + self.xi) ** 2)
        
        lb = (2. * self.tau * np.log(1/self.delta)/(5. * self.rc * self.N * self.xi_coef)) ** 0.5
        ub = 2. * self.rc / self.tau_xi
        print("S_lmbd is defined by these two bounds", lb, ub)
        self.lmbd_cands = np.linspace(lb, ub, num = num_p)
                
    
    def find_lmbd_from_candidates(self, kl_a, mm):
        
        g_values = g_fun(self.lmbd_cands * self.tau_xi)
        
        values_1 = kl_a/self.lmbd_cands
        values_2 = self.xi_coef * self.lmbd_cands * g_values * mm
        
        to_min = values_1 + values_2
        lmbd_index = np.argmin(to_min)
        
        return self.lmbd_cands[lmbd_index]
    
    
    def upper_bound(self, x, a, ps, r):
        
        bsize = x.size(0)
        dist_x = self.policy(x)
        dist_x_a = dist_x[np.arange(bsize), a]
        
        mm = self.compute_mean_second_moment(dist_x, ps)
        bias = self.compute_bias_term(dist_x, ps)
        risk_mean = torch.mean(self.compute_risk(dist_x_a, a, ps, r))
        
        kl_c = ((self.normal_kl_div() + self.unc_kl1)/(2. * self.N)) ** 0.5
        kl_a = (self.normal_kl_div() + self.unc_kl2)/(self.rc * self.N)
        
        lmbd = self.find_lmbd_from_candidates(kl_a.detach().item(), mm.detach().item())
        
        first_part = risk_mean - self.xi * bias + kl_c
        second_part = kl_a/lmbd + self.xi_coef * lmbd * g_fun(lmbd * self.tau_xi / self.rc) * mm / self.rc
        
        bernstein_bound = first_part + second_part
        
        return bernstein_bound
    
    def upper_bound_all_dataloader(self, dataloader):
            
            bandit_size = self.rc * self.N
            kl_c = ((self.normal_kl_div() + self.unc_kl1)/(2. * self.N)) ** 0.5
            kl_a = (self.normal_kl_div() + self.unc_kl2)/(bandit_size)
            
            risk, bias, mm = 0., 0., 0.

            for (x, a, ps, r) in tqdm(dataloader) :

                x, a, ps, r = x.to(self.dev), a.to(self.dev), ps.to(self.dev), r.to(self.dev)
                bsize = x.size(0)
                dist_x = self.policy(x)
                dist_x_a = dist_x[np.arange(bsize), a]

                mm += self.compute_mean_second_moment(dist_x, ps) * bsize
                bias += self.compute_bias_term(dist_x, ps) * bsize
                risk += torch.sum(self.compute_risk(dist_x_a, a, ps, r))


            risk_mean, bias_mean, mm_mean = risk/bandit_size, bias/bandit_size, mm/bandit_size
            print('Risk :', risk_mean)
            print('Bias :', bias_mean)
            print('Second Moment :', mm_mean)

            lmbd = self.find_lmbd_from_candidates(kl_a.item(), mm_mean.item())

            first_part = risk_mean - self.xi * bias_mean + kl_c
            second_part = kl_a/lmbd + self.xi_coef * lmbd * g_fun(lmbd * self.tau_xi / self.rc) * mm_mean / self.rc

            bernstein_bound = first_part + second_part

            return bernstein_bound
        
        
    def training_step(self, train_batch, batch_idx):
        
        x, a, ps, r = train_batch
        bern_bound = self.upper_bound(x, a, ps, r)
        
        return bern_bound

    
#############################################
################################# Our generalization bound with softmax policies
#############################################

class OurSoftmax(SmoothSoftmaxPolicy):
    
    def __init__(self, n_actions, context_dim, beta, N, diag = False, loc_weight=None, delta = 0.05, device = torch.device("cpu")):
        
        super().__init__(n_actions = n_actions, context_dim = context_dim, 
                         beta = beta, N = N, lmbd = 1., diag = diag, loc_weight = loc_weight, device = device)
        
        
        self.delta = delta
        self.unc_kl1 = np.log((2. * self.N ** 0.5)/self.delta)
        self.unc_kl2 = np.log(1/self.delta)        

    
    def upper_bound(self, x, a, ps, r):
        
        bsize = x.size(0)
        dist_x = self.policy(x)
        dist_x_a = dist_x[np.arange(bsize), a]
        
        mm1 = self.compute_mean_second_moment(dist_x, ps)
        mm2 = self.compute_empirical_second_moment(dist_x_a, a, ps, r)
        bias_mean = self.compute_mean_bias(dist_x, ps)
        risk_mean = torch.mean(self.compute_risk(dist_x_a, a, ps, r))
        
        sigma = torch.exp(self.log_scale)
        L_sig = torch.exp(-0.5 * sigma ** 2)
        L_2sig = torch.exp(- 2. * sigma ** 2)

        biased_risk_mean = L_sig * risk_mean

        
        kl_c = ((self.normal_kl_div() + self.unc_kl1)/(2. * self.N)) ** 0.5
        kl_a = (self.normal_kl_div() + self.unc_kl2) / self.N
        
        lmbd = 2 * np.sqrt(kl_a.detach().item() * (mm1.detach().item() + mm2.detach().item()))
        
        first_part = biased_risk_mean + kl_c + bias_mean
        second_part = kl_a / lmbd + 0.5 * lmbd * (mm1 + mm2)
        our_bound = first_part + second_part
        
        return L_2sig * our_bound
    
    def upper_bound_all_dataloader(self, dataloader):
            
            kl_c = ((self.normal_kl_div() + self.unc_kl1)/(2. * self.N)) ** 0.5
            kl_a = (self.normal_kl_div() + self.unc_kl2) / self.N

            
            risk, bias, mm1, mm2, bias_ = 0., 0., 0., 0., 0.

            for (x, a, ps, r) in tqdm(dataloader) :

                x, a, ps, r = x.to(self.dev), a.to(self.dev), ps.to(self.dev), r.to(self.dev)
                bsize = x.size(0)
                dist_x = self.policy(x)
                dist_x_a = dist_x[np.arange(bsize), a]

                mm1 += self.compute_mean_second_moment(dist_x, ps) * bsize
                mm2 += self.compute_empirical_second_moment(dist_x_a, a, ps, r) * bsize
                bias_ += self.compute_mean_bias(dist_x, ps) * bsize
                risk += torch.sum(self.compute_risk(dist_x_a, a, ps, r))


            risk_mean, mm1_mean, mm2_mean, bias_mean = risk/self.N, mm1/self.N, mm2/self.N, bias_/self.N
            
            sigma = torch.exp(self.log_scale)
            L_sig = torch.exp(-0.5 * sigma ** 2)
            L_2sig = torch.exp(- 2. * sigma ** 2)

            biased_risk_mean = L_sig * risk_mean

            print('Risk :', risk_mean)
            print('Bias:', bias_mean)
            print('Theoretical Second Moment :', mm1_mean)
            print('Empirical Second Moment :', mm2_mean)

            lmbd = 2 * np.sqrt(kl_a.detach().item() * (mm1_mean.detach().item() + mm2_mean.detach().item()))

            first_part = biased_risk_mean + kl_c + bias_mean
            second_part = kl_a / lmbd + 0.5 * lmbd * (mm1_mean + mm2_mean)

            our_bound = first_part + second_part

            return L_2sig * our_bound
    
        
    def training_step(self, train_batch, batch_idx):
        x, a, ps, r = train_batch
        loss = self.upper_bound(x, a, ps, r)
        return loss    

    
#############################################
################################# London et al. generalization bound with Gaussian policies
#############################################

class OurGaussian(SmoothGaussianPolicy):
    
    def __init__(self, n_actions, context_dim, beta, N, diag = False, loc_weight=None, delta = 0.05, device = torch.device("cpu")):
        
        super().__init__(n_actions = n_actions, context_dim = context_dim, 
                         beta = beta, N = N, lmbd = 1., diag = diag, loc_weight = loc_weight, device = device)
        
        
        self.delta = delta
        self.unc_kl1 = np.log((4. * self.N ** 0.5)/self.delta)
        self.unc_kl2 = np.log(4/self.delta)
    
    def upper_bound(self, x, a, ps, r, n_samples = 32):
        
        bsize = x.size(0)
        dist_x = self.policy(x, n_samples = n_samples)
        dist_x_a = dist_x[np.arange(bsize), a]
        
        mm1 = self.compute_mean_second_moment(dist_x, ps)
        mm2 = self.compute_empirical_second_moment(dist_x_a, a, ps, r)
        bias_mean = self.compute_mean_bias(dist_x, ps)
        risk_mean = torch.mean(self.compute_risk(dist_x_a, a, ps, r))
        
        kl_c = ((self.normal_kl_div() + self.unc_kl1)/(2. * self.N)) ** 0.5
        kl_a = (self.normal_kl_div() + self.unc_kl2) / self.N
        
        lmbd = 2 * np.sqrt(kl_a.detach().item() * (mm1.detach().item() + mm2.detach().item()))
        
        first_part = risk_mean + kl_c + bias_mean
        second_part = kl_a / lmbd + 0.5 * lmbd * (mm1 + mm2)
        our_bound = first_part + second_part
        
        return our_bound
    
    def upper_bound_all_dataloader(self, dataloader, n_samples = 4):
            
            kl_c = ((self.normal_kl_div() + self.unc_kl1)/(2. * self.N)) ** 0.5
            kl_a = (self.normal_kl_div() + self.unc_kl2) / self.N

            
            risk, bias, mm1, mm2, bias_ = 0., 0., 0., 0., 0.

            for (x, a, ps, r) in tqdm(dataloader) :

                x, a, ps, r = x.to(self.dev), a.to(self.dev), ps.to(self.dev), r.to(self.dev)
                bsize = x.size(0)
                dist_x = self.policy(x, n_samples = n_samples)
                dist_x_a = dist_x[np.arange(bsize), a]

                mm1 += self.compute_mean_second_moment(dist_x, ps) * bsize
                mm2 += self.compute_empirical_second_moment(dist_x_a, a, ps, r) * bsize
                bias_ += self.compute_mean_bias(dist_x, ps) * bsize
                risk += torch.sum(self.compute_risk(dist_x_a, a, ps, r))


            risk_mean, mm1_mean, mm2_mean, bias_mean = risk/self.N, mm1/self.N, mm2/self.N, bias_/self.N
            
            print('Risk :', risk_mean)
            print('Bias:', bias_mean)
            print('Theoretical Second Moment :', mm1_mean)
            print('Empirical Second Moment :', mm2_mean)

            lmbd = 2 * np.sqrt(kl_a.detach().item() * (mm1_mean.detach().item() + mm2_mean.detach().item()))

            first_part = risk_mean + kl_c + bias_mean
            second_part = kl_a / lmbd + 0.5 * lmbd * (mm1_mean + mm2_mean)

            our_bound = first_part + second_part

            return our_bound
    
        
    def training_step(self, train_batch, batch_idx):
        x, a, ps, r = train_batch
        loss = self.upper_bound(x, a, ps, r)
        return loss    
    
    
    
#############################################
################################# Our generalization bound with Gaussian policies
#############################################

class OurMixedLogit(SmoothMixedLogitPolicy):
    
    def __init__(self, n_actions, context_dim, beta, N, diag = False, loc_weight=None, delta = 0.05, device = torch.device("cpu")):
        
        super().__init__(n_actions = n_actions, context_dim = context_dim, 
                         beta = beta, N = N, lmbd = 1., diag = diag, loc_weight = loc_weight, device = device)
        
        
        self.delta = delta
        self.unc_kl1 = np.log((4. * self.N ** 0.5)/self.delta)
        self.unc_kl2 = np.log(4/self.delta)
    
    def upper_bound(self, x, a, ps, r):
        
        bsize = x.size(0)
        dist_x = self.policy(x)
        dist_x_a = dist_x[np.arange(bsize), a]
        
        mm1 = self.compute_mean_second_moment(dist_x, ps)
        mm2 = self.compute_empirical_second_moment(dist_x_a, a, ps, r)
        bias_mean = self.compute_mean_bias(dist_x, ps)
        risk_mean = torch.mean(self.compute_risk(dist_x_a, a, ps, r))
        
        kl_c = ((self.normal_kl_div() + self.unc_kl1)/(2. * self.N)) ** 0.5
        kl_a = (self.normal_kl_div() + self.unc_kl2) / self.N
        
        lmbd = 2 * np.sqrt(kl_a.detach().item() * (mm1.detach().item() + mm2.detach().item()))
        
        first_part = risk_mean + kl_c + bias_mean
        second_part = kl_a / lmbd + 0.5 * lmbd * (mm1 + mm2)
        our_bound = first_part + second_part
        
        return our_bound
    
    def upper_bound_all_dataloader(self, dataloader):
            
            kl_c = ((self.normal_kl_div() + self.unc_kl1)/(2. * self.N)) ** 0.5
            kl_a = (self.normal_kl_div() + self.unc_kl2) / self.N

            
            risk, bias, mm1, mm2, bias_ = 0., 0., 0., 0., 0.

            for (x, a, ps, r) in tqdm(dataloader) :

                x, a, ps, r = x.to(self.dev), a.to(self.dev), ps.to(self.dev), r.to(self.dev)
                bsize = x.size(0)
                dist_x = self.policy(x)
                dist_x_a = dist_x[np.arange(bsize), a]

                mm1 += self.compute_mean_second_moment(dist_x, ps) * bsize
                mm2 += self.compute_empirical_second_moment(dist_x_a, a, ps, r) * bsize
                bias_ += self.compute_mean_bias(dist_x, ps) * bsize
                risk += torch.sum(self.compute_risk(dist_x_a, a, ps, r))


            risk_mean, mm1_mean, mm2_mean, bias_mean = risk/self.N, mm1/self.N, mm2/self.N, bias_/self.N
            
            print('Risk :', risk_mean)
            print('Bias:', bias_mean)
            print('Theoretical Second Moment :', mm1_mean)
            print('Empirical Second Moment :', mm2_mean)

            lmbd = 2 * np.sqrt(kl_a.detach().item() * (mm1_mean.detach().item() + mm2_mean.detach().item()))

            first_part = risk_mean + kl_c + bias_mean
            second_part = kl_a / lmbd + 0.5 * lmbd * (mm1_mean + mm2_mean)

            our_bound = first_part + second_part

            return our_bound
    
        
    def training_step(self, train_batch, batch_idx):
        x, a, ps, r = train_batch
        loss = self.upper_bound(x, a, ps, r)
        return loss