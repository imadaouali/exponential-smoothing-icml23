import torch
from torch import nn, optim
import pytorch_lightning as pl
import numpy as np
from tqdm import tqdm
from sklearn import datasets, preprocessing
from scipy.optimize import minimize_scalar, minimize
from .utils import *

probit = torch.distributions.normal.Normal(0., 1.).cdf

##############################
######################## supervised policy to train the logging policy
#############################

class SupervisedPolicy(pl.LightningModule):
    def __init__(self, n_actions, context_dim, reg, softmax = False, multilabel = False, device = torch.device("cpu")):
        super().__init__()
        self.linear = nn.Linear(context_dim, n_actions, bias=False).to(device)
        self.dev = device
        self.loss_fun = torch.nn.CrossEntropyLoss()
        self.reg = reg
        self.a = n_actions
        self.mask = torch.eye(self.a, dtype=bool).view(1, self.a, self.a, 1)
        self.softmax = softmax
        self.alpha = 1.
        self.multilabel = multilabel
        self.logsoftmax = nn.LogSoftmax(dim = -1)
        
    def policy_a(self, x, a, n_samples = 32):
        
        bs = x.size(0)
        
        scores = self.alpha * self.linear(x)
        
        if self.softmax :
            probs = torch.softmax(scores, dim = 1)
            return probs[torch.arange(bs), a]
        
        scores_a = scores[torch.arange(bs), a].unsqueeze(-1)
        
        diff = scores_a - scores
        
        indices = torch.ones_like(diff).scatter_(1, a.unsqueeze(1), 0.).bool()
        diffs_masked = diff[indices].reshape(bs, self.a - 1, 1)
        
        eps = torch.randn(bs, 1, n_samples)
        diffs_stoch = eps + diffs_masked 
        
        dist_x_a = torch.mean(torch.prod(probit(diffs_stoch), dim = -2), dim = -1)
        
        return dist_x_a
    
    
    def policy(self, x, n_samples = 32):

        bs = x.size(0)
        scores = self.alpha * self.linear(x)
        
        if self.softmax :
            probs = torch.softmax(scores, dim = 1)
            return probs
        
        eps = torch.randn(bs, 1, 1, n_samples)
        diffs = (scores.unsqueeze(-1) - scores.unsqueeze(1)).unsqueeze(-1)
        diffs_masked = diffs.masked_select(~self.mask).view(bs, self.a, self.a - 1, 1)
        
        diffs_stoch = eps + diffs_masked 
        
        dist_x = torch.mean(torch.prod(probit(diffs_stoch), dim = -2), dim = -1)
        
        return dist_x
    
    def forward(self, x):
        dist_x = self.policy(x, n_samples = 1024)
        return dist_x
    

    def sample(self, x):
        scores = self.alpha * self.linear(x)
        eps = torch.randn_like(scores) if not self.softmax else -torch.log(-torch.log(torch.rand_like(scores)))
        return torch.argmax(scores + eps, dim = 1)

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=.1, weight_decay=self.reg)
        return optimizer

    def training_step(self, train_batch, batch_idx):
        x, y = train_batch
        x = x.view(x.size(0), -1)
        if self.multilabel : 
            logsoftmax = self.logsoftmax(self.linear(x))
            loss = - torch.mean(y * logsoftmax)
        else : 
            loss = self.loss_fun(self.linear(x), y)
        return loss    
    
    
##############################
######################## Gaussian policy with hard clipping
#############################

class ClipGaussianPolicy(pl.LightningModule):
    
    def __init__(self, n_actions, context_dim, tau, N, lmbd, diag = False, loc_weight=None, device = torch.device("cpu")):
        super().__init__()
        
        if loc_weight is not None: 
            cloned_loc = torch.clone(loc_weight)
            self.q_mean = nn.Parameter(data = cloned_loc)
        else : 
            self.q_mean = nn.Parameter(data = 0.01 * torch.randn(context_dim, n_actions))
        
        if diag : 
            self.q_log_sigma = nn.Parameter(data = torch.zeros_like(self.q_mean))
        else :
            self.log_scale = nn.Parameter(data = torch.zeros(()))
        
        self.diag = diag
        self.tau = tau
        self.d = context_dim
        self.a = n_actions
        self.lmbd = lmbd
        self.mu_0 = torch.clone(self.q_mean.data).to(device)
        self.N = N
        self.dev = device
        
    def policy_a(self, x, a, n_samples = 32):
        
        bs = x.size(0)
        helper = torch.arange(bs)
        
        if self.diag:
            normalizer = torch.matmul(x**2, torch.exp(2. * self.q_log_sigma).T) ** .5
        else :
            normalizer = torch.ones([bs, self.a]).to(self.dev) * torch.exp(self.log_scale)
        
        normalizer_a = normalizer[helper, a].unsqueeze(-1)
        
        scores = torch.matmul(x, self.q_mean.T)
        scores_a = scores[helper, a].unsqueeze(-1)
        
        diff = (scores_a - scores).unsqueeze(-1)
        sigma_eps = torch.randn(bs, 1, n_samples).to(self.dev) * normalizer_a.view(bs, 1, 1)
        
        diffs_stoch = (sigma_eps + diff)/(normalizer.unsqueeze(-1))
        
        indices = torch.ones_like(diffs_stoch, dtype = bool)
        indices[helper, a] = False
        
        diffs_masked = diffs_stoch[indices].reshape(bs, self.a - 1, n_samples)
        
        dist_x_a = torch.mean(torch.prod(probit(diffs_masked), dim = -2), dim = -1)
        
        return dist_x_a

    def policy(self, x, n_samples = 32):
        
        bs = x.size(0)
        if self.diag:
            normalizer = torch.matmul(x**2, torch.exp(2. * self.q_log_sigma).T) ** .5
        else :
            normalizer = torch.ones([bs, self.a]).to(self.dev) * torch.exp(self.log_scale)
        
        scores = torch.matmul(x, self.q_mean.T)
        
        sigma_eps = torch.randn(bs, 1, 1, n_samples).to(self.dev) * normalizer.view(bs, self.a, 1, 1)
        
        diffs = (scores.unsqueeze(-1) - scores.unsqueeze(1)).unsqueeze(-1)

        diffs_stoch = (sigma_eps + diffs)/normalizer.view(bs, self.a, 1, 1)
        
        prob_diffs = probit(diffs_stoch)
        prob_diffs.diagonal(dim1=1, dim2=2).fill_(1.)
        dist_x = torch.mean(torch.prod(prob_diffs, dim = -2), dim = -1)
        
        return dist_x

    def normal_kl_div(self):
        if self.diag :
            v_part = torch.sum(torch.exp(2. * self.q_log_sigma) - 2. * self.q_log_sigma - 1.)
        else : 
            v_part = self.a * self.d * (torch.exp(2. * self.log_scale) - 2. * self.log_scale - 1.)
            
        m_part = torch.sum((self.q_mean - self.mu_0) ** 2)
        kl_div = 0.5 * (v_part + m_part)
        return kl_div
    
    
    def compute_mean_second_moment(self, dist_x, ps):
        clipped_ps = torch.where(ps < self.tau, self.tau * torch.ones_like(ps), ps)
        
        sc_moment = (ps * dist_x)/(clipped_ps ** 2)
        second_moment = torch.sum(sc_moment, dim = 1)
        
        return torch.mean(second_moment)

    def compute_bias_term(self, dist_x, ps):
        
        bias_vector = torch.where(ps < self.tau, 1 - ps/self.tau, torch.zeros_like(ps))
        return torch.mean(torch.sum(dist_x * bias_vector, dim = 1))
         

    def forward(self, x):
        dist_x = self.policy(x, n_samples = 512)
        return dist_x


    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-3)
        return optimizer
    
    def compute_risk(self, dist_x_a, a, ps, r):
        
        bsize = dist_x_a.size(0)
        p_a = ps[torch.arange(bsize), a]
        clipped_p_a = torch.where(p_a < self.tau, self.tau * torch.ones_like(p_a), p_a)
        
        w = dist_x_a/clipped_p_a
        
        risk = r * w 
        
        return risk
    
##############################
######################## Softmax policy with hard clipping
#############################

class ClipSoftmaxPolicy(pl.LightningModule):
    
    def __init__(self, n_actions, context_dim, tau, N, lmbd, diag = False, loc_weight=None, device = torch.device("cpu")):
        super().__init__()
        
        if loc_weight is not None: 
            cloned_loc = torch.clone(loc_weight)
            self.q_mean = nn.Parameter(data = cloned_loc)
        else : 
            self.q_mean = nn.Parameter(data = 0.01 * torch.randn(context_dim, n_actions))
        
        if diag : 
            self.q_log_sigma = nn.Parameter(data = torch.zeros_like(self.q_mean))
        else :
            self.log_scale = nn.Parameter(data = torch.zeros(()))
        
        self.diag = diag
        self.tau = tau
        self.d = context_dim
        self.a = n_actions
        self.lmbd = lmbd
        self.mu_0 = torch.clone(self.q_mean.data).to(device)
        self.N = N
        self.dev = device
        
    def policy_a(self, x, a, n_samples = 1):
        bs = x.size(0)
        helper = torch.arange(bs)
        scores = torch.matmul(x, self.q_mean.T)
        probs = torch.softmax(scores, dim = 1)
        return probs[helper, a]

    def policy(self, x, n_samples = 1):
        bs = x.size(0)
        scores = torch.matmul(x, self.q_mean.T)
        probs = torch.softmax(scores, dim = 1)
        return probs

    def normal_kl_div(self):
        if self.diag :
            v_part = torch.sum(torch.exp(2. * self.q_log_sigma) - 2. * self.q_log_sigma - 1.)
        else : 
            v_part = self.a * self.d * (torch.exp(2. * self.log_scale) - 2. * self.log_scale - 1.)
            
        m_part = torch.sum((self.q_mean - self.mu_0) ** 2)
        kl_div = 0.5 * (v_part + m_part)
        return kl_div
    
    def compute_mean_second_moment(self, dist_x, ps):
        clipped_ps = torch.where(ps < self.tau, self.tau * torch.ones_like(ps), ps)
        
        sc_moment = (ps * dist_x)/(clipped_ps ** 2)
        second_moment = torch.sum(sc_moment, dim = 1)
        return torch.mean(second_moment)

    def compute_bias_term(self, dist_x, ps):
        
        bias_vector = torch.where(ps < self.tau, 1 - ps/self.tau, torch.zeros_like(ps))
        return torch.mean(torch.sum(dist_x * bias_vector, dim = 1))

    def forward(self, x):
        dist_x = self.policy(x)
        return dist_x


    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-3)
        return optimizer
    
    def compute_risk(self, dist_x_a, a, ps, r):
        
        bsize = dist_x_a.size(0)
        p_a = ps[torch.arange(bsize), a]
        clipped_p_a = torch.where(p_a < self.tau, self.tau * torch.ones_like(p_a), p_a)
        w = dist_x_a/clipped_p_a
        risk = r * w 
        return risk


##############################
######################## MixedLogit policy with hard clipping
#############################
    
    
class ClipMixedLogitPolicy(pl.LightningModule):
    
    def __init__(self, n_actions, context_dim, tau, N, lmbd, diag = False, loc_weight=None, device = torch.device("cpu")):
        super().__init__()
        
        if loc_weight is not None: 
            cloned_loc = torch.clone(loc_weight)
            self.q_mean = nn.Parameter(data = cloned_loc)
        else : 
            self.q_mean = nn.Parameter(data = 0.01 * torch.randn(context_dim, n_actions))
        
        if diag : 
            self.q_log_sigma = nn.Parameter(data = torch.zeros_like(self.q_mean))
        else :
            self.log_scale = nn.Parameter(data = torch.zeros(()))
        
        self.diag = diag
        self.tau = tau
        self.d = context_dim
        self.a = n_actions
        self.lmbd = lmbd
        self.mu_0 = torch.clone(self.q_mean.data).to(device)
        self.N = N
        self.dev = device
        
    def policy_a(self, x, a, n_samples = 32):
        bs = x.size(0)
        helper = torch.arange(bs)
        sigma = torch.exp(self.log_scale)
        
        scores = torch.matmul(x, self.q_mean.T).unsqueeze(-1)
        scores_noised = scores + sigma * torch.randn(bs, 1, n_samples).to(self.dev) #torch.randn_like(scores)
        probs = torch.mean(torch.softmax(scores_noised, dim = 1), dim = -1)
        return probs[helper, a]

    def policy(self, x, n_samples = 32):
        bs = x.size(0)
        sigma = torch.exp(self.log_scale)
        scores = torch.matmul(x, self.q_mean.T).unsqueeze(-1)
        scores_noised = scores + sigma * torch.randn(bs, 1, n_samples).to(self.dev) #torch.randn_like(scores)
        probs = torch.mean(torch.softmax(scores_noised, dim = 1), dim = -1)
        return probs

    def normal_kl_div(self):
        if self.diag :
            v_part = torch.sum(torch.exp(2. * self.q_log_sigma) - 2. * self.q_log_sigma - 1.)
        else : 
            v_part = self.a * self.d * (torch.exp(2. * self.log_scale) - 2. * self.log_scale - 1.)
            
        m_part = torch.sum((self.q_mean - self.mu_0) ** 2)
        kl_div = 0.5 * (v_part + m_part)
        return kl_div
    
    def compute_mean_second_moment(self, dist_x, ps):
        clipped_ps = torch.where(ps < self.tau, self.tau * torch.ones_like(ps), ps)
        
        sc_moment = (ps * dist_x)/(clipped_ps ** 2)
        second_moment = torch.sum(sc_moment, dim = 1)
        return torch.mean(second_moment)

    def compute_bias_term(self, dist_x, ps):
        
        bias_vector = torch.where(ps < self.tau, 1 - ps/self.tau, torch.zeros_like(ps))
        return torch.mean(torch.sum(dist_x * bias_vector, dim = 1))

    def forward(self, x):
        dist_x = self.policy(x)
        return dist_x


    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-3)
        return optimizer
    
    def compute_risk(self, dist_x_a, a, ps, r):
        
        bsize = dist_x_a.size(0)
        p_a = ps[torch.arange(bsize), a]
        clipped_p_a = torch.where(p_a < self.tau, self.tau * torch.ones_like(p_a), p_a)
        w = dist_x_a/clipped_p_a
        risk = r * w 
        return risk


##############################
######################## Gaussian policy with exp smoothing
#############################

class SmoothGaussianPolicy(pl.LightningModule):
    
    def __init__(self, n_actions, context_dim, N, lmbd, beta=1, diag = False, loc_weight=None, device = torch.device("cpu")):
        super().__init__()
        
        if loc_weight is not None: 
            cloned_loc = torch.clone(loc_weight)
            self.q_mean = nn.Parameter(data = cloned_loc)
        else : 
            self.q_mean = nn.Parameter(data = 0.01 * torch.randn(context_dim, n_actions))
        
        if diag : 
            self.q_log_sigma = nn.Parameter(data = torch.zeros_like(self.q_mean))
        else :
            self.log_scale = nn.Parameter(data = torch.zeros(()))
        
        self.diag = diag
        self.d = context_dim
        self.a = n_actions
        self.lmbd = lmbd
        self.beta = beta
        self.mu_0 = torch.clone(self.q_mean.data).to(device)
        self.N = N
        self.dev = device
        
    def policy_a(self, x, a, n_samples = 32):
        
        bs = x.size(0)
        helper = torch.arange(bs)
        
        if self.diag:
            normalizer = torch.matmul(x**2, torch.exp(2. * self.q_log_sigma).T) ** .5
        else :
            normalizer = torch.ones([bs, self.a]).to(self.dev) * torch.exp(self.log_scale)
        
        normalizer_a = normalizer[helper, a].unsqueeze(-1)
        
        scores = torch.matmul(x, self.q_mean.T)
        scores_a = scores[helper, a].unsqueeze(-1)
        
        diff = (scores_a - scores).unsqueeze(-1)
        sigma_eps = torch.randn(bs, 1, n_samples).to(self.dev) * normalizer_a.view(bs, 1, 1)
        
        diffs_stoch = (sigma_eps + diff)/(normalizer.unsqueeze(-1))
        
        indices = torch.ones_like(diffs_stoch, dtype = bool)
        indices[helper, a] = False
        
        diffs_masked = diffs_stoch[indices].reshape(bs, self.a - 1, n_samples)
        
        dist_x_a = torch.mean(torch.prod(probit(diffs_masked), dim = -2), dim = -1)
        
        return dist_x_a

    def policy(self, x, n_samples = 32):
        
        bs = x.size(0)
        
        if self.diag:
            normalizer = torch.matmul(x**2, torch.exp(2. * self.q_log_sigma).T) ** .5
        else :
            normalizer = torch.ones([bs, self.a]).to(self.dev) * torch.exp(self.log_scale)
        
        scores = torch.matmul(x, self.q_mean.T)
        
        sigma_eps = torch.randn(bs, 1, 1, n_samples).to(self.dev) * normalizer.view(bs, self.a, 1, 1)
        
        diffs = (scores.unsqueeze(-1) - scores.unsqueeze(1)).unsqueeze(-1)

        diffs_stoch = (sigma_eps + diffs)/normalizer.view(bs, self.a, 1, 1)
        
        prob_diffs = probit(diffs_stoch)
        prob_diffs.diagonal(dim1=1, dim2=2).fill_(1.)
        dist_x = torch.mean(torch.prod(prob_diffs, dim = -2), dim = -1)
        
        return dist_x

    def normal_kl_div(self):
        if self.diag :
            v_part = torch.sum(torch.exp(2. * self.q_log_sigma) - 2. * self.q_log_sigma - 1.)
        else : 
            v_part = self.a * self.d * (torch.exp(2. * self.log_scale) - 2. * self.log_scale - 1.)
        m_part = torch.sum((self.q_mean - self.mu_0) ** 2)
        kl_div = 0.5 * (v_part + m_part)
        return kl_div
    
    def compute_mean_bias(self, dist_x, ps):
        term_ = dist_x * (ps**(1 - self.beta))
        return 1 - torch.mean(torch.sum(term_, dim = 1))
    
    def compute_mean_second_moment(self, dist_x, ps):
        sc_moment = dist_x / (ps**(2 * self.beta-1))
        second_moment = torch.sum(sc_moment, dim = 1)
        return torch.mean(second_moment)
    
    def compute_empirical_second_moment(self, dist_x_a, a, ps, r):
        bsize = dist_x_a.size(0)
        p_a = ps[torch.arange(bsize), a]
        sc_moment = dist_x_a / (p_a**(2*self.beta))
        return torch.mean((r**2) * sc_moment)         

    def forward(self, x):
        dist_x = self.policy(x, n_samples = 512)
        return dist_x

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-3)
        return optimizer
    
    def compute_risk(self, dist_x_a, a, ps, r):
        bsize = dist_x_a.size(0)
        p_a = ps[torch.arange(bsize), a]
        w = dist_x_a / (p_a**self.beta)
        risk = r * w 
        return risk
    

##############################
######################## Softmax policy with exp smoothing
#############################


class SmoothSoftmaxPolicy(pl.LightningModule):
    
    def __init__(self, n_actions, context_dim, N, lmbd, beta=1, diag = False, loc_weight=None, device = torch.device("cpu")):
        super().__init__()
        
        if loc_weight is not None: 
            cloned_loc = torch.clone(loc_weight)
            self.q_mean = nn.Parameter(data = cloned_loc)
        else : 
            self.q_mean = nn.Parameter(data = 0.01 * torch.randn(context_dim, n_actions))
        
        if diag : 
            self.q_log_sigma = nn.Parameter(data = torch.zeros_like(self.q_mean))
        else :
            self.log_scale = nn.Parameter(data = torch.zeros(()))
        
        self.diag = diag
        self.d = context_dim
        self.a = n_actions
        self.lmbd = lmbd
        self.beta = beta
        self.mu_0 = torch.clone(self.q_mean.data).to(device)
        self.N = N
        self.dev = device
        
    def policy_a(self, x, a, n_samples = 1):
        bs = x.size(0)
        helper = torch.arange(bs)
        scores = torch.matmul(x, self.q_mean.T)
        probs = torch.softmax(scores, dim = 1)
        return probs[helper, a]

    def policy(self, x, n_samples = 1):
        bs = x.size(0)
        scores = torch.matmul(x, self.q_mean.T)
        probs = torch.softmax(scores, dim = 1)
        return probs

    def normal_kl_div(self):
        if self.diag :
            v_part = torch.sum(torch.exp(2. * self.q_log_sigma) - 2. * self.q_log_sigma - 1.)
        else : 
            v_part = self.a * self.d * (torch.exp(2. * self.log_scale) - 2. * self.log_scale - 1.)
        m_part = torch.sum((self.q_mean - self.mu_0) ** 2)
        kl_div = 0.5 * (v_part + m_part)
        return kl_div
    
    def compute_mean_bias(self, dist_x, ps):
        term_ = dist_x * (ps**(1 - self.beta))
        return 1 - torch.mean(torch.sum(term_, dim = 1))
    
    def compute_mean_second_moment(self, dist_x, ps):
        sc_moment = dist_x / (ps**(2 * self.beta-1))
        second_moment = torch.sum(sc_moment, dim = 1)
        return torch.mean(second_moment)
    
    def compute_empirical_second_moment(self, dist_x_a, a, ps, r):
        bsize = dist_x_a.size(0)
        p_a = ps[torch.arange(bsize), a]
        sc_moment = dist_x_a / (p_a**(2*self.beta))
        return torch.mean((r**2) * sc_moment)         

    def forward(self, x):
        dist_x = self.policy(x)
        return dist_x

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-3)
        return optimizer
    
    def compute_risk(self, dist_x_a, a, ps, r):
        bsize = dist_x_a.size(0)
        p_a = ps[torch.arange(bsize), a]
        w = dist_x_a / (p_a**self.beta)
        risk = r * w 
        return risk    
    
##############################
######################## MixedLogit policy with exp smoothing
#############################


class SmoothMixedLogitPolicy(pl.LightningModule):
    
    def __init__(self, n_actions, context_dim, N, lmbd, beta=1, diag = False, loc_weight=None, device = torch.device("cpu")):
        super().__init__()
        
        if loc_weight is not None: 
            cloned_loc = torch.clone(loc_weight)
            self.q_mean = nn.Parameter(data = cloned_loc)
        else : 
            self.q_mean = nn.Parameter(data = 0.01 * torch.randn(context_dim, n_actions))
        
        if diag : 
            self.q_log_sigma = nn.Parameter(data = torch.zeros_like(self.q_mean))
        else :
            self.log_scale = nn.Parameter(data = torch.zeros(()))
        
        self.diag = diag
        self.d = context_dim
        self.a = n_actions
        self.lmbd = lmbd
        self.beta = beta
        self.mu_0 = torch.clone(self.q_mean.data).to(device)
        self.N = N
        self.dev = device
        
    def policy_a(self, x, a, n_samples = 32):
        bs = x.size(0)
        helper = torch.arange(bs)
        sigma = torch.exp(self.log_scale)
        
        scores = torch.matmul(x, self.q_mean.T).unsqueeze(-1)
        scores_noised = scores + sigma * torch.randn(bs, 1, n_samples).to(self.dev) #torch.randn_like(scores)
        probs = torch.mean(torch.softmax(scores_noised, dim = 1), dim = -1)
        return probs[helper, a]

    def policy(self, x, n_samples = 32):
        bs = x.size(0)
        sigma = torch.exp(self.log_scale)
        scores = torch.matmul(x, self.q_mean.T).unsqueeze(-1)
        scores_noised = scores + sigma * torch.randn(bs, 1, n_samples).to(self.dev)#torch.randn_like(scores)
        probs = torch.mean(torch.softmax(scores_noised, dim = 1), dim = -1)
        return probs

    def normal_kl_div(self):
        if self.diag :
            v_part = torch.sum(torch.exp(2. * self.q_log_sigma) - 2. * self.q_log_sigma - 1.)
        else : 
            v_part = self.a * self.d * (torch.exp(2. * self.log_scale) - 2. * self.log_scale - 1.)
        m_part = torch.sum((self.q_mean - self.mu_0) ** 2)
        kl_div = 0.5 * (v_part + m_part)
        return kl_div
    
    def compute_mean_bias(self, dist_x, ps):
        term_ = dist_x * (ps**(1 - self.beta))
        return 1 - torch.mean(torch.sum(term_, dim = 1))
    
    def compute_mean_second_moment(self, dist_x, ps):
        sc_moment = dist_x / (ps**(2 * self.beta-1))
        second_moment = torch.sum(sc_moment, dim = 1)
        return torch.mean(second_moment)
    
    def compute_empirical_second_moment(self, dist_x_a, a, ps, r):
        bsize = dist_x_a.size(0)
        p_a = ps[torch.arange(bsize), a]
        sc_moment = dist_x_a / (p_a**(2*self.beta))
        return torch.mean((r**2) * sc_moment)         

    def forward(self, x):
        dist_x = self.policy(x)
        return dist_x

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-3)
        return optimizer
    
    def compute_risk(self, dist_x_a, a, ps, r):
        bsize = dist_x_a.size(0)
        p_a = ps[torch.arange(bsize), a]
        w = dist_x_a / (p_a**self.beta)
        risk = r * w 
        return risk