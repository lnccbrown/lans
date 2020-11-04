import numpy as np
import scipy as scp
from scipy import special
import pandas as pd
import os
import shutil

class ddm_data_simulator():
    def __init__(self):
        self.model_params = dict({'mu': 0,
                                  'sigma_2': 1,
                                  'theta': 1})
        self.bernoulli_p = 'will be initiated upon a call to the make_data() function'
        self.sample_params = dict({'n_samples': 10000})
        self.mu = (self.model_params['mu'] * self.model_params['theta']) / self.model_params['sigma_2']
        self.mu_tilde = 1
        self.t_tilde_large_mu = 2.5
        self.t_tilde_small_mu = self.compute_t_tilde_small_mu()
        self.a = self.compute_a()
        self.C_f_1_s = self.compute_C_f_1_s()
        self.C_f_1_l = self.compute_C_f_1_l()
        self.F_1_inf = self.compute_F_1_inf()
        self.F_1_l_tilde_small_mu = self.compute_F_1_l_t(self.t_tilde_small_mu)
        self.F_1_s_tilde_small_mu = self.compute_F_1_s_t(self.t_tilde_small_mu)

    def acceptt(self,
                t_star = [],
                f_t_star = [],
                c_2 = []):
        #print('f_t_star: ', f_t_star)
        z = np.random.uniform(low = 0, high = f_t_star, size = 1)
        b = np.exp(- c_2)
        k_tilde = 3

        #print('z: ', z)
        #print('b: ', b)

        while True:
            if z > b:
                return 0

            b = b - (k_tilde * np.exp(- c_2 * np.power(k_tilde, 2)))
            #print('b: ', b)

            if z <= b:
                return 1

            k_tilde = k_tilde + 2
            b = b + (k_tilde * np.exp(- c_2 * np.power(k_tilde, 2)))
            #print('b: ', b)
            k_tilde = k_tilde + 2
            if k_tilde > 10:
                return 1

    def sample_small_mu(self):
        # supply a, C_f_1_s, C_f_2_s, F_1_s(t_tilde), F_1(inf)
        while True:
            P = np.random.uniform(low = 0, high = self.F_1_inf)
            #print('in small sample mu, P: ', P)
            if P <= (self.C_f_1_s * self.F_1_s_tilde_small_mu):
                t_star = self.compute_F_1_s_t_inv(P / self.C_f_1_s)
                #print('in sample small mu, t_star: ', t_star)
                if self.acceptt(t_star = t_star,
                                f_t_star =  np.exp( - ( 1 / (2 * self.a * t_star)) - np.sqrt(((self.a - 1) * np.power(self.mu, 2)) / self.a) + (np.power(self.mu, 2) * t_star) / 2),
                                c_2 = (1 / (2 * t_star))
                                ):
                    return t_star

            else:
                t_star = self.compute_F_1_l_t_inv(((P - self.C_f_1_s * self.compute_F_1_s_t(self.t_tilde_small_mu)) / self.C_f_1_l)  + self.compute_F_1_l_t(self.t_tilde_small_mu))
                #print('in sample small mu, t_star: ', t_star)
                if self.acceptt(t_star = t_star,
                                f_t_star = np.exp((- np.power(np.pi, 2) * t_star) / 8),
                                c_2 = (np.power(np.pi, 2) * t_star) / 8
                                ):
                    return t_star

    def sample_large_mu(self):
        if t_star >= 0.63662:
            C_s = 0
            C_l = - np.log(np.pi / 4) - (0.5 * np.log(2 * np.pi))
        else:
            C_l = - ((np.power(np.pi, 2) * t_tilde) / 8) + (1.5 * np.log(t_tilde) + (1 / (2 * t_tilde)))
            C_2 = C_l  + (0.5 * np.log(2 * np.pi)) + np.log(np.pi / 4)
        while true:
            t_star = np.random.wald(mean = (1/np.abs(self.mu)), scale = 1)

            if t_star <= t_tilde:
                if self.acceptt(t_star = t_star,
                                f_t_star = np.exp(C_s - (1/(2 * t_star))),
                                c_2 = (1 / (2 * t_star))
                                ):
                    return t_star
            else:
                if self.acceptt(t_star = t_star,
                                f_t_star = np.exp(C_l - (1 / (2 * t_star)) - (1.5 * np.log(t_star))),
                                c_2 = (np.power(np.pi, 2) * t_star) / 8
                                ):
                    return t_star

    def sample_wfpt(self):
        if self.mu <= self.mu_tilde:
            t_star = self.sample_small_mu()

        else:
            t_star = self.sample_large_mu()

        return ((t_star * np.power(self.model_params['theta'], 2)) / self.model_params['sigma_2']), np.random.choice([1, -1], p = [self.bernoulli_p, 1 - self.bernoulli_p])

    def make_data(self):
        self.bernoulli_p = 1 / (1 + np.exp(-2 * self.mu))
        data = np.zeros((self.sample_params['n_samples'],2))
        for i in range(0, self.sample_params['n_samples'], 1):
            data[i, 0], data[i, 1] = self.sample_wfpt()
            if i % 1000 == 0:
                print(i, ' data points sampled')
        return data

    def compute_t_tilde_small_mu(self):
        return 0.12 + 0.5 * np.exp(- self.mu/3)

    def compute_a(self):
        return ((3 + np.sqrt(9 + 4 * np.power(self.mu, 2))) / 6)

    def compute_C_f_1_s(self):
        return (np.sqrt(self.a) * (np.exp(self.mu) + np.exp(- self.mu)) * np.exp(- np.power(np.sqrt((self.a - 1)), 2) * np.power(self.mu, 2) / self.a))

    def compute_C_f_1_l(self):
        return (2 * np.pi * (np.exp(self.mu) + np.exp( - self.mu))) / (4 * np.power(self.mu, 2) + np.power(np.pi, 2))

    def compute_F_1_l_t(self,
                        t = []):
        return 1 - np.exp(-(4 * np.power(self.mu, 2) + np.power(np.pi, 2)) * t / 8)

    def compute_F_1_l_t_inv(self,
                            P = []):
        return - (8) / (4 * np.power(self.mu, 2) + np.power(np.pi, 2)) *  np.log(1 - P)
    def compute_F_1_s_t(self,
                        t = []):
        return special.erfc( 1 / np.sqrt(2 * self.a * t))
    def compute_F_1_s_t_inv(self,
                            P = []):
        return 1 / (2 * self.a * np.power(special.erfcinv(P),2))
    def compute_F_1_inf(self):
        return self.C_f_1_s * self.compute_F_1_s_t(t = self.t_tilde_small_mu) + self.C_f_1_l * (1 - self.compute_F_1_l_t(t = self.t_tilde_small_mu))
# write function F_1_s(t_tilde)
