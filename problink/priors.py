import problink.xidmod as xidmod
from scipy import stats
import numpy as np

class prior(object):
    def __init__(self):
        #-----set up prior info-----
        prior_q_H, bins_q_H=xidmod.gender_prior_setup(0.95,1.0)# gender same
        prior_q_K, bins_q_K=xidmod.gender_prior_setup(0.4,0.6)# gender not same
        self.prior_q_H=prior_q_H
        self.prior_q_K=prior_q_K
        self.bins_q_H=bins_q_H
        self.bins_q_K=bins_q_K


        prior_q_H_mu, bins_q_H_mut=xidmod.mutation_prior_setup(0.99,1.0)#mutation same
        self.mut_pdf=xidmod.load_mut_pdf()#pdf of mutation (used if not the same)
        self.prior_q_H_mu=prior_q_H_mu
        self.bins_q_H_mut=bins_q_H_mut

        prior_q_H_age_dia, bins_q_H_age_dia=xidmod.age_dia_prior_H_setup(0.99,1.0)#age_dia same

        self.prior_q_H_age_dia=prior_q_H_age_dia
        self.bins_q_H_age_dia=bins_q_H_age_dia


        self.bins_percentile_hgt=np.arange(0.0,100.0,2)#bins for height percentile

        self.pdf_percentile_hgt=stats.norm.pdf(self.bins_percentile_hgt,40.0,34)
        self.prior_hgt=self.pdf_percentile_hgt/np.trapz(self.pdf_percentile_hgt, x=self.bins_percentile_hgt)#prior for height percentile
        self.sigma_hgt=2

        self.bins_percentile_BMI=np.arange(0.0,100.0,2)#bins for BMI percentile
        self.pdf_percentile_BMI=stats.norm.pdf(self.bins_percentile_BMI,50.0,34)
        self.prior_BMI=self.pdf_percentile_BMI/np.trapz(self.pdf_percentile_BMI, x=self.bins_percentile_BMI)#prior for height percentile
        self.sigma_BMI=2.0

