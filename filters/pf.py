"""
Sudhanva Sreesha
ssreesha@umich.edu
28-Mar-2018

This file implements the Particle Filter.
"""

import numpy as np
from numpy.random import uniform, multivariate_normal
from scipy.stats import norm as gaussian

from filters.localization_filter import LocalizationFilter
from tools.task import get_gaussian_statistics
from tools.task import get_observation
from tools.task import sample_from_odometry
from tools.task import wrap_angle


class PF(LocalizationFilter):
    def __init__(self, initial_state, alphas, beta, num_particles, global_localization):
        super(PF, self).__init__(initial_state, alphas, beta)
        # TODO add here specific class variables for the PF
        self.M = num_particles
        self.X = multivariate_normal(self.mu, self.Sigma, self.M)
        self.w = np.ones(self.M)/sum(np.ones(self.M))
        self.gl = global_localization


    def predict(self, u):
        # TODO Implement here the PF, perdiction part
        for i in range(self.M):
            self.X[i] = sample_from_odometry(self.X[i], u, self._alphas)

        # self._state_bar.mu = self.mu
        # self._state_bar.Sigma = self.Sigma

    def update(self, z):
        distances = np.zeros((self.M))
        for i in range(self.M):
            distances[i] = get_observation(self.X[i], z[1])[0] - z[0]
        distances_pdf = gaussian(distances)
        self.w *= distances_pdf.pdf(distances)
        self.w = self.w/sum(self.w)

        # Resample
        cum_dist = np.cumsum(self.w)
        cum_dist[-1] = 1.

        indexes = np.searchsorted(cum_dist, np.random.uniform(0, 1, self.M))
        self.X[:] = self.X[indexes]
        self.w.fill(1.0/self.M)

        np.random.shuffle(self.X)
        self._state.mu = np.mean(self.X, axis=0)[np.newaxis].T
        self._state.Sigma = np.diag(np.var(self.X, axis=0))
