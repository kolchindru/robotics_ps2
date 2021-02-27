"""
This file implements the Extended Kalman Filter.
"""

import numpy as np

from filters.localization_filter import LocalizationFilter
from tools.task import get_motion_noise_covariance
from tools.task import get_observation as get_expected_observation
from tools.task import get_prediction
from tools.task import wrap_angle


class EKF(LocalizationFilter):
    def predict(self, u):
        # TODO Implement here the EKF, perdiction part. HINT: use the auxiliary functions imported above from tools.task
        M = get_motion_noise_covariance(u, self._alphas)

        mu_prev = self.mu
        sigma_prev = self.Sigma

        G = np.array([
            [1, 0, -u[1] * np.sin(mu_prev[2] + u[0])],
            [0, 1, u[1] * np.cos(mu_prev[2] + u[0])],
            [0, 0, 1]
        ])

        V = np.array([
            [-u[1] * np.sin(mu_prev[2] + u[0]), np.cos(mu_prev[2] + u[0]), 0],
            [u[1] * np.cos(mu_prev[2] + u[0]), np.sin(mu_prev[2] + u[0]), 0],
            [1, 0, 1]
        ])

        mu_bar = get_prediction(mu_prev, u)
        R = V @ M @ V.T
        sigma_bar = G @ sigma_prev @ G.T + R

        self._state_bar.mu = mu_bar[np.newaxis].T
        self._state_bar.Sigma = sigma_bar

    def update(self, z):
        # TODO implement correction step

        mu_bar = self.mu_bar
        Sigma_bar = self.Sigma_bar

        land_x = self._field_map.landmarks_poses_x[int(z[-1])]
        land_y = self._field_map.landmarks_poses_y[int(z[-1])]
        H = np.array([
            (land_y - mu_bar[1])/((land_x - mu_bar[0]) ** 2 + (land_y - mu_bar[1]) ** 2),
            -(land_x - mu_bar[0])/((land_x - mu_bar[0]) ** 2 + (land_y - mu_bar[1]) ** 2),
            -1
        ])

        innovation = z - get_expected_observation(mu_bar, z[-1])
        K = Sigma_bar @ H.T / (H @ Sigma_bar @ H.T + self._Q)
        mu = mu_bar + K * innovation[0]
        Sigma = (np.eye(3) - np.outer(K, H)) @ Sigma_bar

        self._state.mu = mu[np.newaxis].T
        self._state.Sigma = Sigma
