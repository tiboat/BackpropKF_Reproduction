import torch
import torch.nn as nn


class KalmanFilter(nn.Module):

    def __init__(self, device, N):
        """
        The constructor for KalmanFilter

        :param device: device on which to perform this model
        :param N: batch size
        """
        super().__init__()
        self.device = device
        self.batch_size = N

        self.Cz = torch.tensor([[1., 0., 0., 0.],
                                [0., 1., 0., 0.]], requires_grad=False, device=device)
        print("Initial Cz: ")
        print(self.Cz)
        print()

        # loss matrix -> get x and y out of mean_xt -> not needing to learn because needed for loss calculation
        self.Cy = torch.tensor([[1., 0., 0., 0.],
                                [0., 1., 0., 0.]], requires_grad=False, device=device)
        print("Initial Cy: ")
        print(self.Cy)
        print()

        # State transition uncertainty
        self.diag_Q = nn.parameter.Parameter(torch.rand(4, requires_grad=True, device=device))
        print("Initial Q: ")
        print(torch.diag(self.diag_Q))
        print()

        self.A = torch.tensor([[1., 0., 1., 0.],
                               [0., 1., 0., 1.],
                               [0., 0., 1., 0.],
                               [0., 0., 0., 1.]], requires_grad=False, device=device)
        print("Initial A: ")
        print(self.A)
        print()

    def forward(self, z_list, L_hat_list, pos0s, vel0s, simplified_cov_update):
        # z_list: [(N, 2), ...] of length T
        # L_hat_list: [(N, 3), ...] of length T
        # pos0s: (T, N, 2) tensor
        # vel0s: (T, N, 2) tensor
        # y_hats_output: (T, N, 2) tensor
        T = len(z_list)

        # Initial belief state (center frame and zero velocity) -> no requires grad because needed only once
        mean0 = torch.zeros((1, 4), requires_grad=False, device=self.device)
        mean0[0][0] = torch.rand(1) * 128
        mean0[0][1] = torch.rand(1) * 128
        mean0[0][2] = (-1 - 1) * torch.rand(1) + 1
        mean0[0][3] = (-1 - 1) * torch.rand(1) + 1
        mean0s = mean0.repeat(self.batch_size, 1)

        # Initial state covariance initialised on random variances -> no requires grad because needed only once
        covar0 = torch.zeros((1, 4, 4), requires_grad=False, device=self.device)
        covar0[0][0][0] = 30.
        covar0[0][1][1] = 30.
        covar0[0][2][2] = .333
        covar0[0][3][3] = .333
        covar0s = covar0.repeat(self.batch_size, 1, 1)

        if pos0s is not None:
            mean0s[:, :2] = pos0s
            covar0s[:, 0, 0] = torch.eye(1)
            covar0s[:, 1, 1] = torch.eye(1)

        if vel0s is not None:
            mean0s[:, 2:4] = vel0s
            covar0s[:, 2, 2] = torch.eye(1)
            covar0s[:, 3, 3] = torch.eye(1)

        means = [mean0s]
        covars = [covar0s]
        y_hats = []

        for t in range(T):
            # z: (N, 2) tensor
            # L_hat: (N, 3) tensor
            z = z_list[t]
            L_hat = L_hat_list[t]
            # [-1] means get the most recent value for covar and mean
            (mean_output, covar_output) = self.kf_update(means[-1], covars[-1], z, L_hat, simplified_cov_update, t)

            # y_hat: (N, 3) tensor
            y_hat = (self.Cy @ mean_output.unsqueeze(-1)).squeeze(-1)
            # y_hat = (self.Cy @ mean_output).squeeze(-1)

            means.append(mean_output)
            covars.append(covar_output)
            y_hats.append(y_hat)

        means.pop(0)
        covars.pop(0)
        y_hats_output = torch.stack(y_hats, 0)

        # self.show_matrices()

        return y_hats_output

    def process_L_hat_single(self, L_hat_single):
        # L_hat_single: (3, ) tensor
        # R: (2, 2) tensor
        L = torch.zeros(2, 2).to(self.device)
        L[0, 0] = torch.exp(L_hat_single[0])
        L[1, 0] = L_hat_single[1]
        L[1, 1] = torch.exp(L_hat_single[2])
        R = torch.matmul(L, L.t())
        return R

    def process_L_hat_batch(self, L_hat):
        # L_hat: (N, 3) tensor
        # R: (N, 2, 2) tensor
        N = L_hat.size(0)
        L_hat_tuple = L_hat.unbind(0)
        R_list = [self.process_L_hat_single(L_hat_single) for L_hat_single in L_hat_tuple]
        R = torch.stack(R_list)
        return R

    def mean_dyn_update(self, mean_prev):
        """
        Computes the mean dynamics update and returns the new predicted mean state vector.

        :param mean_prev: mean state vector of the previous time step (t-1)
        :return: predicted mean state vector of this time step (t)
        """
        # µ(xt + 1) = Aµt
        return self.A @ mean_prev.unsqueeze(-1)

    def covar_dyn_update(self, covar_prev):
        """
        Computes the covariance dynamics update and returns the predicted estimate uncertainty

        :param covar_prev: estimate uncertainty of the previous time step (t-1)
        :return: predicted estimate uncertainty of this time step (t)
        """
        # Σ(xt+1) = AΣ(xt)AT + Q
        return self.A @ covar_prev @ self.A.t() + torch.diag(self.diag_Q)

    def kalman_gain(self, covar_predicted, R, Cz):
        """
        Returns the kalman gain.

        :param Cz: Cz matrix
        :param R: R matrix
        :param covar_predicted: predicted estimate uncertainty of this time step
        :return: kalman gain
        """
        # K(t+1) = Σ(xt+1)CzT(CzΣ(xt+1)CzT + R(t+1))^−1
        # Cz is (N, 2, 4) -> Cz.transpose(-2, -1) is (N, 4, 2)
        return covar_predicted @ Cz.transpose(-2, -1) @ (Cz @ covar_predicted @ Cz.transpose(-2, -1) + R).inverse()

    def mean_obs_update(self, K, z, mean_predicted, Cz):
        """
        Performs the mean observation update and returns the corrected mean state vector of this time step.

        :param Cz: Cz matrix
        :param K: kalman gain of this time step
        :param z: observation of this time step
        :param mean_predicted: predicted mean state vector of this time step
        :return: corrected mean state vector of this time step
        """
        # µ(xt+1) = µ(xt+1) + K(t+1)(z(t+1) − Czµ(xt+1))
        return (mean_predicted + K @ (z.unsqueeze(-1) - Cz @ mean_predicted)).squeeze(-1)

    def simplified_covar_obs_update(self, K, covar_predicted, Cz):
        """
        Performs the simplified covariance observation update and returns the corrected estimate uncertainty of this
        time step

        :param Cz: Cz matrix
        :param K: kalman gian of this time step
        :param covar_predicted: predicted estimate uncertainty of this time step
        :return: corrected estimate uncertainty of this time step
        """
        # Σ(xt+1) = (I − K(t+1)Cz)Σ(xt+1)
        return (torch.eye(4).to(self.device) - K @ Cz) @ covar_predicted

    def covar_obs_update(self, K, covar_predicted, Cz, R):
        """
        Performs the covariance observation update and returns the corrected estimate uncertainty of this
        time step

        :param R: R matrix
        :param Cz: Cz matrix
        :param K: kalman gain of this time step
        :param covar_predicted: predicted estimate uncertainty of this time step
        :return: corrected estimate uncertainty of this time step
        """
        # Σ(xt+1) = (I − K(t+1)Cz) Σ(xt+1) (I − K(t+1)Cz)T + K R KT
        IKCz = (torch.eye(4).to(self.device) - K @ Cz)
        return IKCz @ covar_predicted @ IKCz.permute(0, 2, 1) + K @ R @ K.permute(0, 2, 1)

    def kf_update(self, mean_input, covar_input, z, L_hat, simplified_cov_update):
        # mean_input: (N, 4) tensor
        # covar_input: (N, 4, 4) tensor
        # L_hat: (N, 3) tensor
        # z: (N, 2) tensor
        # mean_output: (N, 4) tensor
        # covar_output: (N, 4, 4) tensor

        # Extract batch size from μ_input
        N = mean_input.shape[0]

        # Add batch dim to self.Cz
        # Shape goes from (2,4) to (N,2,4) tensor
        Cz = self.Cz.unsqueeze(0).repeat(N, 1, 1)

        # mean is (N, 4, 1)
        mean_predicted = self.mean_dyn_update(mean_input)

        # (N, 4, 4) + (4, 4) = (N, 4, 4) tensor
        covar_predicted = self.covar_dyn_update(covar_input)

        R = self.process_L_hat_batch(L_hat)  # (N, 2, 2) tensor

        # K is (N, 4, 2)
        K = self.kalman_gain(covar_predicted, R, Cz)

        mean_output = self.mean_obs_update(K, z, mean_predicted, Cz) # (N, 4) tensor

        if simplified_cov_update:
            covar_output = self.simplified_covar_obs_update(K, covar_predicted, Cz)  # (N, 5, 5) tensor
        else:
            covar_output = self.covar_obs_update(K, covar_predicted, Cz, R)  # (N, 5, 5) tensor

        return mean_output, covar_output

    def show_matrices(self):
        print(" Cz: ")
        print(self.Cz)

        print(" Q: ")
        print(torch.diag(self.diag_Q))

        print(" A: ")
        print(self.A)
