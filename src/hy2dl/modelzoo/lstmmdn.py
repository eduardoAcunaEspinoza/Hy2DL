import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from hy2dl.modelzoo.inputlayer import InputLayer
from hy2dl.utils.config import Config
from hy2dl.utils.distributions import Distribution

PI = torch.tensor(math.pi)

class LSTMMDN(nn.Module):
    """LSTM with a Mixture Density Network (MDN) head layer.

    This class implements an LSTM layer followed by a MDN head, which maps the
    hidden states produced by the LSTM into the parameters of a mixture
    distribution for predictions for a specified number of time steps.

    Parameters
    ----------
    cfg : Config
        Configuration object containing model hyperparameters and settings.


    Notes
    -----
    Notation used for the shapes of tensors:
    - B: batch_size
    - I: num_input_features
    - L: seq_length
    - N: predict_last_n
    - K: num_components
    - T: num_targets
    - S: num_samples
    - Q: num_quantiles

    """

    def __init__(self, cfg: Config):

        super().__init__()

        self.embedding_net = InputLayer(cfg)

        self.lstm = nn.LSTM(input_size=self.embedding_net.output_size, hidden_size=cfg.hidden_size, batch_first=True)

        self.dropout = torch.nn.Dropout(p=cfg.dropout_rate)

        self.distribution = Distribution.from_string(cfg.distribution)
        match self.distribution:
            case Distribution.GAUSSIAN:
                self.num_params = 2
            case Distribution.LAPLACIAN:
                self.num_params = 3

        self.fc_params = nn.Linear(cfg.hidden_size, self.num_params * cfg.num_mixture_components * cfg.output_features)

        self.fc_weights = nn.Sequential(
            nn.Linear(cfg.hidden_size, cfg.num_mixture_components * cfg.output_features),
            nn.Unflatten(-1, (cfg.num_mixture_components, cfg.output_features)),
            nn.Softmax(dim=-2)
        )

        self.num_mixture_components = cfg.num_mixture_components
        self.predict_last_n = cfg.predict_last_n

        self.output_features = cfg.output_features

        self._reset_parameters(cfg=cfg)

    def _reset_parameters(self, cfg: Config):
        """Special initialization of the bias.
        
        Sets the initial forget gate bias to a specified value (if any).

        Parameters
        ----------
        cfg : Config
            Configuration object containing model hyperparameters and settings.
        """
        if cfg.initial_forget_bias is not None:
            self.lstm.bias_hh_l0.data[cfg.hidden_size : 2 * cfg.hidden_size] = cfg.initial_forget_bias


    def forward(self, sample):
        """Forward pass of LSTM-MDN
        
        Runs a forward pass through the LSTM-MDN network which returns the
        the parameters and weights of the mixture distribution of predictions
        for N time steps of the sequence.

        Parameters
        ----------
        sample: dict[str, torch.Tensor | dict[str, torch.Tensor]]
            Dictionary with the different tensors / dictionaries that will be
            used for the forward pass.

        Returns
        -------
        dict
            Dictionary containing:
            - 'params': dict of distribution parameters [B, N, K, T]
            - 'weights': mixture weights of shape [B, N, K, T]

        """
        # Pre-process data to be sent to the LSTM
        x_lstm = self.embedding_net(sample)

        # Forward pass through the LSTM
        out, _ = self.lstm(x_lstm)
        
        # Extract sequence of interest
        out = out[:, -self.predict_last_n:, :]
        out = self.dropout(out)

        # Probabilistic things
        w = self.fc_weights(out)

        params = self.fc_params(out)
        match self.distribution:
            case Distribution.GAUSSIAN:
                loc, scale = params.chunk(2, dim=-1)
                scale = F.softplus(scale)
                params = {"loc": loc, "scale": scale}
            case Distribution.LAPLACIAN:
                loc, scale, kappa = params.chunk(3, dim=-1)
                scale = F.softplus(scale)
                kappa = F.softplus(kappa)
                params = {"loc": loc, "scale": scale, "kappa": kappa}
        
        params = {k: v.reshape(v.shape[0], v.shape[1], self.num_mixture_components, self.output_features) for k, v in params.items()}
        
        return {"params": params, "weights": w}

    def sample(self, sample: dict[str, torch.Tensor | dict[str, torch.Tensor]], num_samples: int) -> torch.Tensor:
        """
        Generate samples from the predicted mixture distribution.

        Parameters
        ----------
        sample : torch.Tensor
            Input sequences of shape [B, L, I]
        num_samples : int
            Number of samples to generate for each prediction step

        Returns
        -------
        torch.Tensor
            Generated samples of shape [B, N, S, T]
        """

        # Forward pass to get mixture parameters and weights
        pred = self(sample)
        params, w = pred["params"], pred["weights"]

        # Get shapes
        B, N, K, T = next(iter(params.values())).shape
        S = num_samples

        # Sample depending on the distribution
        match self.distribution:
            case Distribution.GAUSSIAN:
                loc, scale = params.values()
                
                samples = torch.randn(B, N, K, S, T).to(loc.device)
            case Distribution.LAPLACIAN:
                loc, scale, kappa = params.values()

                u = torch.rand(B, N, K, S, T).to(loc.device)

                # Sampling left or right of the mode?
                kappa = kappa.unsqueeze(-2).repeat((1, 1, 1, S, 1))
                p_at_mode = kappa**2 / (1 + kappa**2)

                mask = u < p_at_mode

                samples = torch.zeros_like(u)

                samples[mask] = kappa[mask] * torch.log(u[mask] * (1 + kappa[mask].pow(2)) / kappa[mask].pow(2)) # Left side
                samples[~mask] = -1 * torch.log((1 - u[~mask]) * (1 + kappa[~mask].pow(2))) / kappa[~mask] # Right side

        # Transform samples to the correct location and scale according to the
        # conventions in SciPy.
        
        # loc, scale: [B, N, K, T]
        # samples: [B, N, K, S, T]
        samples = samples * scale.unsqueeze(-2) + loc.unsqueeze(-2)  # [B, N, K, S, T]

        # Select samples according to weights using a multinomial distribution
        # i.e. sample the multionomial distribution defined by the weights
        # to determine which component to gather.

        # w: [B, N, K, T]
        # Reshape w to [B * N * T, K] for multinomial
        w_reshaped = w.permute(0, 1, 3, 2).reshape(-1, w.size(2))  # [B * N * T, K]
        indices = torch.multinomial(w_reshaped, S, replacement=True)  # [B * N * T, S]

        # Reshape indices back to proper dimensions
        indices = indices.view(B, N, T, S)  # [B, N, T, S]
        indices = indices.permute(0, 1, 3, 2)  # [B, N, S, T]
        indices = indices.unsqueeze(2)  # [B, N, 1, S, T]

        # Now gather from the K dimension (dim=2)
        samples = torch.gather(samples, dim=2, index=indices)  # [B, N, 1, S, T]
        samples = samples.squeeze(2)  # [B, N, S, T]

        return samples

    def mean(self, sample: dict[str, torch.Tensor | dict[str, torch.Tensor]]) -> torch.Tensor:
        """
        Compute the mean of the mixture distribution.
               
        Parameters
        ----------
        sample : torch.Tensor
            Input sequences of shape [B, L, I]
        
        Returns
        -------
        torch.Tensor
            Predictive mean of shape [B, N, T]
        """

        with torch.no_grad():
            # Forward pass to get mixture parameters and weights
            pred = self(sample)
            params, w = pred["params"], pred["weights"]

            match self.distribution:
                case Distribution.GAUSSIAN:
                    # Reference: https://en.wikipedia.org/wiki/Normal_distribution
                    mean = params["loc"]
                case Distribution.LAPLACIAN:
                    # Reference: https://en.wikipedia.org/wiki/Asymmetric_Laplace_distribution
                    loc, scale, kappa = params.values()
                    mean = loc + scale * (1 - kappa.pow(2)) / kappa
            mean = (mean * w).sum(axis=-2)
        return mean

    def _calc_logpdf(self, sample: dict[str, torch.Tensor | dict[str, torch.Tensor]], xi: torch.Tensor) -> torch.Tensor:
        """
        Calculate the log density of `xi` in the mixture PDF.

        Parameters
        ----------
        sample : dict[str, torch.Tensor | dict[str, torch.Tensor]]
            Input sample containing mixture parameters and weights.
        xi : torch.Tensor
            Points at which to evaluate the log PDF. Tensor of shape [B, N, T].

        Returns
        -------
        torch.Tensor
            The log PDF values at `xi`. Tensor of shape [B, N, T].

        Notes
        -----
        This can be used as a loss function if `xi` are the target values.
        """

        # Forward pass to get mixture parameters and weights
        pred = self(sample)
        params, weights = pred["params"], pred["weights"]

        xi = xi.unsqueeze(-2) # [B, N, 1, T]

        match self.distribution:
            case Distribution.GAUSSIAN:
                # Reference: https://en.wikipedia.org/wiki/Normal_distribution
                loc, scale = params.values()
                scale = torch.clamp(scale, min=1e-6)
                p = (xi - loc) / scale
                log_p = -0.5 * p.pow(2) - torch.log(scale) - 0.5 * torch.log(2 * PI)

            case Distribution.LAPLACIAN:
                # Reference: https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.laplace_asymmetric.html
                loc, scale, kappa = params.values()
                scale = torch.clamp(scale, min=1e-6)
                kappa = torch.clamp(kappa, min=1e-6)
                

                p = (xi - loc) / scale
                mask = (p >= 0)

                log_p = torch.zeros_like(p)

                log_p[mask] = -1 * p[mask] * kappa[mask]
                log_p[~mask] = p[~mask] / kappa[~mask]

                log_p = log_p - torch.log(kappa + 1 / kappa) - torch.log(scale)

        log_w = torch.log(torch.clamp(weights, min=1e-10))
        log_p = torch.logsumexp(log_p + log_w, dim=-2) # [B, N, T]
    
        return log_p
    
    def _calc_cdf(self, sample: dict[str, torch.Tensor | dict[str, torch.Tensor]], xi: torch.Tensor) -> torch.Tensor:
        """
        Calculate the value of the mixture CDF at `xi`.

        Parameters
        ----------
        sample : dict[str, torch.Tensor | dict[str, torch.Tensor]]
            Input sample containing mixture parameters and weights.
        xi : torch.Tensor
            Points at which to evaluate the log PDF. Tensor of shape [B, N, T].

        Returns
        -------
        torch.Tensor
            The log PDF values at `xi`. Tensor of shape [B, N, T].
        """
        # Forward pass to get mixture parameters and weights
        pred = self(sample)
        params, weights = pred["params"], pred["weights"]

        xi = xi.unsqueeze(-2) # [B, N, 1, T]

        match self.distribution:
            case Distribution.GAUSSIAN:
                # Reference: https://en.wikipedia.org/wiki/Normal_distribution
                loc, scale = params.values() # loc: [B, N, K, T]
                z = (xi - loc) / (scale * math.sqrt(2)) 
                cdf = 0.5 * (1 + torch.erf(z))

            case Distribution.LAPLACIAN:
                # References:
                # - https://en.wikipedia.org/wiki/Asymmetric_Laplace_distribution
                # - https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.laplace_asymmetric.html
                loc, scale, kappa = params.values()
                z = (xi - loc) / scale
                mask = (z >= 0)
                cdf = torch.zeros_like(z)
                cdf[mask] = 1 - (1 / (1 + kappa[mask].pow(2))) * torch.exp(-1 * kappa[mask] * z[mask])
                cdf[~mask] = (kappa[~mask].pow(2) / (1 + kappa[~mask].pow(2))) * torch.exp(z[~mask] / kappa[~mask])

        # Mix CDF (weighted mixture over components)
        cdf = (weights * cdf).sum(dim=-2)  # [B, N, T]
        return cdf
    
    def quantile(self, sample: dict[str, torch.Tensor | dict[str, torch.Tensor]], q: list[float], max_iter: int = 50, tol: float = 1e-3) -> torch.Tensor:
        """
        Compute quantiles of the predicted mixture distribution using Newton's
        method.

        Iteratively solves F(x) = q for x, for each quantile probability q,
        where F is the mixture CDF. Uses Newton-Raphson iteration:
        x_{n+1} = x_n - (F(x_n) - q) / f(x_n) where f is the PDF.
        
        Parameters
        ----------
        x : torch.Tensor
            Input sequences of shape [B, L, I]
        q : list[float]
            List of quantile probabilities (between 0 and 1)
        max_iter : int, default=50
            Maximum number of Newton iterations
        tol : float, default=1e-3
            Convergence tolerance for Newton's method
        
        Returns
        -------
        torch.Tensor
            Quantile values of shape [B, N, Q, T]
        
        """
        out = []
        with torch.no_grad():
            # Solve one quantile at a time
            for qi in q:
                # Mean as the initial guess
                xi = self.mean(sample)  # [B, N, T]
                
                for _ in range(max_iter):
                    pdf = self._calc_logpdf(sample, xi).exp()   # [B, N, T]
                    cdf = self._calc_cdf(sample, xi)            # [B, N, T]
                    
                    # Newton step
                    delta = (cdf - qi) / (pdf + 1e-12)     # [B, N, T]
                    xi.sub_(delta) # Substract delta in-place
                    
                    # Convergence check
                    if delta.abs().max() < tol:
                        break
                                        
                out.append(xi.clone())
        
        # Stack quantiles -> [B, N, Q, T]
        return torch.stack(out, dim=2)