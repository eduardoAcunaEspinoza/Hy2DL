import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from hy2dl.modelzoo.inputlayer import InputLayer
from hy2dl.utils.config import Config
from hy2dl.utils.distributions import Distribution

PI = torch.tensor(math.pi)

class LSTMMDN(nn.Module):
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
        """Special initialization of the bias."""
        if cfg.initial_forget_bias is not None:
            self.lstm.bias_hh_l0.data[cfg.hidden_size : 2 * cfg.hidden_size] = cfg.initial_forget_bias


    def forward(self, sample):
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
    
    def sample(self, x, num_samples):
        S = num_samples
        params, w = self(x).values()
        B, N, K, T = next(iter(params.values())).shape
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

        # Forgive me father for I have sinned.
        
        # loc, scale: [B, N, K, T]
        # samples: [B, N, K, S, T]
        samples = samples * scale.unsqueeze(-2) + loc.unsqueeze(-2)  # [B, N, K, S, T]

        # Select samples according to weights
        # w: [B, N, K, T]
        # Reshape w to [B * N * T, K] for multinomial
        w_reshaped = w.permute(0, 1, 3, 2).reshape(-1, w.size(2))  # [B * N * T, K]
        indices = torch.multinomial(w_reshaped, S, replacement=True)  # [B * N * T, S]

        # Reshape indices back to proper dimensions
        indices = indices.view(B, N, T, S)  # [B, N, T, S]
        indices = indices.permute(0, 1, 3, 2)  # [B, N, S, T]
        indices = indices.unsqueeze(2)  # [B, N, 1, S, T]

        # Now gather from the num_mixture_components dimension (dim=2)
        samples = torch.gather(samples, dim=2, index=indices)  # [B, N, 1, S, T]
        samples = samples.squeeze(2)  # [B, N, S, T]

        return samples
    
    def mean(self, sample):
        with torch.no_grad():
            params, w = self(sample).values()
            match self.distribution:
                case Distribution.GAUSSIAN:
                    mean = params["loc"]
                case Distribution.LAPLACIAN:
                    loc, scale, kappa = params.values()
                    mean = loc + scale * (1 - kappa.pow(2)) / kappa
            mean = (mean * w).sum(axis=-2)
        return mean
    
    def _calc_logpdf(self, x, xi):
        """
        Calculate the density of `xi` in the mixture PDF of `x`.

        Parameters
        ----------
        x : torch.Tensor 
            Tensor of shape [B, L, I].
        xi : torch.Tensor
            The points at which to evaluate the PDF. Tensor of shape [B, N, T].

        Returns
        -------
        torch.Tensor
            The PDF values at `xi`. Tensor of shape [B, N, T].
        """

        xi = xi.unsqueeze(-2) # [B, N, 1, T]

        params, weights = self(x).values() # loc: [B, N, K, T]
        match self.distribution:
            case Distribution.GAUSSIAN:
                loc, scale = params.values()
                scale = torch.clamp(scale, min=1e-6)
                p = (xi - loc) / scale
                log_p = -0.5 * p.pow(2) - torch.log(scale) - 0.5 * torch.log(2 * PI)

            case Distribution.LAPLACIAN:
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
    
    def _calc_cdf(self, x, xi):
        """
        Calculate the mixture CDF at points `xi`.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor [B, L, I].
        xi : torch.Tensor
            Evaluation points [B, N, T].

        Returns
        -------
        torch.Tensor
            Mixture CDF values [B, N, T].
        """
        xi = xi.unsqueeze(-2) # [B, N, 1, T]

        params, weights = self(x).values()

        match self.distribution:
            case Distribution.GAUSSIAN:
                loc, scale = params.values() # loc: [B, N, K, T]
                z = (xi - loc) / (scale * math.sqrt(2)) 
                cdf = 0.5 * (1 + torch.erf(z))

            case Distribution.LAPLACIAN:
                loc, scale, kappa = params.values()
                z = (xi - loc) / scale
                mask = (z >= 0)
                cdf = torch.zeros_like(z)
                cdf[mask] = 1 - (1 / (1 + kappa[mask].pow(2))) * torch.exp(-1 * kappa[mask] * z[mask])
                cdf[~mask] = (kappa[~mask].pow(2) / (1 + kappa[~mask].pow(2))) * torch.exp(z[~mask] / kappa[~mask])

        # Mix CDF (weighted mixture over components)
        cdf = (weights * cdf).sum(dim=-2)  # [B, N, T]
        return cdf
    
    def quantile(self, x, q: list[float], max_iter: int = 50, tol: float = 1e-6):
        out = []
        with torch.no_grad():
            # Solve one quantile at a time
            for qi in q:
                # Mean as the initial guess
                xi = self.mean(x)  # [B, N, T]
                
                for _ in range(max_iter):
                    pdf = self._calc_logpdf(x, xi).exp()   # [B, N, T]
                    cdf = self._calc_cdf(x, xi)            # [B, N, T]
                    
                    # Newton step
                    delta = (cdf - qi) / (pdf + 1e-12)     # [B, N, T]
                    xi.sub_(delta) # Substract delta in-place
                    
                    # Convergence check
                    if delta.abs().max() < tol:
                        break
                                        
                out.append(xi.clone())
        
        # Stack quantiles -> [B, N, Q, T]
        return torch.stack(out, dim=2)