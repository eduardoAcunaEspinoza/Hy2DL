import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from hy2dl.utils.config import Config

PI = torch.tensor(math.pi)


class BaseDistribution(nn.Module):
    """Base class for mixture distributions.

    Notation
    ----------
    - B: batch_size
    - L: seq_length
    - N: predict_last_n
    - K: num_mixture_components
    - T: num_targets
    - S: num_samples
    - Q: num_quantiles

    """

    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg

    def backtransform_parameters(
        self, params: dict[str, torch.Tensor], target_scaler: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        """Back-transform parameters from the standardized space to the original space.

        Parameters
        ----------
        params: dict[str, torch.Tensor]
            Dictionary containing the parameters of the mixture distribution, in the standardized space, with shape
            [B, N, K, T] for each parameter
        target_scaler: dict[str, torch.Tensor]
            Dictionary containing the mean and standard deviation of the target variables, with shape [T] for each
            variable.

        Returns
        -------
        dict[str, torch.Tensor]
            Dictionary containing the parameters of the mixture distribution, in the original space, with shape
            [B, N, K, T] for each parameter

        """
        raise NotImplementedError

    def calc_cdf(self, params: dict[str, torch.Tensor], weights: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """Calculate the cumulative distribution function (CDF) of the mixture distribution at the given value `x`.

        Parameters
        ----------
        params: dict[str, torch.Tensor]
            Dictionary containing the parameters of the mixture distribution, with shape [B, N, K, T] for each parameter
        weights: torch.Tensor
            Mixture weights, with shape [B, N, K, T].
        x: torch.Tensor
            Value at which to evaluate the CDF, with shape [B, N, T].

        Returns
        -------
        torch.Tensor
            CDF evaluated at `x`, with shape [B, N, T].

        """
        raise NotImplementedError

    def calc_logpdf(self, params: dict[str, torch.Tensor], weights: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """Calculate the log probability density function (log PDF) of the mixture distribution at the given value `x`.

        Parameters
        ----------
        params: dict[str, torch.Tensor]
            Dictionary containing the parameters of the mixture distribution, with shape [B, N, K, T] for each parameter
        weights: torch.Tensor
            Mixture weights, with shape [B, N, K, T].
        x: torch.Tensor
            Value at which to evaluate the CDF, with shape [B, N, T].

        Returns
        -------
        torch.Tensor
            log PDF evaluated at `x`, with shape [B, N, T].

        """
        raise NotImplementedError

    def map_parameters(self, raw_params: torch.Tensor) -> dict[str, torch.Tensor]:
        """Map LSTM output to the distribution parameters

        Parameters
        ----------
        raw_params: torch.Tensor
            Raw parameters output by the model, with shape [B, N, K * T] for each parameter.

        Returns
        -------
        dict[str, torch.Tensor]
            Dictionary containing the parameters of the mixture distribution, with shape [B, N, K, T] for each
            parameter.
        """
        raise NotImplementedError

    def mean(self, params: dict[str, torch.Tensor]) -> torch.Tensor:
        """Calculate the mean of the mixture distribution given its parameters.

        Parameters
        ----------
        params: dict[str, torch.Tensor]
            Dictionary containing the parameters of the mixture distribution, with shape [B, N, K, T] for each parameter

        Returns
        -------
        torch.Tensor
            Mean of the mixture distribution, with shape [B, N, T]

        """
        raise NotImplementedError

    def quantile(
        self,
        params: dict[str, torch.Tensor],
        weights: torch.Tensor,
        q: list[float],
        max_iter: int = 50,
        tol: float = 1e-3,
    ) -> torch.Tensor:
        """Compute quantiles of the predicted mixture distribution using Newton's method.

        Iteratively solves F(x) = q for x, for each quantile probability q, where F is the mixture CDF.
        Uses Newton-Raphson iteration: x_{n+1} = x_n - (F(x_n) - q) / f(x_n) where f is the PDF.

        Parameters
        ----------
        params : dict[str, torch.Tensor]
            Dictionary containing the parameters of the mixture distribution, with shape [B, N, K, T] for each parameter
        weights : torch.Tensor
            Mixture weights, with shape [B, N, K, T].
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

        B, N, K, T = weights.shape
        Q = len(q)

        # we expand the batch dimension to [B * Q] so we can solve all quantiles at once
        expanded_w = weights.unsqueeze(1).repeat(1, Q, 1, 1, 1).view(B * Q, N, K, T)
        expanded_params = {}
        for key, val in params.items():
            expanded_params[key] = val.unsqueeze(1).repeat(1, Q, 1, 1, 1).view(B * Q, N, K, T)  # [B*Q, N, K, T]

        q = torch.tensor(q, device=weights.device, dtype=weights.dtype)
        q = q.view(1, Q, 1, 1).repeat(B, 1, N, T).view(B * Q, N, T)  # [B*Q, N, T]

        # initial guess
        x = self.mean(expanded_params, expanded_w)  # Shape: [B*Q, N, T]

        for _ in range(max_iter):
            # Calculate PDF and CDF for all Q quantiles simultaneously
            pdf = self.calc_logpdf(params=expanded_params, weights=expanded_w, x=x).exp()  # [B*Q, N, T]
            cdf = self.calc_cdf(params=expanded_params, weights=expanded_w, x=x)  # [B*Q, N, T]

            # Newton step
            delta = (cdf - q) / (pdf + 1e-12)
            x.sub_(delta)

            # Convergence check (stops when the maximum error across ALL quantiles/batches is below tol)
            if delta.abs().max() < tol:
                break

        return x.view(B, Q, N, T).transpose(1, 2)  # [B*Q, N, T] -> [B, N, Q, T]

    def sample(self, params: dict[str, torch.Tensor], weights: torch.Tensor, num_samples: int) -> torch.Tensor:
        """Generate samples from the mixture distribution.

        Parameters
        ----------
        params: dict[str, torch.Tensor]
            Dictionary containing the parameters of the mixture distribution, with shape [B, N, K, T] for each parameter
        weights: torch.Tensor
            Mixture weights, with shape [B, N, K, T].
        num_samples: int
            Number of samples to generate for each prediction step

        Returns
        -------
        torch.Tensor
            Generated samples of shape [B, N, S, T]

        """
        raise NotImplementedError

    def _reshape_params(self, params: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Reshape parameters from [B, N, K * T] to [B, N, K, T]

        Parameters
        ----------
        params: dict[str, torch.Tensor]
            Dictionary containing the parameters of the mixture distribution, with shape [B, N, K * T] for each
            parameter

        Returns
        -------
        dict[str, torch.Tensor]
            Dictionary containing the reshaped parameters of the mixture distribution, with shape [B, N, K, T] for each
            parameter
        """
        return {
            k: v.reshape(v.shape[0], v.shape[1], self.cfg.num_mixture_components, self.cfg.output_features)
            for k, v in params.items()
        }

    def _sample_from_mixture(self, samples: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        """Helper function to sample from a mixture distribution.

        Function to sample from a mixture distribution, given samples drawn from each mixture component and the mixture
        weights.

        Parameters
        ----------
        samples: torch.Tensor
            Samples drawn from each mixture component, with shape [B, N, K, S, T]
        weights: torch.Tensor
            Mixture weights, with shape [B, N, K, T]
        Returns
        -------
        torch.Tensor
            Samples drawn from the mixture distribution, with shape [B, N, S, T]
        """

        B, N, K, S, T = samples.shape
        # Select samples according to weights using a multinomial distribution (i.e. sample the multionomial
        # distribution defined by the weights to determine which component to gather.
        w_reshaped = weights.permute(0, 1, 3, 2).reshape(-1, weights.size(2))  # [B * N * T, K]
        indices = torch.multinomial(w_reshaped, S, replacement=True)  # [B * N * T, S]

        # Reshape indices
        indices = indices.view(B, N, T, S).permute(0, 1, 3, 2).unsqueeze(2)  # [B, N, 1, S, T]

        # Now gather from the K dimension (dim=2)
        samples = torch.gather(samples, dim=2, index=indices)  # [B, N, 1, S, T]
        samples = samples.squeeze(2)  # [B, N, S, T]

        return samples

    @property
    def parameters(self) -> tuple[str]:
        pass


class GaussianMixture(BaseDistribution):
    """Gaussian mixture distribution."""

    def backtransform_parameters(
        self, params: dict[str, torch.Tensor], target_scaler: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        mean_data = target_scaler["mean"].view(1, 1, 1, -1)  # [1, 1, 1, T]
        std_data = target_scaler["std"].view(1, 1, 1, -1)  # [1, 1, 1, T]
        return {"loc": params["loc"] * std_data + mean_data, "scale": params["scale"] * std_data}

    def calc_cdf(self, params: dict[str, torch.Tensor], weights: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        x = x.unsqueeze(-2)  # [B, N, 1, T]
        loc, scale = params["loc"], params["scale"]
        z = (x - loc) / (scale * math.sqrt(2))
        cdf = 0.5 * (1 + torch.erf(z))
        cdf = (weights * cdf).sum(dim=-2)  # [B, N, T]
        return cdf

    def calc_logpdf(self, params: dict[str, torch.Tensor], weights: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        x = x.unsqueeze(-2)  # [B, N, 1, T]
        loc, scale = params["loc"], params["scale"]
        scale = torch.clamp(scale, min=1e-6)
        p = (x - loc) / scale
        log_p = -0.5 * p.pow(2) - torch.log(scale) - 0.5 * torch.log(2 * PI)
        log_w = torch.log(torch.clamp(weights, min=1e-10))
        log_p = torch.logsumexp(log_p + log_w, dim=2)  # [B, N, T]
        return log_p

    def map_parameters(self, raw_params: torch.Tensor) -> dict[str, torch.Tensor]:
        loc, scale = raw_params.chunk(2, dim=-1)
        params = {"loc": loc, "scale": F.softplus(scale)}
        return self._reshape_params(params)

    def mean(self, params: dict[str, torch.Tensor]) -> torch.Tensor:
        # Reference: https://en.wikipedia.org/wiki/Normal_distribution
        return params["loc"]

    def sample(self, params: dict[str, torch.Tensor], weights: torch.Tensor, num_samples: int) -> torch.Tensor:
        loc, scale = params["loc"], params["scale"]

        B, N, K, T = loc.shape
        S = num_samples

        # Sample each mixture component
        samples = torch.randn(B, N, K, S, T).to(loc.device)
        # Scale and shift samples to the correct location and scale (according to the conventions in SciPy)
        samples = samples * scale.unsqueeze(-2) + loc.unsqueeze(-2)

        # Combine samples from each mixture component according to the mixture weights
        samples = self._sample_from_mixture(samples, weights)
        return samples

    @property
    def parameters(self) -> tuple[str]:
        return ("loc", "scale")


class AsymmetricLaplaceMixture(BaseDistribution):
    """Asymmetric Laplace mixture distribution."""

    def backtransform_parameters(
        self, params: dict[str, torch.Tensor], target_scaler: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        mean_data = target_scaler["mean"].view(1, 1, 1, -1)  # [1, 1, 1, T]
        std_data = target_scaler["std"].view(1, 1, 1, -1)  # [1, 1, 1, T]
        return {
            "loc": params["loc"] * std_data + mean_data,
            "scale": params["scale"] * std_data,
            "kappa": params["kappa"],
        }

    def calc_cdf(self, params: dict[str, torch.Tensor], weights: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        x = x.unsqueeze(-2)  # [B, N, 1, T]
        loc, scale, kappa = params["loc"], params["scale"], params["kappa"]
        z = (x - loc) / scale
        mask = z >= 0
        cdf = torch.zeros_like(z)
        cdf[mask] = 1 - (1 / (1 + kappa[mask].pow(2))) * torch.exp(-1 * kappa[mask] * z[mask])
        cdf[~mask] = (kappa[~mask].pow(2) / (1 + kappa[~mask].pow(2))) * torch.exp(z[~mask] / kappa[~mask])
        cdf = (weights * cdf).sum(dim=-2)  # [B, N, T]
        return cdf

    def calc_logpdf(self, params: dict[str, torch.Tensor], weights: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        x = x.unsqueeze(-2)  # [B, N, 1, T]
        loc, scale, kappa = params["loc"], params["scale"], params["kappa"]
        scale = torch.clamp(scale, min=1e-6)
        kappa = torch.clamp(kappa, min=1e-6)
        p = (x - loc) / scale
        mask = p >= 0
        log_p = torch.zeros_like(p)
        log_p[mask] = -1 * p[mask] * kappa[mask]
        log_p[~mask] = p[~mask] / kappa[~mask]
        log_p = log_p - torch.log(kappa + 1 / kappa) - torch.log(scale)
        log_w = torch.log(torch.clamp(weights, min=1e-10))
        log_p = torch.logsumexp(log_p + log_w, dim=2)  # [B, N, T]
        return log_p

    def map_parameters(self, raw_params: torch.Tensor) -> dict[str, torch.Tensor]:
        loc, scale, kappa = raw_params.chunk(3, dim=-1)
        params = {"loc": loc, "scale": F.softplus(scale), "kappa": F.softplus(kappa)}
        return self._reshape_params(params)

    def mean(self, params: dict[str, torch.Tensor]) -> torch.Tensor:
        # Reference: https://en.wikipedia.org/wiki/Asymmetric_Laplace_distribution
        return params["loc"] + params["scale"] * (1 - params["kappa"].pow(2)) / params["kappa"]

    def sample(self, params: dict[str, torch.Tensor], weights: torch.Tensor, num_samples: int) -> torch.Tensor:
        # Reference: https://en.wikipedia.org/wiki/Asymmetric_Laplace_distribution
        loc, scale, kappa = params["loc"], params["scale"], params["kappa"]

        B, N, K, T = loc.shape
        S = num_samples

        # Sample each mixture component
        u = torch.rand(B, N, K, S, T).to(loc.device)  # uniform samples
        samples = torch.zeros_like(u)
        # Assymetric function -> sample depending on whether we are sampling from the left or right of the mode
        kappa = kappa.unsqueeze(-2).repeat((1, 1, 1, S, 1))
        p_at_mode = kappa**2 / (1 + kappa**2)
        mask = u < p_at_mode
        samples[mask] = kappa[mask] * torch.log(u[mask] * (1 + kappa[mask].pow(2)) / kappa[mask].pow(2))  # Left side
        samples[~mask] = -1 * torch.log((1 - u[~mask]) * (1 + kappa[~mask].pow(2))) / kappa[~mask]  # Right side

        # Scale and shift samples to the correct location and scale (according to the conventions in SciPy)
        samples = samples * scale.unsqueeze(-2) + loc.unsqueeze(-2)

        # Combine samples from each mixture component according to the mixture weights
        samples = self._sample_from_mixture(samples, weights)
        return samples

    @property
    def parameters(self) -> tuple[str]:
        return ("loc", "scale", "kappa")
