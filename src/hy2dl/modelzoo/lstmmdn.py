import torch
import torch.nn as nn
import torch.nn.functional as F

from hy2dl.modelzoo.inputlayer import InputLayer
from hy2dl.utils.config import Config
from hy2dl.utils.distributions import Distribution


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

        self.fc_params = nn.Linear(cfg.hidden_size, self.num_params * cfg.num_components * cfg.output_features)

        self.fc_weights = nn.Sequential(
            nn.Linear(cfg.hidden_size, cfg.num_components * cfg.output_features),
            nn.Unflatten(-1, (cfg.num_components, cfg.output_features)),
            nn.Softmax(dim=-2)
        )

        self.num_components = cfg.num_components
        self.predict_last_n = cfg.predict_last_n

        self.output_features = cfg.output_features

        self._reset_parameters(cfg=cfg)

    def _reset_parameters(self, cfg: Config):
        """Special initialization of the bias."""
        if cfg.initial_forget_bias is not None:
            self.lstm.bias_hh_l0.data[cfg.hidden_size : 2 * cfg.hidden_size] = cfg.initial_forget_bias


    def forward(self, sample):
        # Pre-process data to be sent to the LSTM
        processed_sample = self.embedding_net(sample)
        x_lstm = self.embedding_net.assemble_sample(processed_sample)

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
        params = {k: v.reshape(v.shape[0], v.shape[1], self.num_components, self.output_features) for k, v in params.items()}
        
        return {"params": params, "weights": w}
    
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
    
    def sample(self, sample, num_samples):
        with torch.no_grad():
            params, w = self(sample).values()
            num_batches, sequence_length, num_components, num_targets = next(iter(params.values())).shape
            match self.distribution:
                case Distribution.GAUSSIAN:
                    loc, scale = params.values()
                    
                    samples = torch.randn(num_batches, sequence_length, num_components, num_samples, num_targets).to(loc.device)
                case Distribution.LAPLACIAN:
                    loc, scale, kappa = params.values()

                    u = torch.rand(num_batches, sequence_length, num_components, num_samples, num_targets).to(loc.device)

                    # Sampling left or right of the mode?
                    kappa = kappa.unsqueeze(-2).repeat((1, 1, 1, num_samples, 1))
                    p_at_mode = kappa**2 / (1 + kappa**2)

                    mask = u < p_at_mode

                    samples = torch.zeros_like(u)

                    samples[mask] = kappa[mask] * torch.log(u[mask] * (1 + kappa[mask].pow(2)) / kappa[mask].pow(2)) # Left side
                    samples[~mask] = -1 * torch.log((1 - u[~mask]) * (1 + kappa[~mask].pow(2))) / kappa[~mask] # Right side

            # Forgive me father for I have sinned.
            
            # samples: [num_batches, sequence_length, num_components, num_samples, output_features]
            # loc, scale: [num_batches, sequence_length, num_components, output_features]
            samples = samples * scale.unsqueeze(-2) + loc.unsqueeze(-2)  # [num_batches, sequence_length, num_components, num_samples, output_features]

            # Select samples according to weights
            # w: [num_batches, sequence_length, num_components, output_features]
            # Reshape w to [num_batches * sequence_length * output_features, num_components] for multinomial
            w_reshaped = w.permute(0, 1, 3, 2).reshape(-1, w.size(2))  # [num_batches * sequence_length * output_features, num_components]
            indices = torch.multinomial(w_reshaped, num_samples, replacement=True)  # [num_batches * sequence_length * output_features, num_samples]

            # Reshape indices back to proper dimensions
            indices = indices.view(num_batches, sequence_length, num_targets, num_samples)  # [num_batches, sequence_length, output_features, num_samples]
            indices = indices.permute(0, 1, 3, 2)  # [num_batches, sequence_length, num_samples, output_features]
            indices = indices.unsqueeze(2)  # [num_batches, sequence_length, 1, num_samples, output_features]

            # Now gather from the num_components dimension (dim=2)
            samples = torch.gather(samples, dim=2, index=indices)  # [num_batches, sequence_length, 1, num_samples, output_features]
            samples = samples.squeeze(2)  # [num_batches, sequence_length, num_samples, output_features]

        return samples