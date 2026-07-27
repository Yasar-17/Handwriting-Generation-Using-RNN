"""
Mixture Density Network RNN for handwriting generation.

Follows Alex Graves' "Generating Sequences With Recurrent Neural Networks".
The LSTM outputs parameters for a mixture of M bivariate Gaussians
(means, standard deviations, correlations) for (delta_x, delta_y),
plus a Bernoulli parameter for pen_up.

Also provides a conditioned variant with windowed attention over character
embeddings for text-conditioned handwriting generation.
"""

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Unconditional MDN-RNN
# ---------------------------------------------------------------------------

class MDNRNN(nn.Module):
    """LSTM-based Mixture Density Network for stroke sequence modeling.

    At each timestep the network predicts:
      - M bivariate Gaussian components for (delta_x, delta_y):
          mu_x, mu_y       : (B, M)  means
          sigma_x, sigma_y : (B, M)  std devs (exp-activated, always > 0)
          rho              : (B, M)  correlation  (tanh-activated, in (-1, 1))
          pi               : (B, M)  mixture weights (softmax-activated, sum to 1)
      - 1 Bernoulli logit for pen_up (sigmoid-activated to get probability)
    """

    def __init__(
        self,
        input_dim: int = 3,
        hidden_dim: int = 256,
        num_layers: int = 3,
        num_mixtures: int = 20,
        dropout: float = 0.2,
    ):
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_mixtures = num_mixtures

        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        mdn_out_dim = num_mixtures * 6  # mu_x, mu_y, sigma_x, sigma_y, rho, pi
        pen_out_dim = 1

        self.dropout = nn.Dropout(dropout)
        self.mdn_head = nn.Linear(hidden_dim, mdn_out_dim)
        self.pen_head = nn.Linear(hidden_dim, pen_out_dim)

        self._init_weights()

    def _init_weights(self) -> None:
        for name, param in self.named_parameters():
            if "weight_ih" in name:
                nn.init.xavier_uniform_(param)
            elif "weight_hh" in name:
                nn.init.orthogonal_(param)
            elif "bias" in name:
                nn.init.zeros_(param)

        nn.init.zeros_(self.pen_head.bias)
        nn.init.zeros_(self.pen_head.weight)

    def forward(
        self,
        x: torch.Tensor,
        hidden: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> tuple[dict[str, torch.Tensor], tuple[torch.Tensor, torch.Tensor]]:
        """
        Args:
            x: (B, T, 3)  normalized (delta_x, delta_y, pen_up)
            hidden: (h_0, c_0) each (num_layers, B, hidden_dim), optional

        Returns:
            params: dict with keys mu_x, mu_y, sigma_x, sigma_y, rho, pi, pen_up
                    each tensor has shape (B, T, ...) as noted in the docstring
            hidden: (h_T, c_T) for autoregressive sampling
        """
        lstm_out, hidden = self.lstm(x, hidden)
        lstm_out = self.dropout(lstm_out)

        mdn_params = self.mdn_head(lstm_out)  # (B, T, M*6)
        pen_logit = self.pen_head(lstm_out)    # (B, T, 1)

        B, T, _ = mdn_params.shape
        M = self.num_mixtures

        mu_x, mu_y, sigma_x, sigma_y, rho, pi = mdn_params.split(M, dim=-1)

        sigma_x = torch.exp(sigma_x)
        sigma_y = torch.exp(sigma_y)
        rho = torch.tanh(rho)
        pi = torch.softmax(pi, dim=-1)
        pen_up = torch.sigmoid(pen_logit).squeeze(-1)

        params = {
            "mu_x": mu_x,
            "mu_y": mu_y,
            "sigma_x": sigma_x,
            "sigma_y": sigma_y,
            "rho": rho,
            "pi": pi,
            "pen_up": pen_up,
        }

        return params, hidden

    def init_hidden(self, batch_size: int, device: torch.device | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        """Create zero-initialized hidden and cell states."""
        d = device or next(self.parameters()).device
        h = torch.zeros(self.num_layers, batch_size, self.hidden_dim, device=d)
        c = torch.zeros(self.num_layers, batch_size, self.hidden_dim, device=d)
        return h, c


# ---------------------------------------------------------------------------
# Windowed attention module (Graves 2013)
# ---------------------------------------------------------------------------

class WindowedAttention(nn.Module):
    """Computes a soft attention window over a character sequence.

    Following Graves' paper, the RNN predicts parameters for K Gaussians
    over character positions. The window moves monotonically forward through
    the text as the sequence is generated.

    At each timestep t, for each window k:
        kappa_t = kappa_{t-1} + exp(kappa_hat_t)   (monotonic progression)
        phi_t(u) = sum_k alpha_k * exp(-beta_k * (kappa_t - u)^2)

    The context vector is the weighted sum of character embeddings.
    """

    def __init__(self, hidden_dim: int, char_embed_dim: int, num_windows: int = 10):
        super().__init__()
        self.num_windows = num_windows
        self.char_embed_dim = char_embed_dim

        self.attention_head = nn.Linear(hidden_dim, num_windows * 3)
        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.xavier_uniform_(self.attention_head.weight)
        nn.init.zeros_(self.attention_head.bias)
        # Bias init: small positive kappa_hat to start moving, moderate beta/alpha
        bias = self.attention_head.bias
        K = self.num_windows
        bias.data[:K] = 0.0       # kappa_hat init
        bias.data[K:2*K] = 2.0    # beta init (moderate precision)
        bias.data[2*K:3*K] = 0.0  # alpha init

    def forward(
        self,
        lstm_out: torch.Tensor,
        char_embeddings: torch.Tensor,
        char_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            lstm_out: (B, T, hidden_dim)  RNN hidden states
            char_embeddings: (B, C, char_embed_dim)  character embedding sequence
            char_mask: (B, C)  boolean mask; True = valid character

        Returns:
            context: (B, T, char_embed_dim)  attention-weighted context vectors
            phi: (B, T, C)  attention weights for visualization
        """
        B, T, _ = lstm_out.shape
        B, C, _ = char_embeddings.shape
        K = self.num_windows

        # Predict attention parameters
        attn_params = self.attention_head(lstm_out)  # (B, T, 3K)
        kappa_hat, beta, alpha = attn_params.split(K, dim=-1)

        # kappa_hat -> cumulative kappa (monotonic progression)
        # kappa_t = sum_{t'=1}^{t} exp(kappa_hat_t')
        kappa = torch.cumsum(torch.exp(kappa_hat), dim=1)  # (B, T, K)

        # Clamp beta and alpha to be positive
        beta = torch.exp(beta)   # (B, T, K)
        alpha = torch.exp(alpha) # (B, T, K)

        # Compute attention weights phi_t(u) for each character position u
        # u has shape (1, 1, C) representing character indices [0, 1, ..., C-1]
        u = torch.arange(C, dtype=lstm_out.dtype, device=lstm_out.device).view(1, 1, C)

        # kappa has shape (B, T, K), u has shape (1, 1, C)
        # diff: (B, T, K, C)
        diff = kappa.unsqueeze(-1) - u.unsqueeze(-2)  # kappa_t_k - u

        # Gaussian: exp(-beta_k * (kappa_k - u)^2)
        # beta: (B, T, K) -> (B, T, K, 1)
        gaussian = torch.exp(-beta.unsqueeze(-1) * diff ** 2)  # (B, T, K, C)

        # Weighted sum over windows: phi = sum_k alpha_k * gaussian_k
        # alpha: (B, T, K) -> (B, T, K, 1)
        phi = (alpha.unsqueeze(-1) * gaussian).sum(dim=2)  # (B, T, C)

        # Apply character mask
        if char_mask is not None:
            phi = phi * char_mask.unsqueeze(1).float()  # (B, T, C)

        # Normalize attention weights
        phi_sum = phi.sum(dim=-1, keepdim=True).clamp(min=1e-8)
        phi = phi / phi_sum  # (B, T, C)

        # Context vector: weighted sum of character embeddings
        context = torch.bmm(phi, char_embeddings)  # (B, T, char_embed_dim)

        return context, phi


# ---------------------------------------------------------------------------
# Conditioned MDN-RNN with windowed attention
# ---------------------------------------------------------------------------

class MDNRNNConditioned(nn.Module):
    """Text-conditioned MDN-RNN with windowed attention over character embeddings.

    The input at each timestep is the concatenation of:
      - stroke features (delta_x, delta_y, pen_up)
      - context vector from windowed attention over the character sequence

    This allows the model to generate handwriting that follows the conditioning text.
    """

    def __init__(
        self,
        input_dim: int = 3,
        hidden_dim: int = 256,
        num_layers: int = 3,
        num_mixtures: int = 20,
        num_windows: int = 10,
        char_vocab_size: int = 80,
        char_embed_dim: int = 32,
        dropout: float = 0.2,
    ):
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_mixtures = num_mixtures
        self.char_vocab_size = char_vocab_size
        self.char_embed_dim = char_embed_dim

        self.char_embedding = nn.Embedding(char_vocab_size, char_embed_dim)

        lstm_input_dim = input_dim + char_embed_dim

        self.lstm = nn.LSTM(
            input_size=lstm_input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        self.attention = WindowedAttention(hidden_dim, char_embed_dim, num_windows)

        mdn_out_dim = num_mixtures * 6
        pen_out_dim = 1

        self.dropout = nn.Dropout(dropout)
        self.mdn_head = nn.Linear(hidden_dim, mdn_out_dim)
        self.pen_head = nn.Linear(hidden_dim, pen_out_dim)

        self._init_weights()

    def _init_weights(self) -> None:
        for name, param in self.named_parameters():
            if "weight_ih" in name:
                nn.init.xavier_uniform_(param)
            elif "weight_hh" in name:
                nn.init.orthogonal_(param)
            elif "bias" in name:
                nn.init.zeros_(param)

        nn.init.zeros_(self.pen_head.bias)
        nn.init.zeros_(self.pen_head.weight)
        nn.init.normal_(self.char_embedding.weight, mean=0.0, std=0.1)

    def forward(
        self,
        x: torch.Tensor,
        char_ids: torch.Tensor,
        char_mask: torch.Tensor | None = None,
        hidden: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> tuple[dict[str, torch.Tensor], tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
        """Step-by-step forward pass with windowed attention fed back into the LSTM.

        The LSTM must be unrolled timestep-by-timestep because the attention
        context at timestep t is computed from the LSTM output at t-1 and is
        part of the LSTM *input* at t. A single batched LSTM call could not
        express this recurrent dependency.

        Args:
            x: (B, T, 3)  normalized (delta_x, delta_y, pen_up)
            char_ids: (B, C)  character indices for the conditioning text
            char_mask: (B, C)  boolean mask; True = valid character
            hidden: (h_0, c_0) each (num_layers, B, hidden_dim), optional

        Returns:
            params: MDN parameters (same keys as MDNRNN)
            hidden: (h_T, c_T) for autoregressive sampling
            phi: (B, T, C) attention weights for visualization
        """
        B, T, _ = x.shape
        char_embeds = self.char_embedding(char_ids)  # (B, C, char_embed_dim)

        # Initial context: mean of character embeddings (no prior attention)
        if char_mask is not None:
            mask_float = char_mask.unsqueeze(-1).float()  # (B, C, 1)
            context = (char_embeds * mask_float).sum(dim=1, keepdim=True) / mask_float.sum(dim=1, keepdim=True).clamp(min=1)
        else:
            context = char_embeds.mean(dim=1, keepdim=True)  # (B, 1, char_embed_dim)

        all_lstm_out = []
        all_phi = []

        for t in range(T):
            x_t = x[:, t:t+1, :]  # (B, 1, 3)
            lstm_in = torch.cat([x_t, context], dim=-1)  # (B, 1, 3+char_embed_dim)

            lstm_out_t, hidden = self.lstm(lstm_in, hidden)
            all_lstm_out.append(lstm_out_t)

            # Compute attention for NEXT timestep
            if t < T - 1:
                context, phi_t = self.attention(lstm_out_t, char_embeds, char_mask)
                all_phi.append(phi_t)
            else:
                all_phi.append(torch.zeros(B, 1, char_embeds.size(1), device=x.device))

        lstm_out = torch.cat(all_lstm_out, dim=1)  # (B, T, hidden_dim)
        phi = torch.cat(all_phi, dim=1)  # (B, T, C)

        lstm_out = self.dropout(lstm_out)

        mdn_params = self.mdn_head(lstm_out)
        pen_logit = self.pen_head(lstm_out)

        M = self.num_mixtures
        mu_x, mu_y, sigma_x, sigma_y, rho, pi = mdn_params.split(M, dim=-1)

        sigma_x = torch.exp(sigma_x)
        sigma_y = torch.exp(sigma_y)
        rho = torch.tanh(rho)
        pi = torch.softmax(pi, dim=-1)
        pen_up = torch.sigmoid(pen_logit).squeeze(-1)

        params = {
            "mu_x": mu_x,
            "mu_y": mu_y,
            "sigma_x": sigma_x,
            "sigma_y": sigma_y,
            "rho": rho,
            "pi": pi,
            "pen_up": pen_up,
        }

        return params, hidden, phi

    def init_hidden(self, batch_size: int, device: torch.device | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        """Create zero-initialized hidden and cell states."""
        d = device or next(self.parameters()).device
        h = torch.zeros(self.num_layers, batch_size, self.hidden_dim, device=d)
        c = torch.zeros(self.num_layers, batch_size, self.hidden_dim, device=d)
        return h, c


# ---------------------------------------------------------------------------
# Sequence Discriminator (1D-CNN based)
# ---------------------------------------------------------------------------

class SequenceDiscriminator(nn.Module):
    """1D-CNN discriminator for real/fake stroke sequences.

    Takes a stroke sequence (B, T, 3) and outputs a real/fake probability.
    Uses strided 1D convolutions to progressively downsample the temporal
    dimension, followed by a final linear layer for classification.
    """

    def __init__(
        self,
        input_dim: int = 3,
        hidden_dim: int = 128,
        num_layers: int = 4,
        dropout: float = 0.2,
    ):
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        layers = []
        in_channels = input_dim

        for i in range(num_layers):
            out_channels = hidden_dim * (2 ** min(i, 2))
            kernel_size = 7 if i == 0 else 5
            stride = 2
            padding = (kernel_size - 1) // 2

            layers.append(
                nn.Conv1d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=padding,
                )
            )
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            layers.append(nn.Dropout(dropout))

            in_channels = out_channels

        self.conv_net = nn.Sequential(*layers)

        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Sequential(
            nn.Linear(in_channels, hidden_dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, a=0.2)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, a=0.2)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T, 3) stroke sequence

        Returns:
            real_prob: (B,) probability that each sequence is real
        """
        x = x.transpose(1, 2)  # (B, 3, T) for Conv1d
        features = self.conv_net(x)  # (B, C, T')
        pooled = self.global_pool(features).squeeze(-1)  # (B, C)
        logit = self.classifier(pooled).squeeze(-1)  # (B,)
        real_prob = torch.sigmoid(logit)
        return real_prob
