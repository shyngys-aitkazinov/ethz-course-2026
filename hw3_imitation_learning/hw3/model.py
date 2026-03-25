"""Model definitions for SO-100 imitation policies."""

from __future__ import annotations

import abc
from typing import Literal, TypeAlias

import torch
from torch import nn


class BasePolicy(nn.Module, metaclass=abc.ABCMeta):
    """Base class for action chunking policies."""

    def __init__(self, state_dim: int, action_dim: int, chunk_size: int) -> None:
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.chunk_size = chunk_size

    @abc.abstractmethod
    def compute_loss(
        self, state: torch.Tensor, action_chunk: torch.Tensor
    ) -> torch.Tensor:
        """Compute training loss for a batch."""
        raise NotImplementedError

    @abc.abstractmethod
    def sample_actions(self, state: torch.Tensor) -> torch.Tensor:
        """Generate a chunk of actions with shape (batch, chunk_size, action_dim)."""
        raise NotImplementedError


# TODO: Students implement ObstaclePolicy here.
class ObstaclePolicy(BasePolicy):
    """Predicts action chunks with an MSE loss.

    A simple MLP that maps a state vector to a flat action chunk
    (chunk_size * action_dim) and reshapes to (B, chunk_size, action_dim).
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        chunk_size: int = 16,
        d_model: int = 256,
        depth: int = 3,
        dropout: float = 0.1,
        activation: str = "gelu",
        use_layer_norm: bool = False,
        gripper_weight: float = 2.0,
    ) -> None:
        super().__init__(state_dim, action_dim, chunk_size)
        self.d_model = d_model
        self.depth = depth
        self.gripper_weight = gripper_weight

        act_fn = {"relu": nn.ReLU, "gelu": nn.GELU, "tanh": nn.Tanh, "silu": nn.SiLU}
        if activation not in act_fn:
            raise ValueError(f"Unsupported activation: {activation}")
        make_act = act_fn[activation]

        layers: list[nn.Module] = [nn.Linear(state_dim, d_model)]
        if use_layer_norm:
            layers.append(nn.LayerNorm(d_model))
        layers.append(make_act())

        for _ in range(depth - 1):
            layers.append(nn.Linear(d_model, d_model))
            if use_layer_norm:
                layers.append(nn.LayerNorm(d_model))
            layers.append(make_act())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))

        layers.append(nn.Linear(d_model, chunk_size * action_dim))
        self.mlp = nn.Sequential(*layers)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Return predicted action chunk of shape (B, chunk_size, action_dim)."""
        flat = self.mlp(state)
        return flat.view(-1, self.chunk_size, self.action_dim)

    def compute_loss(
        self, state: torch.Tensor, action_chunk: torch.Tensor
    ) -> torch.Tensor:
        pred = self.forward(state)  # (B, H, D)
        sq_err = (pred - action_chunk) ** 2  # (B, H, D)
        # Upweight gripper (last dim) relative to position dims
        weights = torch.ones(self.action_dim, device=pred.device)
        weights[-1] = self.gripper_weight
        return (sq_err * weights).mean()

    def sample_actions(self, state: torch.Tensor) -> torch.Tensor:
        return self.forward(state)


# TODO: Students implement MultiTaskPolicy here.
class MultiTaskPolicy(BasePolicy):
    """Goal-conditioned policy for the multicube scene.

    Supports two modes controlled by ``use_film``:
    - **True** (default): FiLM (Feature-wise Linear Modulation). The goal one-hot
      (last ``goal_dim`` dims of the state) is split off and fed through
      a small conditioning network that produces per-layer scale (γ) and
      shift (β) vectors.  Each hidden layer is modulated:
      ``h = γ * layer(h) + β``, giving the goal direct control over every
      layer's activations.
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        chunk_size: int = 16,
        d_model: int = 256,
        depth: int = 3,
        dropout: float = 0.1,
        activation: str = "gelu",
        use_layer_norm: bool = False,
        gripper_weight: float = 2.0,
        use_film: bool = True,
        goal_dim: int = 3,
    ) -> None:
        super().__init__(state_dim, action_dim, chunk_size)
        self.d_model = d_model
        self.depth = depth
        self.gripper_weight = gripper_weight
        self.use_film = use_film
        self.goal_dim = goal_dim

        act_fn = {"relu": nn.ReLU, "gelu": nn.GELU, "tanh": nn.Tanh, "silu": nn.SiLU}
        if activation not in act_fn:
            raise ValueError(f"Unsupported activation: {activation}")
        make_act = act_fn[activation]

        if use_film:
            # FiLM mode: trunk takes state WITHOUT goal one-hot
            trunk_input_dim = state_dim - goal_dim
            self.input_proj = nn.Linear(trunk_input_dim, d_model)

            # Hidden layers (no Sequential — we apply FiLM between them)
            self.hidden_layers = nn.ModuleList()
            self.hidden_norms = nn.ModuleList() if use_layer_norm else None
            self.hidden_acts = nn.ModuleList()
            self.hidden_drops = nn.ModuleList()
            for _ in range(depth - 1):
                self.hidden_layers.append(nn.Linear(d_model, d_model))
                if use_layer_norm:
                    self.hidden_norms.append(nn.LayerNorm(d_model))
                self.hidden_acts.append(make_act())
                self.hidden_drops.append(
                    nn.Dropout(dropout) if dropout > 0 else nn.Identity()
                )

            self.output_head = nn.Linear(d_model, chunk_size * action_dim)

            # FiLM conditioning network: goal → per-layer (γ, β)
            # One generator per hidden layer + one for the input projection.
            # Single linear layer keeps param count close to the plain MLP variant.
            # Zero-initialized so the model starts as an identity (no modulation).
            n_film_layers = depth  # input proj + (depth-1) hidden
            self.film_generators = nn.ModuleList([
                nn.Linear(goal_dim, 2 * d_model)
                for _ in range(n_film_layers)
            ])
            for gen in self.film_generators:
                nn.init.zeros_(gen.weight)
                nn.init.zeros_(gen.bias)
            self.input_act = make_act()
        else:
            # Plain MLP mode (identical to ObstaclePolicy)
            layers: list[nn.Module] = [nn.Linear(state_dim, d_model)]
            if use_layer_norm:
                layers.append(nn.LayerNorm(d_model))
            layers.append(make_act())

            for _ in range(depth - 1):
                layers.append(nn.Linear(d_model, d_model))
                if use_layer_norm:
                    layers.append(nn.LayerNorm(d_model))
                layers.append(make_act())
                if dropout > 0:
                    layers.append(nn.Dropout(dropout))

            layers.append(nn.Linear(d_model, chunk_size * action_dim))
            self.mlp = nn.Sequential(*layers)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Return predicted action chunk of shape (B, chunk_size, action_dim)."""
        if self.use_film:
            # Split state into features and goal one-hot
            x = state[:, : -self.goal_dim]
            goal = state[:, -self.goal_dim :]

            # Input projection + FiLM modulation
            h = self.input_proj(x)
            gamma, beta = self.film_generators[0](goal).chunk(2, dim=-1)
            h = (1.0 + gamma) * h + beta  # γ offset by 1 for identity init
            h = self.input_act(h)

            # Hidden layers with FiLM
            for i, layer in enumerate(self.hidden_layers):
                h = layer(h)
                if self.hidden_norms is not None:
                    h = self.hidden_norms[i](h)
                gamma, beta = self.film_generators[i + 1](goal).chunk(2, dim=-1)
                h = (1.0 + gamma) * h + beta
                h = self.hidden_acts[i](h)
                h = self.hidden_drops[i](h)

            flat = self.output_head(h)
        else:
            flat = self.mlp(state)

        return flat.view(-1, self.chunk_size, self.action_dim)

    def compute_loss(
        self, state: torch.Tensor, action_chunk: torch.Tensor
    ) -> torch.Tensor:
        pred = self.forward(state)  # (B, H, D)
        sq_err = (pred - action_chunk) ** 2  # (B, H, D)
        weights = torch.ones(self.action_dim, device=pred.device)
        weights[-1] = self.gripper_weight
        return (sq_err * weights).mean()

    def sample_actions(self, state: torch.Tensor) -> torch.Tensor:
        return self.forward(state)


PolicyType: TypeAlias = Literal["obstacle", "multitask"]


def build_policy(
    policy_type: PolicyType,
    *,
    state_dim: int,
    action_dim: int,
    chunk_size: int = 16,
    d_model: int = 256,
    depth: int = 3,
    dropout: float = 0.1,
    activation: str = "gelu",
    use_layer_norm: bool = False,
    gripper_weight: float = 2.0,
    use_film: bool = True,
    goal_dim: int = 3,
) -> BasePolicy:
    if policy_type == "obstacle":
        return ObstaclePolicy(
            state_dim=state_dim,
            action_dim=action_dim,
            chunk_size=chunk_size,
            d_model=d_model,
            depth=depth,
            dropout=dropout,
            activation=activation,
            use_layer_norm=use_layer_norm,
            gripper_weight=gripper_weight,
        )
    if policy_type == "multitask":
        return MultiTaskPolicy(
            state_dim=state_dim,
            action_dim=action_dim,
            chunk_size=chunk_size,
            d_model=d_model,
            depth=depth,
            dropout=dropout,
            activation=activation,
            use_layer_norm=use_layer_norm,
            gripper_weight=gripper_weight,
            use_film=use_film,
            goal_dim=goal_dim,
        )
    raise ValueError(f"Unknown policy type: {policy_type}")
