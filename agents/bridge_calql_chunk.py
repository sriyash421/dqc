from __future__ import annotations

from functools import partial
from typing import Any, Sequence

import distrax
import flax
import flax.linen as nn
import jax
import jax.numpy as jnp
import ml_collections
import optax

from utils.flax_utils import ModuleDict, TrainState, nonpytree_field
from utils.networks import MLP, ensemblize


class FilmConditioning(nn.Module):
    @nn.compact
    def __call__(self, conv_filters: jnp.ndarray, conditioning: jnp.ndarray):
        projected_cond_add = nn.Dense(
            features=conv_filters.shape[-1],
            kernel_init=nn.initializers.zeros,
            bias_init=nn.initializers.zeros,
        )(conditioning)
        projected_cond_mult = nn.Dense(
            features=conv_filters.shape[-1],
            kernel_init=nn.initializers.zeros,
            bias_init=nn.initializers.zeros,
        )(conditioning)
        return conv_filters * (1 + projected_cond_add[..., None, None, :]) + projected_cond_mult[..., None, None, :]


class AddSpatialCoordinates(nn.Module):
    dtype: Any = jnp.float32

    @nn.compact
    def __call__(self, x):
        h, w = x.shape[-3], x.shape[-2]
        ys, xs = jnp.meshgrid(jnp.linspace(-1.0, 1.0, h), jnp.linspace(-1.0, 1.0, w), indexing="ij")
        grid = jnp.stack([ys, xs], axis=-1).astype(self.dtype)
        if x.ndim == 4:
            grid = jnp.broadcast_to(grid, (x.shape[0], h, w, 2))
        return jnp.concatenate([x, grid], axis=-1)


class ResNetBlock(nn.Module):
    filters: int
    strides: tuple[int, int] = (1, 1)
    act: Any = nn.swish

    @nn.compact
    def __call__(self, x):
        residual = x
        conv = partial(nn.Conv, use_bias=False, kernel_init=nn.initializers.kaiming_normal())
        norm = partial(nn.GroupNorm, num_groups=4, epsilon=1e-5)
        y = conv(self.filters, (3, 3), self.strides)(x)
        y = norm()(y)
        y = self.act(y)
        y = conv(self.filters, (3, 3))(y)
        y = norm(scale_init=nn.initializers.zeros)(y)
        if residual.shape != y.shape:
            residual = conv(self.filters, (1, 1), self.strides, name="conv_proj")(residual)
            residual = norm(name="norm_proj")(residual)
        return self.act(residual + y)


class BridgeFilmResNet34(nn.Module):
    pooling_method: str = "avg"
    add_spatial_coordinates: bool = True

    @nn.compact
    def __call__(self, observations: jnp.ndarray, cond_var: jnp.ndarray, train: bool = True):
        del train
        x = observations.astype(jnp.float32) / 127.5 - 1.0
        if self.add_spatial_coordinates:
            x = AddSpatialCoordinates()(x)
        x = nn.Conv(64, (7, 7), (2, 2), padding=[(3, 3), (3, 3)], use_bias=False, kernel_init=nn.initializers.kaiming_normal(), name="conv_init")(x)
        x = nn.GroupNorm(num_groups=4, epsilon=1e-5, name="norm_init")(x)
        x = nn.swish(x)
        x = nn.max_pool(x, (3, 3), strides=(2, 2), padding="SAME")
        for stage, blocks in enumerate((3, 4, 6, 3)):
            for block in range(blocks):
                stride = (2, 2) if stage > 0 and block == 0 else (1, 1)
                x = ResNetBlock(64 * 2**stage, strides=stride, act=nn.swish)(x)
                x = FilmConditioning()(x, cond_var)
        if self.pooling_method != "avg":
            raise ValueError(f"Unsupported pooling_method for BridgeFilmResNet34: {self.pooling_method}")
        return jnp.mean(x, axis=(-3, -2))


class LanguageImageEncoder(nn.Module):
    image_encoder: nn.Module

    @nn.compact
    def __call__(self, observations_and_goals, train: bool = True):
        observations, goals = observations_and_goals
        return self.image_encoder(observations["image"], cond_var=goals["language"], train=train)


class Critic(nn.Module):
    encoder: nn.Module
    hidden_dims: Sequence[int]
    layer_norm: bool = True

    @nn.compact
    def __call__(self, observations_and_goals, actions, train: bool = True):
        obs_enc = self.encoder(observations_and_goals, train=train)
        x = jnp.concatenate([obs_enc, actions], axis=-1)
        x = MLP(self.hidden_dims, activate_final=True, layer_norm=self.layer_norm)(x)
        q = nn.Dense(1)(x)
        return jnp.squeeze(q, -1)


class GaussianPolicy(nn.Module):
    encoder: nn.Module
    hidden_dims: Sequence[int]
    action_dim: int
    layer_norm: bool = True
    tanh_squash_distribution: bool = True

    @nn.compact
    def __call__(self, observations_and_goals, train: bool = True):
        obs_enc = self.encoder(observations_and_goals, train=train)
        x = MLP(self.hidden_dims, activate_final=True, layer_norm=self.layer_norm)(obs_enc)
        mean = nn.Dense(self.action_dim)(x)
        log_std = nn.Dense(self.action_dim)(x)
        std = jnp.exp(jnp.clip(log_std, -5.0, 2.0))
        dist = distrax.MultivariateNormalDiag(mean, std)
        if not self.tanh_squash_distribution:
            return dist
        return distrax.Transformed(dist, distrax.Block(distrax.Tanh(), ndims=1))


class BridgeCalQLChunkAgent(flax.struct.PyTreeNode):
    rng: Any
    network: Any
    config: Any = nonpytree_field()

    def forward_critic(self, observations, goals, actions, *, params=None, train=True):
        del train
        if actions.ndim == 3:
            return jax.vmap(
                lambda action: self.network.select("chunk_critic")((observations, goals), action, params=params),
                in_axes=1,
                out_axes=-1,
            )(actions)
        return self.network.select("chunk_critic")((observations, goals), actions, params=params)

    def forward_target_critic(self, observations, goals, actions):
        if actions.ndim == 3:
            return jax.vmap(
                lambda action: self.network.select("target_chunk_critic")((observations, goals), action),
                in_axes=1,
                out_axes=-1,
            )(actions)
        return self.network.select("target_chunk_critic")((observations, goals), actions)

    def sample_policy_actions(self, observations, goals, rng, *, repeat=None, params=None, train=True):
        dist = self.network.select("actor")((observations, goals), params=params, train=train)
        if repeat is None:
            return dist.sample_and_log_prob(seed=rng)
        actions, log_probs = dist.sample_and_log_prob(seed=rng, sample_shape=repeat)
        return jnp.swapaxes(actions, 0, 1), jnp.swapaxes(log_probs, 0, 1)

    def critic_loss(self, batch, params, rng):
        batch_size = batch["high_value_rewards"].shape[0]
        rng, next_rng, cql_rng = jax.random.split(rng, 3)
        next_actions, next_log_probs = self.sample_policy_actions(
            batch["high_value_next_observations"],
            batch["high_value_goals"],
            next_rng,
            repeat=self.config["cql_n_actions"] if self.config["cql_max_target_backup"] else None,
            train=False,
        )
        target_next_qs = self.forward_target_critic(batch["high_value_next_observations"], batch["high_value_goals"], next_actions)
        target_next_q = jnp.min(target_next_qs, axis=0)
        if self.config["cql_max_target_backup"]:
            idxs = jnp.argmax(target_next_q, axis=-1)
            target_next_q = target_next_q[jnp.arange(batch_size), idxs]
            next_log_probs = next_log_probs[jnp.arange(batch_size), idxs]
        if self.config["backup_entropy"]:
            target_next_q = target_next_q - self.config["temperature"] * next_log_probs
        target_q = batch["high_value_rewards"] + (self.config["discount"] ** batch["high_value_backup_horizon"]) * batch["high_value_masks"] * target_next_q

        q_pred = self.forward_critic(batch["observations"], batch["high_value_goals"], batch["high_value_action_chunks"], params=params)
        td_loss = jnp.mean((q_pred - target_q[None]) ** 2)

        cql_q_diff, cql_info = self.cql_q_diff(batch, cql_rng, params)
        cql_loss = jnp.clip(cql_q_diff, self.config["cql_clip_diff_min"], self.config["cql_clip_diff_max"]).mean()
        critic_loss = td_loss + self.config["cql_alpha"] * cql_loss
        return critic_loss, {
            "critic_loss": critic_loss,
            "td_loss": td_loss,
            "target_q": target_q.mean(),
            "q_mean": q_pred.mean(),
            "q_min": q_pred.min(),
            "q_max": q_pred.max(),
            "cql_loss": cql_loss,
            "cql_diff": cql_q_diff.mean(),
            **cql_info,
        }

    def cql_q_diff(self, batch, rng, params):
        batch_size = batch["high_value_rewards"].shape[0]
        action_dim = batch["high_value_action_chunks"].shape[-1]
        q_pred = self.forward_critic(batch["observations"], batch["high_value_goals"], batch["high_value_action_chunks"], params=params)
        rng, random_rng, current_rng, next_rng = jax.random.split(rng, 4)
        random_actions = jax.random.uniform(random_rng, (batch_size, self.config["cql_n_actions"], action_dim), minval=-1.0, maxval=1.0)
        current_actions, current_log_probs = self.sample_policy_actions(batch["observations"], batch["high_value_goals"], current_rng, repeat=self.config["cql_n_actions"], params=params)
        next_actions, next_log_probs = self.sample_policy_actions(batch["high_value_next_observations"], batch["high_value_goals"], next_rng, repeat=self.config["cql_n_actions"], params=params)
        all_actions = jnp.concatenate([random_actions, current_actions, next_actions], axis=1)
        q_samples = self.forward_critic(batch["observations"], batch["high_value_goals"], all_actions, params=params)
        mc_lower_bound = jnp.repeat(batch["mc_returns"][:, None], self.config["cql_n_actions"] * 2, axis=1)
        q_pi = q_samples[:, :, self.config["cql_n_actions"]:]
        calql_bound_rate = jnp.mean(q_pi < mc_lower_bound[None])
        q_pi = jnp.maximum(q_pi, mc_lower_bound[None])
        q_samples = jnp.concatenate([q_samples[:, :, :self.config["cql_n_actions"]], q_pi], axis=-1)
        random_density = jnp.log(0.5**action_dim)
        importance_prob = jnp.concatenate(
            [
                jnp.broadcast_to(random_density, (batch_size, self.config["cql_n_actions"])),
                current_log_probs,
                next_log_probs,
            ],
            axis=1,
        )
        q_samples = q_samples - importance_prob[None]
        cql_ood = jax.scipy.special.logsumexp(q_samples / self.config["cql_temp"], axis=-1) * self.config["cql_temp"]
        return cql_ood - q_pred, {"cql_ood_values": cql_ood.mean(), "calql_bound_rate": calql_bound_rate}

    def actor_loss(self, batch, params, rng):
        dist = self.network.select("actor")((batch["observations"], batch["high_value_goals"]), params=params, train=True)
        log_probs = dist.log_prob(batch["high_value_action_chunks"])
        loss = -jnp.mean(log_probs)
        return loss, {"actor_bc_loss": loss, "actor_log_prob": log_probs.mean()}

    @jax.jit
    def total_loss(self, batch, params, rng):
        rng, critic_rng, actor_rng = jax.random.split(rng, 3)
        critic_loss, critic_info = self.critic_loss(batch, params, critic_rng)
        actor_loss, actor_info = self.actor_loss(batch, params, actor_rng)
        info = {f"chunk_critic/{k}": v for k, v in critic_info.items()}
        info.update({f"actor/{k}": v for k, v in actor_info.items()})
        return critic_loss + self.config["actor_loss_weight"] * actor_loss, info

    def target_update(self, network):
        new_target_params = jax.tree_util.tree_map(
            lambda p, tp: p * self.config["tau"] + tp * (1.0 - self.config["tau"]),
            network.params["modules_chunk_critic"],
            network.params["modules_target_chunk_critic"],
        )
        network.params["modules_target_chunk_critic"] = new_target_params

    @staticmethod
    def _update(self, batch):
        new_rng, rng = jax.random.split(self.rng)

        def loss_fn(params):
            return self.total_loss(batch, params, rng)

        new_network, info = self.network.apply_loss_fn(loss_fn)
        self.target_update(new_network)
        return self.replace(network=new_network, rng=new_rng), info

    @jax.jit
    def update(self, batch):
        return self._update(self, batch)

    @partial(jax.jit, static_argnames=("q_agg",))
    def score_chunks(self, observations, goals, action_chunks, q_agg="min"):
        qs = self.forward_critic(observations, goals, action_chunks, train=False)
        if q_agg == "min":
            return qs.min(axis=0)
        if q_agg == "mean":
            return qs.mean(axis=0)
        raise ValueError(f"Unsupported q_agg: {q_agg}")

    @classmethod
    def create(cls, seed, example_batch, config):
        rng = jax.random.PRNGKey(seed)
        rng, init_rng = jax.random.split(rng)
        action_dim = example_batch["actions"].shape[-1]
        chunk_action_dim = example_batch["high_value_action_chunks"].shape[-1]
        if config["backup_horizon"] * action_dim != chunk_action_dim:
            raise ValueError("high_value_action_chunks does not match backup_horizon * action_dim")
        if config["policy_chunk_size"] != 1:
            raise ValueError("BridgeCalQLChunkAgent requires policy_chunk_size=1")
        if not config["use_chunk_critic"]:
            raise ValueError("BridgeCalQLChunkAgent requires use_chunk_critic=True")

        def make_encoder():
            return LanguageImageEncoder(
                BridgeFilmResNet34(
                    pooling_method=config["encoder_kwargs"]["pooling_method"],
                    add_spatial_coordinates=config["encoder_kwargs"]["add_spatial_coordinates"],
                )
            )

        critic_cls = partial(Critic, encoder=make_encoder(), hidden_dims=config["critic_hidden_dims"], layer_norm=config["layer_norm"])
        target_critic_cls = partial(Critic, encoder=make_encoder(), hidden_dims=config["critic_hidden_dims"], layer_norm=config["layer_norm"])
        critic_def = ensemblize(critic_cls, config["num_qs"])(name="chunk_critic")
        target_critic_def = ensemblize(target_critic_cls, config["num_qs"])(name="target_chunk_critic")
        actor_def = GaussianPolicy(
            encoder=make_encoder(),
            hidden_dims=config["actor_hidden_dims"],
            action_dim=chunk_action_dim,
            layer_norm=config["actor_layer_norm"],
            tanh_squash_distribution=True,
        )
        networks = ModuleDict({"chunk_critic": critic_def, "target_chunk_critic": target_critic_def, "actor": actor_def})
        network_args = {
            "chunk_critic": ((example_batch["observations"], example_batch["high_value_goals"]), example_batch["high_value_action_chunks"]),
            "target_chunk_critic": ((example_batch["observations"], example_batch["high_value_goals"]), example_batch["high_value_action_chunks"]),
            "actor": ((example_batch["observations"], example_batch["high_value_goals"]),),
        }
        params = networks.init(init_rng, **network_args)["params"]
        params["modules_target_chunk_critic"] = params["modules_chunk_critic"]
        tx = optax.adam(config["lr"])
        network = TrainState.create(networks, params, tx=tx)
        config["action_dim"] = action_dim
        config["chunk_action_dim"] = chunk_action_dim
        return cls(rng, network=network, config=flax.core.FrozenDict(**config))


def get_config():
    return ml_collections.ConfigDict(
        dict(
            agent_name="bridge_calql_chunk",
            lr=3e-4,
            batch_size=1024,
            discount=0.98,
            tau=0.005,
            num_qs=2,
            backup_horizon=8,
            policy_chunk_size=1,
            use_chunk_critic=True,
            actor_loss_weight=1.0,
            critic_hidden_dims=(1024, 1024, 1024, 1024, 1024),
            actor_hidden_dims=(1024, 1024, 1024, 1024, 1024),
            layer_norm=True,
            actor_layer_norm=True,
            cql_n_actions=4,
            cql_max_target_backup=True,
            cql_temp=1.0,
            cql_alpha=5.0,
            cql_clip_diff_min=-jnp.inf,
            cql_clip_diff_max=jnp.inf,
            backup_entropy=False,
            temperature=1.0,
            encoder="resnetv1-34-bridge-film",
            encoder_kwargs=dict(pooling_method="avg", add_spatial_coordinates=True),
            text_processor="muse_embedding",
            action_dim=ml_collections.config_dict.placeholder(int),
            chunk_action_dim=ml_collections.config_dict.placeholder(int),
        )
    )
