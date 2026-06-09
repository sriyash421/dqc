from __future__ import annotations

from functools import partial
from typing import Any, Sequence

import flax
import flax.linen as nn
import jax
import jax.numpy as jnp
import ml_collections
import optax

from agents.bridge_calql_chunk import BridgeFilmResNet34, LanguageImageEncoder
from utils.flax_utils import ModuleDict, TrainState, nonpytree_field
from utils.networks import MLP, ensemblize


class ConditionalScalar(nn.Module):
    encoder: nn.Module
    hidden_dims: Sequence[int]
    layer_norm: bool = True

    @nn.compact
    def __call__(self, observations_and_goals, actions=None, train: bool = True):
        obs_enc = self.encoder(observations_and_goals, train=train)
        inputs = [obs_enc]
        if actions is not None:
            inputs.append(actions)
        x = jnp.concatenate(inputs, axis=-1)
        x = MLP(self.hidden_dims, activate_final=True, layer_norm=self.layer_norm)(x)
        value = nn.Dense(1)(x)
        return jnp.squeeze(value, -1)


class ConditionalVectorField(nn.Module):
    encoder: nn.Module
    hidden_dims: Sequence[int]
    action_dim: int
    layer_norm: bool = True

    @nn.compact
    def __call__(self, observations_and_goals, actions, times, train: bool = True):
        obs_enc = self.encoder(observations_and_goals, train=train)
        x = jnp.concatenate([obs_enc, actions, times], axis=-1)
        x = MLP(self.hidden_dims, activate_final=True, layer_norm=self.layer_norm)(x)
        return nn.Dense(self.action_dim)(x)


class BridgeDQCChunkAgent(flax.struct.PyTreeNode):
    """Bridge/SOAR image+MUSE DQC agent without CQL terms."""

    rng: Any
    network: Any
    config: Any = nonpytree_field()

    def _flatten_full_chunks(self, action_chunks):
        if action_chunks.ndim == 3 and action_chunks.shape[-1] == self.config["action_dim"]:
            return action_chunks.reshape((action_chunks.shape[0], -1))
        return action_chunks

    def _partial_chunks(self, batch):
        if "partial_action_chunks" in batch:
            partial = batch["partial_action_chunks"]
            if partial.ndim == 3:
                return partial.reshape((partial.shape[0], -1))
            return partial
        return batch["high_value_action_chunks"][..., : self.config["partial_action_dim"]]

    def forward_chunk_critic(self, observations, goals, action_chunks, *, params=None, train=True):
        action_chunks = self._flatten_full_chunks(action_chunks)
        if action_chunks.ndim == 3:
            return jax.vmap(
                lambda action: self.network.select("chunk_critic")((observations, goals), action, params=params, train=train),
                in_axes=1,
                out_axes=-1,
            )(action_chunks)
        return self.network.select("chunk_critic")((observations, goals), action_chunks, params=params, train=train)

    def forward_target_chunk_critic(self, observations, goals, action_chunks):
        action_chunks = self._flatten_full_chunks(action_chunks)
        if action_chunks.ndim == 3:
            return jax.vmap(
                lambda action: self.network.select("target_chunk_critic")((observations, goals), action, train=False),
                in_axes=1,
                out_axes=-1,
            )(action_chunks)
        return self.network.select("target_chunk_critic")((observations, goals), action_chunks, train=False)

    def forward_action_critic(self, observations, goals, actions, *, params=None, train=True):
        return self.network.select("action_critic")((observations, goals), actions, params=params, train=train)

    def forward_target_action_critic(self, observations, goals, actions):
        return self.network.select("target_action_critic")((observations, goals), actions, train=False)

    def forward_value(self, observations, goals, *, params=None, train=True):
        return self.network.select("value")((observations, goals), params=params, train=train)

    @staticmethod
    def expectile_loss(pred, target, expectile):
        weight = jnp.where(target >= pred, expectile, 1.0 - expectile)
        return weight * jnp.square(pred - target)

    @staticmethod
    def quantile_loss(pred, target, quantile):
        weight = jnp.where(target >= pred, quantile, 1.0 - quantile)
        return weight * jnp.abs(pred - target)

    def chunk_critic_loss(self, batch, params):
        next_v = self.forward_value(batch["high_value_next_observations"], batch["high_value_goals"], train=False)
        target_q = batch["high_value_rewards"] + (
            self.config["discount"] ** batch["high_value_backup_horizon"]
        ) * batch["high_value_masks"] * next_v
        target_q = jax.lax.stop_gradient(target_q)
        pred_q = self.forward_chunk_critic(
            batch["observations"],
            batch["high_value_goals"],
            batch["high_value_action_chunks"],
            params=params,
            train=True,
        )
        loss = jnp.mean(jnp.square(pred_q - target_q[None]))
        return loss, {
            "critic_loss": loss,
            "target_q": target_q.mean(),
            "q_mean": pred_q.mean(),
            "q_min": pred_q.min(),
            "q_max": pred_q.max(),
        }

    def action_critic_and_value_loss(self, batch, params):
        partial_actions = self._partial_chunks(batch)
        full_qs = self.forward_chunk_critic(
            batch["observations"],
            batch["high_value_goals"],
            batch["high_value_action_chunks"],
            train=False,
        )
        if self.config["q_agg"] == "mean":
            full_q = full_qs.mean(axis=0)
        elif self.config["q_agg"] == "min":
            full_q = full_qs.min(axis=0)
        else:
            raise ValueError(f"Unsupported q_agg: {self.config['q_agg']}")
        full_q = jax.lax.stop_gradient(full_q)

        action_q = self.forward_action_critic(
            batch["observations"],
            batch["high_value_goals"],
            partial_actions,
            params=params,
            train=True,
        )
        valid = batch["valids"][..., self.config["policy_chunk_size"] - 1]
        action_critic_loss = jnp.mean(
            self.expectile_loss(action_q, full_q[None], self.config["distill_expectile"]) * valid[None]
        )

        target_action_qs = self.forward_target_action_critic(batch["observations"], batch["high_value_goals"], partial_actions)
        if self.config["q_agg"] == "mean":
            target_action_q = target_action_qs.mean(axis=0)
        else:
            target_action_q = target_action_qs.min(axis=0)
        target_action_q = jax.lax.stop_gradient(target_action_q)
        value = self.forward_value(batch["observations"], batch["high_value_goals"], params=params, train=True)
        if self.config["backup_loss"] == "quantile":
            value_loss = jnp.mean(self.quantile_loss(value, target_action_q, self.config["backup_quantile"]) * valid)
        elif self.config["backup_loss"] == "expectile":
            value_loss = jnp.mean(self.expectile_loss(value, target_action_q, self.config["backup_quantile"]) * valid)
        else:
            raise ValueError(f"Unsupported backup_loss: {self.config['backup_loss']}")
        return action_critic_loss + value_loss, {
            "critic_loss": action_critic_loss,
            "value_loss": value_loss,
            "q_mean": action_q.mean(),
            "q_min": action_q.min(),
            "q_max": action_q.max(),
            "v_mean": value.mean(),
            "v_min": value.min(),
            "v_max": value.max(),
            "adv": (target_action_q - value).mean(),
        }

    def actor_loss(self, batch, params, rng):
        partial_actions = self._partial_chunks(batch)
        batch_size = partial_actions.shape[0]
        rng, x_rng, t_rng = jax.random.split(rng, 3)
        x_0 = jax.random.normal(x_rng, partial_actions.shape)
        x_1 = partial_actions
        t = jax.random.uniform(t_rng, (batch_size, 1))
        x_t = (1.0 - t) * x_0 + t * x_1
        vel = x_1 - x_0
        pred = self.network.select("actor_bc")(
            (batch["observations"], batch["high_value_goals"]),
            x_t,
            t,
            params=params,
            train=True,
        )
        valid = batch["valids"][..., self.config["policy_chunk_size"] - 1]
        loss = jnp.mean(jnp.mean(jnp.square(pred - vel), axis=-1) * valid)
        return loss, {"bc_flow_loss": loss}

    @jax.jit
    def total_loss(self, batch, params, rng):
        rng, actor_rng = jax.random.split(rng)
        chunk_loss, chunk_info = self.chunk_critic_loss(batch, params)
        action_value_loss, action_value_info = self.action_critic_and_value_loss(batch, params)
        actor_loss, actor_info = self.actor_loss(batch, params, actor_rng)
        info = {f"chunk_critic/{k}": v for k, v in chunk_info.items()}
        info.update({f"action_critic/{k}": v for k, v in action_value_info.items()})
        info.update({f"actor/{k}": v for k, v in actor_info.items()})
        total = chunk_loss + action_value_loss + self.config["actor_loss_weight"] * actor_loss
        return total, info

    def target_update(self, network, source_name, target_name):
        new_target_params = jax.tree_util.tree_map(
            lambda p, tp: p * self.config["tau"] + tp * (1.0 - self.config["tau"]),
            network.params[f"modules_{source_name}"],
            network.params[f"modules_{target_name}"],
        )
        network.params[f"modules_{target_name}"] = new_target_params

    @staticmethod
    def _update(self, batch):
        new_rng, rng = jax.random.split(self.rng)

        def loss_fn(params):
            return self.total_loss(batch, params, rng)

        new_network, info = self.network.apply_loss_fn(loss_fn)
        self.target_update(new_network, "chunk_critic", "target_chunk_critic")
        self.target_update(new_network, "action_critic", "target_action_critic")
        return self.replace(network=new_network, rng=new_rng), info

    @jax.jit
    def update(self, batch):
        return self._update(self, batch)

    @partial(jax.jit, static_argnames=("q_agg",))
    def score_chunks(self, observations, language_embeddings, action_chunks, q_agg="min"):
        goals = {"language": language_embeddings} if not isinstance(language_embeddings, dict) else language_embeddings
        qs = self.forward_chunk_critic(observations, goals, action_chunks, train=False)
        if q_agg == "min":
            return qs.min(axis=0)
        if q_agg == "mean":
            return qs.mean(axis=0)
        raise ValueError(f"Unsupported q_agg: {q_agg}")

    @jax.jit
    def compute_flow_actions(self, observations, goals, noises):
        actions = noises
        for i in range(self.config["flow_steps"]):
            t = jnp.full((observations["image"].shape[0], 1), i / self.config["flow_steps"])
            vels = self.network.select("actor_bc")((observations, goals), actions, t, train=False)
            actions = actions + vels / self.config["flow_steps"]
        return jnp.clip(actions, -1.0, 1.0)

    @classmethod
    def create(cls, seed, example_batch, config):
        config = ml_collections.ConfigDict(config)
        if "target_update_rate" in config:
            config.tau = config.target_update_rate
        if "num_critics" in config:
            config.num_qs = config.num_critics
        action_dim = int(example_batch["actions"].shape[-1])
        full_chunk_dim = int(example_batch["high_value_action_chunks"].shape[-1])
        expected_full_chunk_dim = int(config["backup_horizon"]) * action_dim
        if full_chunk_dim != expected_full_chunk_dim:
            raise ValueError(f"Expected full chunk dim {expected_full_chunk_dim}, got {full_chunk_dim}")
        if int(config["policy_chunk_size"]) != 1:
            raise ValueError("BridgeDQCChunkAgent currently requires policy_chunk_size=1")
        partial_action_dim = int(config["policy_chunk_size"]) * action_dim

        def make_encoder():
            return LanguageImageEncoder(
                BridgeFilmResNet34(
                    pooling_method=config["encoder_kwargs"]["pooling_method"],
                    add_spatial_coordinates=config["encoder_kwargs"]["add_spatial_coordinates"],
                )
            )

        critic_cls = partial(ConditionalScalar, encoder=make_encoder(), hidden_dims=config["critic_hidden_dims"], layer_norm=config["layer_norm"])
        target_critic_cls = partial(
            ConditionalScalar,
            encoder=make_encoder(),
            hidden_dims=config["critic_hidden_dims"],
            layer_norm=config["layer_norm"],
        )
        action_critic_cls = partial(
            ConditionalScalar,
            encoder=make_encoder(),
            hidden_dims=config["action_critic_hidden_dims"],
            layer_norm=config["layer_norm"],
        )
        target_action_critic_cls = partial(
            ConditionalScalar,
            encoder=make_encoder(),
            hidden_dims=config["action_critic_hidden_dims"],
            layer_norm=config["layer_norm"],
        )
        value_def = ConditionalScalar(encoder=make_encoder(), hidden_dims=config["value_hidden_dims"], layer_norm=config["layer_norm"])
        actor_def = ConditionalVectorField(
            encoder=make_encoder(),
            hidden_dims=config["actor_hidden_dims"],
            action_dim=partial_action_dim,
            layer_norm=config["actor_layer_norm"],
        )
        networks = ModuleDict(
            {
                "chunk_critic": ensemblize(critic_cls, config["num_qs"])(name="chunk_critic"),
                "target_chunk_critic": ensemblize(target_critic_cls, config["num_qs"])(name="target_chunk_critic"),
                "action_critic": ensemblize(action_critic_cls, config["num_qs"])(name="action_critic"),
                "target_action_critic": ensemblize(target_action_critic_cls, config["num_qs"])(name="target_action_critic"),
                "value": value_def,
                "actor_bc": actor_def,
            }
        )
        observations_and_goals = (example_batch["observations"], example_batch["high_value_goals"])
        partial_actions = example_batch["high_value_action_chunks"][..., :partial_action_dim]
        ex_times = example_batch["actions"][..., :1]
        network_args = {
            "chunk_critic": (observations_and_goals, example_batch["high_value_action_chunks"]),
            "target_chunk_critic": (observations_and_goals, example_batch["high_value_action_chunks"]),
            "action_critic": (observations_and_goals, partial_actions),
            "target_action_critic": (observations_and_goals, partial_actions),
            "value": (observations_and_goals,),
            "actor_bc": (observations_and_goals, partial_actions, ex_times),
        }
        rng = jax.random.PRNGKey(seed)
        rng, init_rng = jax.random.split(rng)
        params = networks.init(init_rng, **network_args)["params"]
        params["modules_target_chunk_critic"] = params["modules_chunk_critic"]
        params["modules_target_action_critic"] = params["modules_action_critic"]
        tx = optax.adam(config["lr"])
        network = TrainState.create(networks, params, tx=tx)
        config.action_dim = action_dim
        config.chunk_action_dim = full_chunk_dim
        config.partial_action_dim = partial_action_dim
        return cls(rng, network=network, config=flax.core.FrozenDict(**config))


def get_config():
    return ml_collections.ConfigDict(
        dict(
            agent_name="bridge_dqc_chunk",
            lr=3e-4,
            batch_size=256,
            discount=0.98,
            tau=0.005,
            target_update_rate=0.005,
            num_qs=2,
            num_critics=2,
            q_agg="min",
            backup_horizon=8,
            policy_chunk_size=1,
            backup_quantile=0.9,
            distill_expectile=0.8,
            backup_loss="quantile",
            best_of_n=32,
            flow_steps=10,
            actor_loss_weight=1.0,
            critic_hidden_dims=(1024, 1024, 1024, 1024, 1024),
            action_critic_hidden_dims=(1024, 1024, 1024, 1024, 1024),
            value_hidden_dims=(1024, 1024, 1024, 1024, 1024),
            actor_hidden_dims=(1024, 1024, 1024, 1024, 1024),
            layer_norm=True,
            actor_layer_norm=True,
            encoder="resnetv1-34-bridge-film",
            encoder_kwargs=dict(pooling_method="avg", add_spatial_coordinates=True),
            text_processor="muse_embedding",
            action_dim=ml_collections.config_dict.placeholder(int),
            chunk_action_dim=ml_collections.config_dict.placeholder(int),
            partial_action_dim=ml_collections.config_dict.placeholder(int),
            action_clip_eps=1e-5,
        )
    )
