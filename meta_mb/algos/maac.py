from collections import OrderedDict
from numbers import Number
import numpy as np
import tensorflow as tf
from collections import OrderedDict
from .base import Algo
from meta_mb.utils import create_feed_dict, td_target
from pdb import set_trace as st
from meta_mb.logger import logger
from tensorflow.python.training import training_util
from random import randint
import time
import os.path as osp
import copy


class MAAC(Algo):
    """ Model-Augmented Actor-Critic (MAAC)"""

    def __init__(
            self,
            policy,
            env,
            dynamics_model,
            obs_dim,
            action_dim,
            Qs=None,
            Q_targets=None,
            discount=0.99,
            name="maac",
            learning_rate=3e-4,
            target_entropy=1.0,
            reward_scale=1.0,
            tau=5e-3,
            target_update_interval=1,
            action_prior='uniform',
            reparameterize=True,
            buffer_size=100000,
            sampler_batch_size=64,
            session=None,
            num_actions_per_next_observation=1,
            prediction_type = 'none',
            T = 0,
            actor_type=None,
            critic_type=None,
            H=0,
            model_used_ratio=1.0,
            experiment_name=None,
            exp_dir=None,
            **kwargs
    ):
        """
        Args:
            env (`SoftlearningEnv`): Environment used for training.
            policy: A policy function approximator.
            initial_exploration_policy: ('Policy'): A policy that we use
                for initial exploration which is not trained by the algorithm.
            Qs: Q-function approximators. The min of these
                approximators will be used. Usage of at least two Q-functions
                improves performance by reducing overestimation bias.
            pool (`PoolBase`): Replay pool to add gathered samples to.
            plotter (`QFPolicyPlotter`): Plotter instance to be used for
                visualizing Q-function during training.
            lr (`float`): Learning rate used for the function approximators.
            discount (`float`): Discount factor for Q-function updates.
            tau (`float`): Soft value function target update weight.
            target_update_interval ('int'): Frequency at which target network
                updates occur in iterations.
            reparameterize ('bool'): If True, we use a gradient estimator for
                the policy derived using the reparameterization trick. We use
                a likelihood ratio based estimator otherwise.
        """
        super(MAAC, self).__init__(policy)

        self.name = name
        self.policy = policy
        self.discount = discount
        self.dynamics_model = dynamics_model
        self.recurrent = False
        self.training_environment = env
        self.action_dim = action_dim
        self.obs_dim = obs_dim
        assert 0 <= target_entropy
        self.target_entropy = (-target_entropy * self.action_dim)

        self.policy_lr = learning_rate
        self.Q_lr = learning_rate
        self.tau = tau
        self.buffer_size = buffer_size
        self.sampler_batch_size = sampler_batch_size
        self.session = session or tf.keras.backend.get_session()
        self.Qs = Qs
        if Q_targets == None:
            self.Q_targets = self.Qs
        else:
            self.Q_targets = Q_targets
        self.reward_scale = reward_scale
        self.target_update_interval = target_update_interval
        self.num_actions_per_next_observation = num_actions_per_next_observation
        self.action_prior = action_prior
        self.reparameterize = reparameterize
        self.actor_type = actor_type
        self.model_used_ratio = model_used_ratio
        self.T = T
        self.H = H
        self.critic_type = critic_type
        self.experiment_name = experiment_name
        self.exp_dir = exp_dir
        self.prediction_type = prediction_type
        
        self.build_graph()

    def build_graph(self):
        self.training_ops = {}
        self.actor_ops = {}
        self._init_global_step()
        obs_ph, action_ph, next_obs_ph, terminal_ph, all_phs_dict = self._make_input_placeholders('',
                                                                                                  recurrent=False,
                                                                                                  next_obs=True)
        self.op_phs_dict = all_phs_dict
        self._init_actor_update()
        self._init_critic_update()
        self._init_diagnostics_ops()


    def _init_global_step(self):
        self.global_step = training_util.get_or_create_global_step()
        self.training_ops.update({
            'increment_global_step': training_util._increment_global_step(1)
        })

    def _make_input_placeholders(self, prefix='', recurrent=False, next_obs=False):
        """
        Args:
            prefix (str) : a string to prepend to the name of each variable
        Returns:
            (tuple) : a tuple containing lists of placeholders for each input type and meta task,
            and for convenience, a list containing all placeholders created
        """
        dist_info_specs = self.policy.distribution.dist_info_specs
        all_phs_dict = OrderedDict()

        # observation ph
        obs_shape = [None, self.policy.obs_dim]
        obs_ph = tf.placeholder(tf.float32, shape=obs_shape, name=prefix + 'obs')
        all_phs_dict['%s%s' % (prefix, 'observations')] = obs_ph

        # action ph
        action_shape = [None, self.policy.action_dim]
        action_ph = tf.placeholder(dtype=tf.float32, shape=action_shape, name=prefix + 'action')
        all_phs_dict['%s%s' % (prefix, 'actions')] = action_ph

        """add the placeholder for terminal here"""
        terminal_shape = [None] if not recurrent else [None, None]
        terminal_ph = tf.placeholder(dtype=tf.bool, shape=terminal_shape, name=prefix + 'dones')
        all_phs_dict['%s%s' % (prefix, 'dones')] = terminal_ph

        rewards_shape = [None] if not recurrent else [None, None]
        rewards_ph = tf.placeholder(dtype=tf.float32, shape=rewards_shape, name=prefix + 'rewards')
        all_phs_dict['%s%s' % (prefix, 'rewards')] = rewards_ph

        if not next_obs:
            return obs_ph, action_ph, all_phs_dict

        else:
            obs_shape = [None, self.policy.obs_dim]
            next_obs_ph = tf.placeholder(dtype=np.float32, shape=obs_shape, name=prefix + 'obs')
            all_phs_dict['%s%s' % (prefix, 'next_observations')] = next_obs_ph

        return obs_ph, action_ph, next_obs_ph, terminal_ph, all_phs_dict

    def step(self, obs, actions, shuffle=True, k = 1):
        next_observation = self.dynamics_model.predict_sym(obs, actions)
        if k != 1:
            next_observation = tf.tile(next_observation, [k, 1])
        dist_info_sym = self.policy.distribution_info_sym(next_observation)
        next_actions_var, dist_info_sym = self.policy.distribution.sample_sym(dist_info_sym)
        rewards = self.training_environment.tf_reward(obs, actions, next_observation)
        dones = self.training_environment.tf_termination_fn(obs, actions, next_observation)
        dones = tf.cast(dones, rewards.dtype)
        dones = tf.reshape(dones, [-1, 1])
        rewards = tf.reshape(rewards, [-1, 1])
        return next_observation, next_actions_var, rewards, dones, dist_info_sym

    def _get_q_target(self):
        next_observations_ph = self.op_phs_dict['next_observations']
        dist_info_sym = self.policy.distribution_info_sym(next_observations_ph)
        next_actions_var, dist_info_sym = self.policy.distribution.sample_sym(dist_info_sym)
        next_log_pis_var = self.policy.distribution.log_likelihood_sym(next_actions_var, dist_info_sym)
        next_log_pis_var = tf.expand_dims(next_log_pis_var, axis=-1)

        input_q_fun = tf.concat([next_observations_ph, next_actions_var], axis=-1)
        next_q_values = [Q.value_sym(input_var=input_q_fun) for Q in self.Q_targets]

        min_next_Q = tf.reduce_min(next_q_values, axis=0)
        next_values_var = min_next_Q - self.alpha * next_log_pis_var

        dones_ph = tf.cast(self.op_phs_dict['dones'], next_values_var.dtype)
        dones_ph = tf.expand_dims(dones_ph, axis=-1)
        rewards_ph = self.op_phs_dict['rewards']
        rewards_ph = tf.expand_dims(rewards_ph, axis=-1)

        if self.critic_type == None:
            target = self.q_target = td_target(
                reward=self.reward_scale * rewards_ph,
                discount=self.discount,
                next_value=(1 - dones_ph) * next_values_var)
            return tf.stop_gradient(target)

        elif self.critic_type == 'STEVE':
            num_models = self.dynamics_model.num_models
            obs = self.op_phs_dict['next_observations']
            dist_info_sym = self.policy.distribution_info_sym(obs)
            actions, _ = self.policy.distribution.sample_sym(dist_info_sym)
            rewards_var = self.reward_scale * rewards_ph
            rewards_var = tf.tile(rewards_var, [num_models, 1])
            target = td_target(
                reward=rewards_var,
                discount=self.discount,
                next_value=(1 - tf.tile(dones_ph, [num_models, 1])) * tf.tile(next_values_var, [num_models, 1]))
            targets = [target]
            for i in range(1, self.H + 1):
                next_observation = self.dynamics_model.predict_sym(obs, actions)
                dist_info_sym = self.policy.distribution_info_sym(next_observation)
                next_actions_var, _ = self.policy.distribution.sample_sym(dist_info_sym)

                expanded_obs = tf.tile(obs, [num_models, 1])
                expanded_actions = tf.tile(actions, [num_models, 1])
                expanded_next_observation = self.dynamics_model.predict_sym(expanded_obs, expanded_actions, shuffle = False)
                dist_info_sym = self.policy.distribution_info_sym(expanded_next_observation)
                expanded_next_actions_var, dist_info_sym = self.policy.distribution.sample_sym(dist_info_sym)
                next_log_pis_var = self.policy.distribution.log_likelihood_sym(expanded_next_actions_var, dist_info_sym)
                next_log_pis_var = tf.expand_dims(next_log_pis_var, axis=-1)

                rewards = self.training_environment.tf_reward(expanded_obs, expanded_actions, expanded_next_observation)
                rewards = tf.expand_dims(rewards, axis=-1)
                dones = tf.cast(self.training_environment.tf_termination_fn(expanded_obs, expanded_actions, expanded_next_observation), rewards.dtype)
                rewards_var = rewards_var + (1 - dones) * (self.discount ** i) * self.reward_scale * rewards

                input_q_fun = tf.concat([expanded_next_observation, expanded_next_actions_var], axis=-1)
                next_q_values = [Q.value_sym(input_var=input_q_fun) for Q in self.Q_targets]

                min_next_Q = tf.reduce_min(next_q_values, axis=0)
                next_values_var = min_next_Q - self.alpha * next_log_pis_var
                target = rewards_var + (self.discount ** (i+1)) * (1 - dones) * next_values_var
                targets.append(target)
                obs, actions = next_observation, next_actions_var
            targets = tf.stack(targets)
            rollout_frames = self.H + 1
            targets = tf.reshape(targets, [rollout_frames, num_models, -1, 1])

            """randomly choose a portion of the models. """
            if self.model_used_ratio < 1:
                num_models = int(self.model_used_ratio * self.dynamics_model.num_models)
                indices_lst = []
                for _ in range(rollout_frames):
                    indices = tf.range(self.dynamics_model.num_models)
                    indices = tf.random.shuffle(indices)
                    indices_lst.append(indices[:num_models])
                indices = tf.stack(indices_lst)
                indices = tf.reshape(indices, [-1])
                rollout_len_indices = tf.reshape(tf.tile(tf.expand_dims(tf.range(rollout_frames), axis = 1),
                                                         [1, num_models]), [-1])
                indices, rollout_len_indices = tf.expand_dims(indices, 1), tf.expand_dims(rollout_len_indices, 1)
                indices = tf.concat([rollout_len_indices, indices], axis = 1)
                targets = tf.gather_nd(targets, indices)
                targets = tf.reshape(targets, [rollout_frames, num_models, -1, 1])

            target_means, target_variances = tf.nn.moments(targets,1)
            target_confidence = 1./(target_variances + 1e-8)
            target_confidence *= tf.matrix_band_part(tf.ones([rollout_frames, 1, 1]), 0, -1)
            target_confidence = target_confidence / tf.reduce_sum(target_confidence, axis=0, keepdims=True)
            Q_target = self.q_target = tf.reduce_sum(target_means * target_confidence, 0)        #

        return tf.stop_gradient(Q_target)

    def _init_critic_update(self):
        """Create minimization operation for critic Q-function.
        Creates a `tf.optimizer.minimize` operation for updating
        critic Q-function with gradient descent, and appends it to
        `self._training_ops` attribute.
        """

        q_target = self._get_q_target()
        assert q_target.shape.as_list() == [None, 1]
        observations_ph = self.op_phs_dict['observations']
        actions_ph = self.op_phs_dict['actions']
        input_q_fun = tf.concat([observations_ph, actions_ph], axis=-1)

        q_values_var = self.q_values_var = [Q.value_sym(input_var=input_q_fun) for Q in self.Qs]
        q_losses = self.q_losses = [tf.losses.mean_squared_error(labels=q_target, predictions=q_value, weights=0.5)
                                    for q_value in q_values_var]

        self.q_optimizers = [tf.train.AdamOptimizer(learning_rate=self.Q_lr,
                                                    name='{}_{}_optimizer'.format(Q.name, i)
                                                    )
                             for i, Q in enumerate(self.Qs)]

        q_training_ops = [
            q_optimizer.minimize(loss=q_loss, var_list=list(Q.vfun_params.values()))
            for i, (Q, q_loss, q_optimizer)
            in enumerate(zip(self.Qs, q_losses, self.q_optimizers))]

        self.training_ops.update({'Q': tf.group(q_training_ops)})

    def _init_actor_update(self):
        """Create minimization operations for policy and entropy.
        Creates a `tf.optimizer.minimize` operations for updating
        policy and entropy with gradient descent, and adds them to
        `self._training_ops` attribute.
        """
        observations_ph = self.op_phs_dict['observations']
        dist_info_sym = self.policy.distribution_info_sym(observations_ph)
        actions_var, dist_info_sym = self.policy.distribution.sample_sym(dist_info_sym)
        log_pis_var = self.policy.distribution.log_likelihood_sym(actions_var, dist_info_sym)
        log_pis_var = tf.expand_dims(log_pis_var, axis=1)
        assert log_pis_var.shape.as_list() == [None, 1]

        log_alpha = tf.get_variable('log_alpha', dtype=tf.float32, initializer=0.0)
        alpha = tf.exp(log_alpha)

        if isinstance(self.target_entropy, Number):
            alpha_loss = -tf.reduce_mean(
                log_alpha * tf.stop_gradient(log_pis_var + self.target_entropy))
            self.log_pis_var = log_pis_var

            self.alpha_optimizer = tf.train.AdamOptimizer(self.policy_lr, name='alpha_optimizer')
            self.alpha_train_op = self.alpha_optimizer.minimize(loss=alpha_loss, var_list=[log_alpha])

            self.training_ops.update({
                'temperature_alpha': self.alpha_train_op
            })

        self.alpha = alpha

        if self.action_prior == 'normal':
            raise NotImplementedError
        elif self.action_prior == 'uniform':
            policy_prior_log_probs = 0.0

        """If actor_type is set to None, it equals to MBPO"""
        if self.actor_type == None:
            input_q_fun = tf.concat([observations_ph, actions_var], axis=-1)
            next_q_values = [Q.value_sym(input_var=input_q_fun) for Q in self.Qs]
            min_q_val_var = tf.reduce_min(next_q_values, axis=0)

        elif self.actor_type == 'MAAC':
            assert self.T >= 0
            obs = observations_ph
            actions = actions_var
            for i in range(self.T+1):
                next_observation = self.dynamics_model.predict_sym(obs, actions)
                dist_info_sym = self.policy.distribution_info_sym(next_observation)
                if self.prediction_type == 'none':
                    next_actions, _ = self.policy.distribution.sample_sym(dist_info_sym)
                expanded_next_observations = tf.tile(next_observation, [self.num_actions_per_next_observation, 1])
                dist_info_sym = self.policy.distribution_info_sym(expanded_next_observations)
                next_actions_var, _ = self.policy.distribution.sample_sym(dist_info_sym)
                rewards = self.training_environment.tf_reward(obs, actions, next_observation)
                rewards = tf.expand_dims(rewards, axis=-1)
                dones_next = tf.cast(self.training_environment.tf_termination_fn(obs, actions, next_observation), rewards.dtype)
                if i == 0 :
                    reward_values = (self.discount**(i)) * self.reward_scale * rewards
                else:
                    reward_values = (self.discount**(i)) * self.reward_scale * rewards * (1 - dones) + reward_values
                obs, actions = next_observation, next_actions
                dones = dones_next

            dones = tf.tile(dones, [self.num_actions_per_next_observation, 1])
            input_q_fun = tf.concat([expanded_next_observations, next_actions_var], axis=-1)
            next_q_values = [(self.discount ** (self.T + 1)) * (1-dones) * Q.value_sym(input_var=input_q_fun) for Q in self.Qs]
            next_q_values = [tf.reshape(value, [self.num_actions_per_next_observation, -1, 1]) for value in next_q_values]
            next_q_values = [tf.reduce_mean(value, axis = 0) for value in next_q_values]
            q_values_var = [reward_values + next_q_values[j] for j in range(2)]
            min_q_val_var = tf.reduce_min(q_values_var, axis=0)

        if self.reparameterize:
            policy_kl_losses = tf.reduce_mean(self.alpha * log_pis_var) - min_q_val_var - tf.reduce_mean(policy_prior_log_probs)
        else:
            raise NotImplementedError

        assert policy_kl_losses.shape.as_list() == [None, 1]

        self.policy_losses = policy_kl_losses
        policy_loss = tf.reduce_mean(policy_kl_losses)

        self.policy_optimizer = tf.train.AdamOptimizer(
            learning_rate=self.policy_lr,
            name="policy_optimizer")

        policy_train_op = self.policy_optimizer.minimize(
            loss=policy_loss,
            var_list=list(self.policy.policy_params.values()))

        self.training_ops.update({'policy_train_op': policy_train_op})

    def _init_diagnostics_ops(self):
        diagnosables = OrderedDict((
            ('Q_value', self.q_values_var),
            ('Q_loss', self.q_losses),
            ('policy_loss', self.policy_losses),
            ('alpha', self.alpha),
            ('Q_targets', self.q_target),
            ('scaled_rewards', self.reward_scale * self.op_phs_dict['rewards']),
            ('log_pis', self.log_pis_var)
        ))

        diagnostic_metrics = OrderedDict((
            ('mean', tf.reduce_mean),
            ('std', tf.math.reduce_std),
        ))
        self.diagnostics_ops = OrderedDict([
            ("%s-%s"%(key,metric_name), metric_fn(values))
            for key, values in diagnosables.items()
            for metric_name, metric_fn in diagnostic_metrics.items()
        ])

    def _update_target(self, tau=None):
        tau = tau or self.tau
        for Q, Q_target in zip(self.Qs, self.Q_targets):
            source_params = Q.get_param_values()
            target_params = Q_target.get_param_values()
            Q_target.set_params(OrderedDict([
                (param_name, tau * source + (1.0 - tau) * target_params[param_name])
                for param_name, source in source_params.items()
            ]))

    def do_training(self, timestep, batch, log=False):
        sess = tf.get_default_session()
        feed_dict = create_feed_dict(placeholder_dict=self.op_phs_dict,
                                     value_dict=batch)
        sess.run(self.training_ops, feed_dict)

        if log:
            diagnostics = sess.run({**self.diagnostics_ops}, feed_dict)
            for k, v in diagnostics.items():
                logger.logkv(k, v)
        if timestep % self.target_update_interval == 0:
            self._update_target()