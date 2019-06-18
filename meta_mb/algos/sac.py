from collections import OrderedDict
from numbers import Number

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from flatten_dict import flatten
from collections import OrderedDict
from .base import Algo
from meta_mb.utils import create_feed_dict
"""===============added==========start============"""
from pdb import set_trace as st
from meta_mb.utils import create_feed_dict

"""===========this should be put somewhere in the utils=============="""
from tensorflow.python.training import training_util
from distutils.version import LooseVersion

if LooseVersion(tf.__version__) > LooseVersion("2.00"):
    from tensorflow import nest
else:
    from tensorflow.contrib.framework import nest

"""flattern nested sequence, tuple, or dict."""
def flatten_input_structure(inputs):
    inputs_flat = nest.flatten(inputs)
    return inputs_flat


def log_pis_fn(inputs):
    shift, log_scale_diag, actions = inputs
    base_distribution = tfp.distributions.MultivariateNormalDiag(
        loc=tf.zeros(output_shape),
        scale_diag=tf.ones(output_shape))
    bijector = tfp.bijectors.Chain((
        squash_bijector,
        tfp.bijectors.Affine(
            shift=shift,
            scale_diag=tf.exp(log_scale_diag)),
    ))
    distribution = (
        tfp.distributions.ConditionalTransformedDistribution(
            distribution=base_distribution,
            bijector=bijector))

    log_pis = distribution.log_prob(actions)[:, None]
    return log_pis

class SquashBijector(tfp.bijectors.Bijector):
    def __init__(self, validate_args=False, name="tanh"):
        super(SquashBijector, self).__init__(
            forward_min_event_ndims=0,
            validate_args=validate_args,
            name=name)

    def _forward(self, x):
        return tf.nn.tanh(x)

    def _inverse(self, y):
        return tf.atanh(y)

    def _forward_log_det_jacobian(self, x):
        return 2. * (np.log(2.) - x - tf.nn.softplus(-2. * x))
"""===============added==========end============"""

def td_target(reward, discount, next_value):
    return reward + discount * next_value


class SAC(Algo):
    """Soft Actor-Critic (SAC)
    References
    ----------
    [1] Tuomas Haarnoja*, Aurick Zhou*, Kristian Hartikainen*, George Tucker,
        Sehoon Ha, Jie Tan, Vikash Kumar, Henry Zhu, Abhishek Gupta, Pieter
        Abbeel, and Sergey Levine. Soft Actor-Critic Algorithms and
        Applications. arXiv preprint arXiv:1812.05905. 2018.
    """

    def __init__(
            self,
            policy,
            Qs,
            training_environment,
            discount=0.99,
            name="sac",
            learning_rate=3e-4,
            target_entropy='auto',
            # evaluation_environment,
            # samplers,
            # pool,
            # plotter=None,
            reward_scale=1.0,
            tau=5e-3,
            target_update_interval=1,
            action_prior='uniform',
            reparameterize=True,
            buffer_size=100000,
            sampler_batch_size=64,
            # save_full_state=False,
            # n_epochs=1000,
            # train_every_n_steps=1,
            # n_train_repeat=1,
            # max_train_repeat_per_timestep=5,
            # n_initial_exploration_steps=0,
            # initial_exploration_policy=None,
            # epoch_length=1000,
            # eval_n_episodes=10,
            # eval_deterministic=True,
            # eval_render_kwargs=None,
            # video_save_frequency=0,
            session=None,
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

        super(SAC, self).__init__(policy)

        """===============added==========start============"""
        self.name = name
        self.policy = policy
        self.discount = discount
        # self.recurrent = getattr(self.policy, 'recurrent', False)
        self.recurrent = False
        self.training_environment = training_environment
        self.target_entropy = (-np.prod(self.training_environment.action_space.shape) if target_entropy == 'auto' else target_entropy)
        self.learning_rate = learning_rate
        self.policy_lr = learning_rate
        self.Q_lr = learning_rate
        self.tau = tau
        self._dataset = None
        self.buffer_size = buffer_size
        self.sampler_batch_size = sampler_batch_size
        # self.sampler = sampler
        # self._n_epochs = n_epochs
        # self._n_train_repeat = n_train_repeat
        # self._max_train_repeat_per_timestep = max(max_train_repeat_per_timestep, n_train_repeat)
        # self._train_every_n_steps = train_every_n_steps
        # self._epoch_length = epoch_length
        # self._n_initial_exploration_steps = n_initial_exploration_steps
        # self._initial_exploration_policy = initial_exploration_policy
        # self._eval_n_episodes = eval_n_episodes
        # self._eval_deterministic = eval_deterministic
        # self._video_save_frequency = video_save_frequency
        # self._eval_render_kwargs = eval_render_kwargs or {}
        self.session = session or tf.keras.backend.get_session()
        self._squash = True
        # self._epoch = 0
        # self._timestep = 0
        # self._num_train_steps = 0
        """===============added==========end============"""
        self.Qs = Qs
        self.Q_targets = tuple(tf.keras.models.clone_model(Q) for Q in Qs)
        self.reward_scale = reward_scale
        self.target_update_interval = target_update_interval
        self.action_prior = action_prior
        self.reparameterize = reparameterize
        self._optimization_keys = ['observations', 'actions', 'next_observations', 'dones', 'rewards', 'advantages']

        self.build_graph()

    def build_graph(self):
        self.op_phs_dict = OrderedDict()
        self.training_ops = {}
        self._init_global_step()
        obs_ph, action_ph, next_obs_ph, ad_ph, terminal_ph, all_phs_dict = self._make_input_placeholders('train', recurrent=False, next_obs=True)
        self.op_phs_dict.update(all_phs_dict)
        # distribution_info_vars = self.policy.distribution_info_sym(obs_ph)
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
        obs_ph = tf.placeholder(tf.float32, shape=obs_shape, name=prefix + '_obs')
        all_phs_dict['%s_%s' % (prefix, 'observations')] = obs_ph

        # action ph
        action_shape = [None, self.policy.action_dim]
        action_ph = tf.placeholder(dtype=tf.float32, shape=action_shape, name=prefix + '_action')
        all_phs_dict['%s_%s' % (prefix, 'actions')] = action_ph

        # advantage ph
        adv_shape = [None] if not recurrent else [None, None]
        adv_ph = tf.placeholder(dtype=tf.float32, shape=adv_shape, name=prefix + '_advantage')
        all_phs_dict['%s_%s' % (prefix, 'advantages')] = adv_ph

        """add the placeholder for terminal here"""
        terminal_shape = [None] if not recurrent else [None, None]
        terminal_ph = tf.placeholder(dtype=tf.bool, shape=terminal_shape, name=prefix + '_dones')
        all_phs_dict['%s_%s' % (prefix, 'dones')] = terminal_ph

        rewards_shape = [None] if not recurrent else [None, None]
        rewards_ph = tf.placeholder(dtype=tf.float32, shape=rewards_shape, name=prefix + '_rewards')
        all_phs_dict['%s_%s' % (prefix, 'rewards')] = rewards_ph

        if not next_obs:
            return obs_ph, action_ph, adv_ph, dist_info_ph_dict, all_phs_dict

        else:
            obs_shape = [None, self.policy.obs_dim]
            next_obs_ph = tf.placeholder(dtype=np.float32, shape=obs_shape, name=prefix + '_obs')
            all_phs_dict['%s_%s' % (prefix, 'next_observations')] = next_obs_ph

        return obs_ph, action_ph, next_obs_ph, adv_ph, terminal_ph, all_phs_dict


    def _get_Q_target(self):
        prefix = 'train_'
        policy_inputs = self.op_phs_dict[prefix + 'next_observations']
        dist = self.policy.distribution_info_sym(policy_inputs)
        mean = dist['mean']
        sd = tf.math.exp(dist['log_std'])
        # next_actions = mean + tf.random.normal(shape=tf.shape(mean), stddev=tf.math.exp(dist['log_std']))
        distribution = tfp.distributions.MultivariateNormalDiag(loc = mean, scale_diag = sd)
        next_actions = distribution.sample()
        next_log_pis = self.policy._dist.log_likelihood_sym(next_actions, dist)
        next_log_pis = tf.expand_dims(next_log_pis, axis = 1)


        next_Q_observations = self.op_phs_dict[prefix + 'next_observations']
        next_Q_inputs = flatten_input_structure({
            'next_Q_observations':next_Q_observations, 'actions': next_actions})
        next_Qs_values = tuple(Q(next_Q_inputs) for Q in self.Q_targets)

        min_next_Q = tf.reduce_min(next_Qs_values, axis=0)
        next_values = min_next_Q - self.alpha * next_log_pis

        terminals = tf.cast(self.op_phs_dict[prefix + 'dones'], next_values.dtype)
        terminals = tf.expand_dims(terminals, axis = 1)
        rewards = self.op_phs_dict[prefix + 'rewards']
        rewards = tf.expand_dims(rewards, axis = 1)
        Q_target = td_target(
            reward=self.reward_scale * rewards,
            discount=self.discount,
            next_value=(1 - terminals) * next_values)

        return tf.stop_gradient(Q_target)

    def _init_critic_update(self):
        """Create minimization operation for critic Q-function.
        Creates a `tf.optimizer.minimize` operation for updating
        critic Q-function with gradient descent, and appends it to
        `self._training_ops` attribute.
        See Equations (5, 6) in [1], for further information of the
        Q-function update rule.
        """
        prefix = 'train_'
        Q_target = self._get_Q_target()
        assert Q_target.shape.as_list() == [None, 1]
        Q_observations = self.op_phs_dict[prefix + 'observations']

        Q_inputs = flatten_input_structure({
            'Q_observations':Q_observations, 'actions': self.op_phs_dict[prefix + 'actions']})

        Q_values = self.Q_values = tuple(Q(Q_inputs) for Q in self.Qs)
        Q_losses = self.Q_losses = tuple(
            tf.losses.mean_squared_error(
                labels=Q_target, predictions=Q_value, weights=0.5)
            for Q_value in Q_values)


        self.Q_optimizers = tuple(
            tf.train.AdamOptimizer(
                learning_rate=self.Q_lr,
                name='{}_{}_optimizer'.format(Q._name, i)
            ) for i, Q in enumerate(self.Qs))

        Q_training_ops = tuple(
            Q_optimizer.minimize(loss=Q_loss, var_list=Q.trainable_variables)
            for i, (Q, Q_loss, Q_optimizer)
            in enumerate(zip(self.Qs, Q_losses, self.Q_optimizers)))

        self.training_ops.update({'Q': tf.group(Q_training_ops)})

    def _init_actor_update(self):
        """Create minimization operations for policy and entropy.
        Creates a `tf.optimizer.minimize` operations for updating
        policy and entropy with gradient descent, and adds them to
        `self._training_ops` attribute.
        See Section 4.2 in [1], for further information of the policy update,
        and Section 5 in [1] for further information of the entropy update.
        """
        prefix = 'train'
        original_inputs = self.op_phs_dict['%s_%s' % (prefix, 'observations')]
        dist = self.policy.distribution_info_sym(original_inputs)
        mean = dist['mean']
        # actions =  mean + tf.random.normal(shape=tf.shape(mean), stddev=tf.math.exp(dist['log_std']))
        sd = tf.math.exp(dist['log_std'])
        distribution = tfp.distributions.MultivariateNormalDiag(loc = mean, scale_diag = sd)
        actions = distribution.sample()
        log_pis = self.policy._dist.log_likelihood_sym(actions, dist)
        log_pis = tf.expand_dims(log_pis, axis = 1)

        assert log_pis.shape.as_list() == [None, 1]

        log_alpha = self.log_alpha = tf.compat.v1.get_variable(
            'log_alpha',
            dtype=tf.float32,
            initializer=0.0)
        alpha = tf.exp(log_alpha)

        if isinstance(self.target_entropy, Number):
            alpha_loss = -tf.reduce_mean(
                log_alpha * tf.stop_gradient(log_pis + self.target_entropy))
            self.log_pis = log_pis

            self.alpha_optimizer = tf.compat.v1.train.AdamOptimizer(
                self.policy_lr, name='alpha_optimizer')
            self.alpha_train_op = self.alpha_optimizer.minimize(
                loss=alpha_loss, var_list=[log_alpha])

            self.training_ops.update({
                'temperature_alpha': self.alpha_train_op
            })

        self.alpha = alpha

        if self.action_prior == 'normal':
            policy_prior = tfp.distributions.MultivariateNormalDiag(
                loc=tf.zeros(self.action_shape),
                scale_diag=tf.ones(self.action_shape))
            policy_prior_log_probs = policy_prior.log_prob(actions)
        elif self.action_prior == 'uniform':
            policy_prior_log_probs = 0.0

        """check dimension here"""
        Q_observations = self.op_phs_dict['%s_%s' % (prefix, 'observations')]

        Q_inputs = flatten_input_structure({
            'Q_observations':Q_observations, 'actions': actions})
        Q_log_targets = tuple(Q(Q_inputs) for Q in self.Qs)
        min_Q_log_target = tf.reduce_min(Q_log_targets, axis=0)

        if self.reparameterize:
            policy_kl_losses = (alpha * log_pis - min_Q_log_target- policy_prior_log_probs)
        else:
            raise NotImplementedError

        policy_kl_losses = tf.expand_dims(policy_kl_losses, axis = 1)
        assert policy_kl_losses.shape.as_list() == [None, 1]

        self.policy_losses = policy_kl_losses
        policy_loss = tf.reduce_mean(policy_kl_losses)

        self.policy_optimizer = tf.compat.v1.train.AdamOptimizer(
            learning_rate=self.policy_lr,
            name="policy_optimizer")

        policy_train_op = self.policy_optimizer.minimize(
            loss=policy_loss,
            var_list=list(self.policy.policy_params.values()))

        self.training_ops.update({'policy_train_op': policy_train_op})

    def _init_diagnostics_ops(self):
        diagnosables = OrderedDict((
            ('Q_value', self.Q_values),
            ('Q_loss', self.Q_losses),
            ('policy_loss', self.policy_losses),
            ('alpha', self.alpha)
        ))

        diagnostic_metrics = OrderedDict((
            ('mean', tf.reduce_mean),
            ('std', lambda x: tfp.stats.stddev(x, sample_axis=None)),
        ))
        self.diagnostics_ops = OrderedDict([
            ("%s-%s"%(key,metric_name), metric_fn(values))
            for key, values in diagnosables.items()
            for metric_name, metric_fn in diagnostic_metrics.items()
        ])


    def _update_target(self, tau=None):
        tau = tau or self.tau
        for Q, Q_target in zip(self.Qs, self.Q_targets):
            source_params = Q.get_weights()
            target_params = Q_target.get_weights()
            Q_target.set_weights([
                tau * source + (1.0 - tau) * target
                for source, target in zip(source_params, target_params)
            ])



    def _do_training(self, iteration, batch):
        """Runs the operations for updating training and target ops."""
        feed_dict = create_feed_dict(placeholder_dict=self.op_phs_dict, value_dict=batch)
        self.session.run(self.training_ops, feed_dict)
        if iteration % self.target_update_interval == 0:
            self._update_target()


    def optimize_policy(self, samples_data, timestep, training_batch, log=True):
        """===============added==========start============"""
        sess = tf.get_default_session()
        prefix = 'train'
        random_batch = training_batch.random_batch(self.sampler_batch_size, prefix)
        self._do_training(iteration=timestep, batch=random_batch)
        #
        # sess = tf.get_default_session()
        # input_dict = self._extract_input_dict(samples_data, self._optimization_keys, prefix)
        # if self._dataset is None:
        #     self._dataset = input_dict
        # else:
        #     for k, v in input_dict.items():
        #         n_new_samples = len(v)
        #         n_max = self.buffer_size - n_new_samples
        #         self._dataset[k] = np.concatenate([self._dataset[k][-n_max:], v], axis=0)
        # num_elements = len(list(self._dataset.values())[0])
        # """===============added==========end=============="""
        # self._do_training(num_elements, iteration=timestep, batch=input_dict)

        # self._num_train_steps += self._n_train_repeat
        # self._train_steps_this_epoch += self._n_train_repeat
