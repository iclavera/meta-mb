from meta_mb.utils.networks.mlp import create_mlp, forward_mlp
from meta_mb.baselines.linear_baseline import LinearFeatureBaseline
from meta_mb.replay_buffers.gc_simple_replay_buffer import SimpleReplayBuffer
from meta_mb.samplers.gc_mb_sample_processor import ModelSampleProcessor
from meta_mb.utils.utils import remove_scope_from_name
from meta_mb.utils import create_feed_dict
from meta_mb.utils import Serializable
from meta_mb.logger import logger

import tensorflow as tf
import numpy as np
from collections import OrderedDict


def td_target(reward, discount, next_value):
    return reward + discount * next_value


class ValueFunction(Serializable):
    def __init__(self,
                 env,
                 vfun_idx,
                 reward_scale,
                 discount,
                 learning_rate,
                 gae_lambda,
                 normalize_adv,
                 positive_adv,
                 hidden_sizes,
                 hidden_nonlinearity,
                 output_nonlinearity,
                 batch_size,
                 num_grad_steps,
                 max_replay_buffer_size,
                 ):

        Serializable.quick_init(self, locals())

        self.obs_dim = env.obs_dim
        self.goal_dim = env.goal_dim
        self.name = f"ve_{vfun_idx}"
        self.hidden_sizes = hidden_sizes
        self.hidden_nonlinearity = hidden_nonlinearity
        self.output_nonlinearity = output_nonlinearity
        self.batch_size = batch_size
        self.reward_scale = reward_scale
        self.discount = discount
        self.learning_rate = learning_rate
        self.num_grad_steps = num_grad_steps

        baseline = LinearFeatureBaseline()

        self.replay_buffer = SimpleReplayBuffer(
            env_spec=env,
            max_replay_buffer_size=max_replay_buffer_size
        )

        self.sample_processor = ModelSampleProcessor(
            reward_fn=env.reward,
            achieved_goal_fn=env.get_achieved_goal,
            baseline=baseline,
            replay_k=-1,
            discount=discount,
            gae_lambda=gae_lambda,
            normalize_adv=normalize_adv,
            positive_adv=positive_adv,
        )

        self.vfun_params = None
        self.input_var = None
        self._assign_ops = None

        self._build_placeholder()
        self._build_network()
        self._init_value_iteration_update()
        self._init_diagnostics_ops()

    def _build_placeholder(self):
        with tf.variable_scope(self.name, reuse=tf.AUTO_REUSE):
            self.obs_ph = tf.placeholder(dtype=tf.float32, shape=(None, self.obs_dim), name='obs')
            self.next_obs_ph = tf.placeholder(dtype=tf.float32, shape=(None, self.obs_dim), name='next_obs')
            self.reward_ph = tf.placeholder(dtype=tf.float32, shape=(None,), name='rewards')
            self.terminal_ph = tf.placeholder(dtype=tf.bool, shape=(None,), name='dones')
            self.goal_ph = tf.placeholder(dtype=tf.float32, shape=(None, self.goal_dim), name='goals')

            self.op_phs_dict = dict(observations=self.obs_ph, next_observations=self.next_obs_ph, rewards=self.reward_ph,
                                    dones=self.terminal_ph, goals=self.goal_ph)

    def _build_network(self):
        """
        Builds computational graph for policy
        """
        with tf.variable_scope(self.name, reuse=tf.AUTO_REUSE):
            # build the actual policy network
            self.input_var, self.output_var = create_mlp(name='v_network',
                                                         output_dim=1,
                                                         hidden_sizes=self.hidden_sizes,
                                                         hidden_nonlinearity=self.hidden_nonlinearity,
                                                         output_nonlinearity=self.output_nonlinearity,
                                                         input_dim=(None, self.obs_dim + self.goal_dim,),
                                                         )

            # save the policy's trainable variables in dicts
            # current_scope = tf.get_default_graph().get_name_scope()
            current_scope = self.name
            trainable_vfun_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=current_scope)
            self.vfun_params = OrderedDict([(remove_scope_from_name(var.name, current_scope), var) for var in trainable_vfun_vars])

            self.vfun_params_ph = self._create_placeholders_for_vars(scope=current_scope + f"/v_network ")
            self.vfun_params_keys = self.vfun_params_ph.keys()
            self._vfun_np = lambda inputs: tf.get_default_session().run(self.output_var, feed_dict={self.input_var: inputs})

    def _init_value_iteration_update(self):
        with tf.variable_scope(self.name, reuse=tf.AUTO_REUSE):

            value_input_var = tf.concat([self.obs_ph, self.goal_ph], axis=1)
            _, v_predict = create_mlp(name='v_network',
                                      output_dim=1,
                                      hidden_sizes=self.hidden_sizes,
                                      hidden_nonlinearity=self.hidden_nonlinearity,
                                      output_nonlinearity=self.output_nonlinearity,
                                      input_var=value_input_var,
                                      reuse=True,
                                      )

            next_value_input_var = tf.concat([self.next_obs_ph, self.goal_ph], axis=1)
            _, next_value_var = create_mlp(name='v_network',
                                      output_dim=1,
                                      hidden_sizes=self.hidden_sizes,
                                      hidden_nonlinearity=self.hidden_nonlinearity,
                                      output_nonlinearity=self.output_nonlinearity,
                                      input_var=next_value_input_var,
                                      reuse=True,
                                      )

            reward_var = tf.expand_dims(self.reward_ph, axis=1)
            reward_var = tf.cast(reward_var, tf.float32)

            done_var = tf.expand_dims(self.terminal_ph, axis=1)
            done_var = tf.cast(done_var, tf.float32)

            v_target = td_target(
                reward=self.reward_scale * reward_var,
                discount=self.discount,
                next_value=(1-done_var) * next_value_var,
            )

            loss = tf.losses.mean_squared_error(labels=v_target, predictions=v_predict)
            opt = tf.train.AdamOptimizer(learning_rate=self.learning_rate, name=f"{self.name}_optimizer")
            training_op = opt.minimize(loss=loss, var_list=list(self.vfun_params.values()))

            self.loss = loss
            self.training_op = training_op
            self.v_predict = v_predict
            self.v_target = v_target

    def _init_diagnostics_ops(self):
        with tf.variable_scope(self.name, reuse=tf.AUTO_REUSE):
            diagnosables = OrderedDict((
                ('loss', self.loss),
                ('target', self.v_target),
                ('predict', self.v_predict),
                ('scaled_rewards', self.reward_scale * self.reward_ph),
            ))

            diagnostic_metrics = OrderedDict((
                ('mean', tf.reduce_mean),
                # ('std', lambda x: tfp.stats.stddev(x, sample_axis=None)),
            ))
            self.diagnostics_ops = OrderedDict([
                ("%s-%s"%(key,metric_name), metric_fn(values))
                for key, values in diagnosables.items()
                for metric_name, metric_fn in diagnostic_metrics.items()
            ])

    def compute_values(self, obs, goals):
        return self._vfun_np(np.concatenate([obs, goals], axis=1)).flatten()

    def train(self, on_policy_paths, itr, log=True, log_prefix='vc-'):
        samples_data = self.sample_processor.process_samples(on_policy_paths, eval=False, log='all', log_prefix='train-')
        self.replay_buffer.add_samples(samples_data['goals'], samples_data['observations'], samples_data['actions'],
                                       samples_data['rewards'], samples_data['dones'],
                                       samples_data['next_observations'])

        sess = tf.get_default_session()
        for _ in range(self.num_grad_steps):
            feed_dict = create_feed_dict(placeholder_dict=self.op_phs_dict,
                                         value_dict=self.replay_buffer.random_batch(batch_size=self.batch_size))
            _ = sess.run(self.training_op, feed_dict=feed_dict)
            if log:
                logger.logkv('train-Itr', itr)
                diagnostics = sess.run({**self.diagnostics_ops}, feed_dict)
                for k, v in diagnostics.items():
                    logger.logkv_mean(log_prefix + k, v)

    def value_sym(self, input_var, params=None):
        """
        Return the symbolic distribution information about the actions.

        Args:
            obs_var (placeholder) : symbolic variable for observations
            params (dict) : a dictionary of placeholders or vars with the parameters of the MLP

        Returns:
            (dict) : a dictionary of tf placeholders for the policy output distribution
        """
        if params is None:
            with tf.variable_scope(self.name, reuse=tf.AUTO_REUSE):
                input_var, output_var = create_mlp(name='v_network',
                                                   output_dim=1,
                                                   hidden_sizes=self.hidden_sizes,
                                                   hidden_nonlinearity=self.hidden_nonlinearity,
                                                   output_nonlinearity=self.output_nonlinearity,
                                                   input_var=input_var,
                                                   reuse=True,
                                                   )
        else:
            raise NotImplementedError
            # input_var, output_var = forward_mlp(output_dim=1,
            #                                     hidden_sizes=self.hidden_sizes,
            #                                     hidden_nonlinearity=self.hidden_nonlinearity,
            #                                     output_nonlinearity=self.output_nonlinearity,
            #                                     input_var=input_var,
            #                                     mlp_params=params,
            #                                     )

        return output_var

    # def distribution_info_keys(self, obs, state_infos):
    #     """
    #     Args:
    #         obs (placeholder) : symbolic variable for observations
    #         state_infos (dict) : a dictionary of placeholders that contains information about the
    #         state of the policy at the time it received the observation
    #
    #     Returns:
    #         (dict) : a dictionary of tf placeholders for the policy output distribution
    #     """
    #     raise ["mean", "log_std"]

    def _create_placeholders_for_vars(self, scope, graph_keys=tf.GraphKeys.TRAINABLE_VARIABLES):
        var_list = tf.get_collection(graph_keys, scope=scope)
        placeholders = []
        for var in var_list:
            var_name = remove_scope_from_name(var.name, scope.split('/')[0])
            placeholders.append((var_name, tf.placeholder(tf.float32, shape=var.shape, name="%s_ph" % var_name)))
        return OrderedDict(placeholders)

    def get_shared_param_values(self):
        state = dict()
        state['network_params'] = self.get_param_values()
        return state

    def set_shared_params(self, state):
        self.set_params(state['network_params'])

    def get_params(self):
        """
        Get the tf.Variables representing the trainable weights of the network (symbolic)

        Returns:
            (dict) : a dict of all trainable Variables
        """
        return self.vfun_params

    def get_param_values(self):
        """
        Gets a list of all the current weights in the network (in original code it is flattened, why?)

        Returns:
            (list) : list of values for parameters
        """
        param_values = tf.get_default_session().run(self.vfun_params)
        return param_values

    def set_params(self, vfun_params):
        """
        Sets the parameters for the graph

        Args:
            policy_params (dict): of variable names and corresponding parameter values
        """
        # from pdb import set_trace as st
        # print(self.get_params().keys(), vfun_params.keys())
        # st()
        assert all([k1 == k2 for k1, k2 in zip(self.get_params().keys(), vfun_params.keys())]), \
            "parameter keys must match with variable"

        if self._assign_ops is None:
            assign_ops, assign_phs = [], []
            for var in self.get_params().values():
                assign_placeholder = tf.placeholder(dtype=var.dtype)
                assign_op = tf.assign(var, assign_placeholder)
                assign_ops.append(assign_op)
                assign_phs.append(assign_placeholder)
            self._assign_ops = assign_ops
            self._assign_phs = assign_phs
        feed_dict = dict(zip(self._assign_phs, vfun_params.values()))
        tf.get_default_session().run(self._assign_ops, feed_dict=feed_dict)

    def __getstate__(self):
        state = {
            'init_args': Serializable.__getstate__(self),
            'network_params': self.get_param_values()
        }
        return state

    def __setstate__(self, state):
        Serializable.__setstate__(self, state['init_args'])
        self.set_params(state['network_params'])
