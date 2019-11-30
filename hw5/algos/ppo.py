from hw5.logger import logger
from hw5.algos.base import Algo
from hw5.optimizers.first_order_optimizer import FirstOrderOptimizer
from hw5.utils import Serializable

import tensorflow as tf
from collections import OrderedDict


class PPO(Algo, Serializable):
    """
    Algorithm for PPO
    policy: neural net that can get actions based on observations
    learning_rate: learning rate for optimization
    clip_eps: 1 + clip_eps and 1 - clip_eps is the range you clip the objective
    max_epochs: max number of epochs
    entropy_bonus: weight on the entropy to encourage exploration
    use_entropy: enable entropy
    use_ppo_obj: enable the ppo objective instead of policy gradient
    use_clipper: enable clipping the objective
    """
    def __init__(
            self,
            policy,
            name="ppo",
            learning_rate=1e-3,
            clip_eps=0.2,
            max_epochs=5,
            entropy_bonus=0.,
            use_entropy=True,
            use_ppo_obj=True,
            use_clipper=True,
            **kwargs
            ):
        Serializable.quick_init(self, locals())
        super(PPO, self).__init__(policy)

        self.optimizer = FirstOrderOptimizer(learning_rate=learning_rate, max_epochs=max_epochs)
        self._optimization_keys = ['observations', 'actions', 'advantages', 'agent_infos']
        self.name = name
        self._clip_eps = clip_eps
        self.entropy_bonus = entropy_bonus
        self.use_entropy = use_entropy
        self.use_ppo_obj = use_ppo_obj
        self.use_clipper = use_clipper
        self.build_graph()

    def build_graph(self):
        """
        Creates the computation graph

        """

        """ Create Variables """

        """ ----- Build graph ----- """
        self.op_phs_dict = OrderedDict()
        obs_ph, action_ph, adv_ph, dist_info_old_ph, all_phs_dict = self._make_input_placeholders('train',)
        self.op_phs_dict.update(all_phs_dict)

        distribution_info_vars = self.policy.distribution_info_sym(obs_ph)
        hidden_ph, next_hidden_var = None, None

        """ policy gradient objective """

        if self.use_ppo_obj:
            """ YOUR CODE HERE FOR PROBLEM 1C --- PROVIDED """
            # hint: you need to implement pi over pi_old in this function. This function is located at hw5.policies.distributions.diagonal_gaussian
            # you don't need to write code here
            likelihood_ratio = self.policy.distribution.likelihood_ratio_sym(action_ph, dist_info_old_ph, distribution_info_vars)
            """ YOUR CODE ENDS """
        else:
            """YOUR CODE HERE FOR PROBLEM 1A --- PROVIDED"""
            # hint: you need to implement log pi in this function. This function is located at hw5.policies.distributions.diagnal_gaussian
            # you don't need to write anything here.
            loglikelihood = self.policy.distribution.log_likelihood_sym(action_ph, distribution_info_vars)
            likelihood_ratio = loglikelihood  # hint: here we abuse the variable name and make ppo and pg share the same var name.
            """YOUR CODE END"""

        if self.use_clipper:
            """ YOUR CODE HERE FOR PROBLEM 1D """
            # hint: as described, you need to first clip the likelihood_ratio between 1 + eps and 1 - eps
            # in the code, eps is self._clip_eps
            # finally you need to find the minimum of the non clipped objective and the clipped one, and we just call it clipped_obj in the code.
            obj_1 = tf.multiply(likelihood_ratio, adv_ph)
            obj_2 = np.clip(obj_1, 1-self._clip_eps, 1+self._clip_eps)
            clipped_obj = min(obj_1, obj_2)
            """ YOUR CODE END """
        else:
            """YOUR CODE HERE FOR PROBLEM 1A"""
            clipped_obj = tf.multiply(likelihood_ratio, adv_ph) # hint: here we also abuse the var name a bit. The obj is not clipped here!!!!
            """YOUR CODE ENDS"""

        if self.use_entropy:
            """YOUR CODE HERE FOR PROBLEM 1E --- PROVIDED"""
            # hint: entropy_bonus * entropy is the entropy obj
            # need to implement in hw5.policies.distributions.diagonal_gaussian
            # we are minimizing the objective, so it should all be negative
            # no code here
            entropy_obj = self.entropy_bonus * tf.reduce_mean(self.policy.distribution.entropy_sym(distribution_info_vars))
            surr_obj = - tf.reduce_mean(clipped_obj) - entropy_obj
            """ YOUR CODE END """
        else:
            # we are minimizing the objective, so it should all be negative
            surr_obj = - tf.reduce_mean(clipped_obj)

        self.optimizer.build_graph(
            loss=surr_obj,
            target=self.policy,
            input_ph_dict=self.op_phs_dict,
            hidden_ph=hidden_ph,
            next_hidden_var=next_hidden_var
        )

    def optimize_policy(self, samples_data, log=True, prefix='', verbose=False):
        """
        Performs policy optimization

        Args:
            samples_data (list) : list of lists of lists of samples (each is a dict) split by gradient update
            log (bool) : whether to log statistics

        Returns:
            None
        """
        input_dict = self._extract_input_dict(samples_data, self._optimization_keys, prefix='train')

        if verbose: logger.log("Optimizing")
        loss_before = self.optimizer.optimize(input_val_dict=input_dict)

        if verbose: logger.log("Computing statistics")
        loss_after = self.optimizer.loss(input_val_dict=input_dict)

        if log:
            logger.logkv(prefix+'LossBefore', loss_before)
            logger.logkv(prefix+'LossAfter', loss_after)

    def __getstate__(self):
        state = dict()
        state['init_args'] = Serializable.__getstate__(self)
        print('getstate\n')
        print(state['init_args'])
        state['policy'] = self.policy.__getstate__()
        state['optimizer'] = self.optimizer.__getstate__()
        return state

    def __setstate__(self, state):
        Serializable.__setstate__(self, state['init_args'])
        self.policy.__setstate__(state['policy'])
        self.optimizer.__getstate__(state['optimizer'])
