from meta_mb.dynamics.layers import MLP
import tensorflow as tf
import numpy as np
from meta_mb.utils.serializable import Serializable
from meta_mb.utils import compile_function
from meta_mb.logger import logger
from meta_mb.reward_model.mlp_reward import MLPRewardModel
import time
from collections import OrderedDict
from meta_mb.dynamics.utils import normalize, denormalize, train_test_split


class MLPRewardEnsemble(MLPRewardModel):
    """
    Class for MLP continous dynamics model
    """

    def __init__(self,
                 name,
                 env,
                 latent_dim,
                 encoder,
                 input_is_img, 
                 buffer,
                 num_models=5,
                 hidden_sizes=(512, 512),
                 hidden_nonlinearity='swish',
                 output_nonlinearity=None,
                 batch_size=500,
                 learning_rate=0.001,
                 weight_normalization=False,  # Doesn't work
                 normalize_input=True,
                 optimizer=tf.train.AdamOptimizer,
                 valid_split_ratio=0.2,  # 0.1
                 rolling_average_persitency=0.99,
                 buffer_size=50000,
                 loss_str='MSE',
                 ):

        Serializable.quick_init(self, locals())

        max_logvar = 1
        min_logvar = 0.1
        
        self.encoder = encoder
        self.input_is_img = input_is_img
        self.latent_dim = latent_dim
        self.buffer = buffer

        self.normalization = None
        self.normalize_input = normalize_input


        self.valid_split_ratio = valid_split_ratio
        self.rolling_average_persitency = rolling_average_persitency

        self.buffer_size_train = int(buffer_size * (1 - valid_split_ratio))
        self.buffer_size_test = int(buffer_size * valid_split_ratio)
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.num_models = num_models
        self.hidden_sizes = hidden_sizes
        self.name = name


        # determine dimensionality of state and action space
        self.obs_space_dims = obs_space_dims = env.observation_space.shape
        self.action_space_dims = action_space_dims = env.action_space.shape[0]
        self.timesteps_counter = 0
        self.used_timesteps_counter = 0

        self.hidden_nonlinearity = hidden_nonlinearity = self._activations[hidden_nonlinearity]
        self.output_nonlinearity = output_nonlinearity = self._activations[output_nonlinearity]

        # """ computation graph for training and simple inference """
        # with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
        #     # placeholders
        #     self.obs_ph = tf.placeholder(tf.float32, shape=(None, obs_space_dims))
        #     self.act_ph = tf.placeholder(tf.float32, shape=(None, action_space_dims))
        #     self.nextobs_ph = tf.placeholder(tf.float32, shape=(None, obs_space_dims))
        #     self.reward_ph = tf.placeholder(tf.float32, shape=(None, 1))
        #
        #     self._create_stats_vars()
        #
        #     # concatenate action and observation --> NN input
        #     self.nn_input = tf.concat([self.obs_ph, self.act_ph, self.nextobs_ph], axis=1)
        #
        #     obs_ph = tf.split(self.nn_input, self.num_models, axis=0)
        #
        #     # create MLP
        #     mlps = []
        #     reward_preds = []
        #     self.obs_next_pred = []
        #     for i in range(num_models):
        #         with tf.variable_scope('model_{}'.format(i), reuse=tf.AUTO_REUSE):
        #             mlp = MLP(name + '/model_{}'.format(i),
        #                       output_dim=1,
        #                       hidden_sizes=hidden_sizes,
        #                       hidden_nonlinearity=hidden_nonlinearity,
        #                       output_nonlinearity=output_nonlinearity,
        #                       input_var=obs_ph[i],
        #                       input_dim=2 * obs_space_dims + action_space_dims,
        #                       )
        #             mlps.append(mlp)
        #
        #         reward_preds.append(mlp.output_var)
        #
        #     self.reward_pred = tf.stack(reward_preds, axis=2)  # shape: (batch_size, ndim_obs, n_models)
        #
        #     # define loss and train_op
        #     if loss_str == 'L2':
        #         self.loss = tf.reduce_mean(tf.linalg.norm(self.reward_ph[:, :, None] - self.reward_pred, axis=1))
        #     elif loss_str == 'MSE':
        #         self.loss = tf.reduce_mean((self.reward_ph[:, :, None] - self.reward_pred) ** 2)
        #     else:
        #         raise NotImplementedError
        #
        #     self.optimizer = optimizer(learning_rate=self.learning_rate)
        #     self.train_op = self.optimizer.minimize(self.loss)
        #
        #     # tensor_utils
        #     self.f_reward_pred = compile_function([self.obs_ph, self.act_ph, self.nextobs_ph], self.reward_pred)

        # """ computation graph for inference where each of the models receives a different batch"""
        """ computation graph for inference where each of the models receives a different batch"""
        with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
            # placeholders
            if not self.input_is_img:
                self.obs_ph = obs = tf.placeholder(tf.float32, shape=(None, latent_dim))
                self.act_ph = act = tf.placeholder(tf.float32, shape=(None, action_space_dims))
                self.nextobs_ph = nextobs = tf.placeholder(tf.float32, shape=(None, latent_dim))
                self.reward_ph = reward = tf.placeholder(tf.float32, shape=(None, 1))

            else:
                self.obs_ph = obs = tf.placeholder(tf.float32, shape=(None, *obs_space_dims))
                self.act_ph = act = tf.placeholder(tf.float32, shape=(None, action_space_dims))
                self.nextobs_ph = nextobs = tf.placeholder(tf.float32, shape=(None, *obs_space_dims))
                self.reward_ph = reward = tf.placeholder(tf.float32, shape=(None, 1))
                obs = tf.stop_gradient(self.encoder(obs))  # TODO: CHECK THAT THIS STOPS THE GRADIENT!
                nextobs = tf.stop_gradient(self.encoder(nextobs))

            # split stack into the batches for each model --> assume each model receives a batch of the same size
            self.obs_model_batches = tf.split(obs, self.num_models, axis=0)
            self.act_model_batches = tf.split(act, self.num_models, axis=0)
            self.nextobs_model_batches = tf.split(nextobs, self.num_models, axis=0)
            self.reward_model_batches = tf.split(reward, self.num_models, axis=0)

            # reuse previously created MLP but each model receives its own batch
            mlps = []
            reward_preds = []
            self.loss_model_batches = []
            self.train_op_model_batches = []
            for i in range(num_models):
                with tf.variable_scope('model_{}'.format(i), reuse=tf.AUTO_REUSE):
                    # concatenate action and observation --> NN input
                    nn_input = tf.concat([self.obs_model_batches[i], self.act_model_batches[i], self.nextobs_model_batches[i]], axis=1)
                    mlp = MLP(name + '/model_{}'.format(i),
                              output_dim=1,
                              hidden_sizes=hidden_sizes,
                              hidden_nonlinearity=hidden_nonlinearity,
                              output_nonlinearity=output_nonlinearity,
                              input_var=nn_input,
                              input_dim=2 * latent_dim + action_space_dims,
                              weight_normalization=weight_normalization)
                    mlps.append(mlp)

                reward_preds.append(mlp.output_var)
                if loss_str == 'L2':
                    loss = tf.reduce_mean(tf.linalg.norm(self.reward_model_batches[i] - mlp.output_var, axis=1))
                elif loss_str == 'MSE':
                    loss = tf.reduce_mean((self.reward_model_batches[i] - mlp.output_var) ** 2)
                else:
                    raise NotImplementedError
                self.loss_model_batches.append(loss)
                self.train_op_model_batches.append(optimizer(learning_rate=self.learning_rate).minimize(loss))
            self.reward_pred_model_batches_stack = tf.concat(reward_preds,
                                                            axis=0)  # shape: (batch_size_per_model*num_models, ndim_obs)

            # # tensor_utils
            # self.f_reward_pred_model_batches = compile_function([self.obs_ph,
            #                                                     self.act_ph,
            #                                                     self.nextobs_ph],
            #                                                    self.reward_pred_model_batches_stack)

        self._networks = mlps
        # LayersPowered.__init__(self, [mlp.output_layer for mlp in mlps])

    def fit(self, obs, act, obs_next, reward, epochs=1000, compute_normalization=True,
            valid_split_ratio=None, rolling_average_persitency=None, verbose=False, log_tabular=False, prefix=''):
        """
        Fits the NN dynamics model
        :param obs: observations - numpy array of shape (n_samples, ndim_obs)
        :param act: actions - numpy array of shape (n_samples, ndim_act)
        :param obs_next: observations after taking action - numpy array of shape (n_samples, ndim_obs)
        :param epochs: number of training epochs
        :param compute_normalization: boolean indicating whether normalization shall be (re-)computed given the data
        :param valid_split_ratio: relative size of validation split (float between 0.0 and 1.0)
        :param (boolean) whether to log training stats in tabular format
        :param verbose: logging verbosity
        """

        if rolling_average_persitency is None: rolling_average_persitency = self.rolling_average_persitency

        sess = tf.get_default_session()

        valid_loss_rolling_average = None
        train_op_to_do = self.train_op_model_batches
        idx_to_remove = []
        epoch_times = []
        epochs_per_model = []

        """ ------- Looping over training epochs ------- """
        for epoch in range(epochs):

            dataset_size_in_trans = self.buffer.size

            # preparations for recording training stats
            epoch_start_time = time.time()
            batch_losses = []

            """ ------- Looping through the shuffled and batched dataset for one epoch -------"""
            for _ in range(dataset_size_in_trans // self.batch_size):
                obs_batch_stack, act_batch_stack, nextobs_batch_stack, reward_batch_stack = self.buffer.generate_reward_batch()

                # run train op
                batch_loss_train_ops = sess.run(self.loss_model_batches + train_op_to_do,
                                                feed_dict={self.obs_ph: obs_batch_stack,
                                                           self.act_ph: act_batch_stack,
                                                           self.nextobs_ph: nextobs_batch_stack,
                                                           self.reward_ph: reward_batch_stack})

                batch_loss = np.array(batch_loss_train_ops[:self.num_models])
                batch_losses.append(batch_loss)

            """ ------- Calculating validation loss ------- """
            obs_test_stack, act_test_stack, nextobs_test_stack, reward_test_stack = self.buffer.generate_reward_batch(test=True)

            # compute validation loss
            valid_loss = sess.run(self.loss_model_batches,
                                  feed_dict={self.obs_ph: obs_test_stack,
                                             self.act_ph: act_test_stack,
                                             self.nextobs_ph: nextobs_test_stack,
                                             self.reward_ph: reward_test_stack})


            valid_loss = np.array(valid_loss)
            if valid_loss_rolling_average is None:
                valid_loss_rolling_average = 1.5 * valid_loss  # set initial rolling to a higher value avoid too early stopping
                valid_loss_rolling_average_prev = 2.0 * valid_loss
                for i in range(len(valid_loss)):
                    if valid_loss[i] < 0:
                        valid_loss_rolling_average[i] = valid_loss[
                                                            i] / 1.5  # set initial rolling to a higher value avoid too early stopping
                        valid_loss_rolling_average_prev[i] = valid_loss[i] / 2.0

            valid_loss_rolling_average = rolling_average_persitency * valid_loss_rolling_average \
                                         + (1.0 - rolling_average_persitency) * valid_loss

            if verbose:
                str_mean_batch_losses = ' '.join(['%.4f' % x for x in np.mean(batch_losses, axis=0)])
                str_valid_loss = ' '.join(['%.4f' % x for x in valid_loss])
                str_valid_loss_rolling_averge = ' '.join(['%.4f' % x for x in valid_loss_rolling_average])
                logger.log(
                    "Training NNDynamicsModel - finished epoch %i --\n"
                    "train loss: %s\nvalid loss: %s\nvalid_loss_mov_avg: %s"
                    % (epoch, str_mean_batch_losses, str_valid_loss, str_valid_loss_rolling_averge)
                )


            for i in range(self.num_models):
                if (valid_loss_rolling_average_prev[i] < valid_loss_rolling_average[
                    i] or epoch == epochs - 1) and i not in idx_to_remove:
                    idx_to_remove.append(i)
                    epochs_per_model.append(epoch)
                    if epoch < epochs - 1:
                        logger.log(
                            'At Epoch {}, stop model {} since its valid_loss_rolling_average decreased'.format(epoch, i))

            train_op_to_do = [op for idx, op in enumerate(self.train_op_model_batches) if idx not in idx_to_remove]

            if not idx_to_remove:
                epoch_times.append(time.time() - epoch_start_time)  # only track epoch times while all models are trained

            if not train_op_to_do:
                if verbose and epoch < epochs - 1:
                    logger.log('Stopping all DynamicsEnsemble Training before reaching max_num_epochs')
                break
            valid_loss_rolling_average_prev = valid_loss_rolling_average

        """ ------- Tabular Logging ------- """
        if log_tabular:
            logger.logkv(prefix + 'AvgModelEpochTime', np.mean(epoch_times))
            assert len(epochs_per_model) == self.num_models
            logger.logkv(prefix + 'AvgEpochs', np.mean(epochs_per_model))
            logger.logkv(prefix + 'StdEpochs', np.std(epochs_per_model))
            logger.logkv(prefix + 'MaxEpochs', np.max(epochs_per_model))
            logger.logkv(prefix + 'MinEpochs', np.min(epochs_per_model))
            logger.logkv(prefix + 'AvgFinalTrainLoss', np.mean(batch_losses))
            logger.logkv(prefix + 'AvgFinalValidLoss', np.mean(valid_loss))
            logger.logkv(prefix + 'AvgFinalValidLossRoll', np.mean(valid_loss_rolling_average))


    def predict_sym(self, obs_ph, act_ph, nextobs_ph):
        """
        Same batch fed into all models. Randomly output one of the predictions for each observation.
        :param obs_ph: (batch_size, obs_space_dims)
        :param act_ph: (batch_size, act_space_dims)
        :return: (batch_size, obs_space_dims)
        """
        assert self.normalize_input
        obs_ph = (obs_ph - self.buffer._mean_obs_var)/(self.buffer._std_obs_var + 1e-8)
        act_ph = (act_ph - self.buffer._mean_act_var)/(self.buffer._std_act_var + 1e-8)
        nextobs_ph = (nextobs_ph - self.buffer._mean_obs_var)/(self.buffer._std_obs_var + 1e-8)

        # shuffle
        perm = tf.range(0, limit=tf.shape(obs_ph)[0], dtype=tf.int32)
        perm = tf.random.shuffle(perm)
        obs_ph, act_ph, nextobs_ph = tf.gather(obs_ph, perm), tf.gather(act_ph, perm), tf.gather(nextobs_ph, perm)
        obs_ph, act_ph, nextobs_ph = tf.split(obs_ph, self.num_models, axis=0), tf.split(act_ph, self.num_models, axis=0), \
                                     tf.split(nextobs_ph, self.num_models, axis=0)

        reward_preds = []
        with tf.variable_scope(self.name, reuse=tf.AUTO_REUSE):
            for i in range(self.num_models):
                with tf.variable_scope('model_{}'.format(i), reuse=True):
                    input_var = tf.concat([obs_ph[i], act_ph[i], nextobs_ph[i]], axis=1)
                    mlp = MLP(self.name + '/model_{}'.format(i),
                              output_dim=1,
                              hidden_sizes=self.hidden_sizes,
                              hidden_nonlinearity=self.hidden_nonlinearity,
                              output_nonlinearity=self.output_nonlinearity,
                              input_var=input_var,
                              input_dim=2 * self.latent_dim + self.action_space_dims,
                              )
                    # denormalize reward_pred
                    reward_pred = mlp.output_var * self.buffer._std_reward_var + self.buffer._mean_reward_var
                    reward_preds.append(reward_pred)

        reward_preds = tf.concat(reward_preds, axis=0)

        # unshuffle
        perm_inv = tf.invert_permutation(perm)
        reward = tf.gather(reward_preds, perm_inv)
        reward = tf.clip_by_value(reward, -1e2, 1e2)
        return tf.reshape(reward, (-1,))





    def predict_std(self, obs, act, nextobs):
        """
        Calculates the std of predicted next observations among the models
        given the batch of current observations and actions
        1:param obs: observations - numpy array of shape (n_samples, ndim_obs)
        :param act: actions - numpy array of shape (n_samples, ndim_act)
        :return: std_pred_obs: std of predicted next observatations - (n_samples, ndim_obs)
        """
        assert self.num_models > 1, "calculating the std requires at "
        pred_reward = self.predict(obs, act, nextobs, pred_type='all')
        assert pred_reward.ndim == 3
        return np.std(pred_reward, axis=2)


    def reinit_model(self):
        sess = tf.get_default_session()
        if '_reinit_model_op' not in dir(self):
            self._reinit_model_op = [tf.variables_initializer(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,
                                                                                scope=self.name + '/model_{}'.format(i)))
                                     for i in range(self.num_models)]
        sess.run(self._reinit_model_op)




    def get_shared_param_values(self):  # to feed policy
        state = dict()
        state['normalization'] = self.normalization
        state['networks_params'] = [nn.get_param_values() for nn in self._networks]
        return state


    def set_shared_params(self, state):
        self.normalization = state['normalization']
        feed_dict = {}
        for i in range(self.num_models):
            feed_dict.update({
                self._mean_obs_ph[i]: self.normalization[i]['obs'][0],
                self._std_obs_ph[i]: self.normalization[i]['obs'][1],
                self._mean_act_ph[i]: self.normalization[i]['act'][0],
                self._std_act_ph[i]: self.normalization[i]['act'][1],
                self._mean_nextobs_ph[i]: self.normalization[i]['nextobs'][0],
                self._std_nextobs_ph[i]: self.normalization[i]['nextobs'][1],
                self._mean_reward_ph[i]: self.normalization[i]['reward'][0],
                self._std_reward_ph[i]: self.normalization[i]['reward'][1],
            })
        sess = tf.get_default_session()
        sess.run(self._assignations, feed_dict=feed_dict)
        for i in range(len(self._networks)):
            self._networks[i].set_params(state['networks_params'][i])


