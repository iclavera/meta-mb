from meta_mb.dynamics.layers import MLP
import tensorflow as tf
from tensorflow.keras.utils import Progbar

import numpy as np
from meta_mb.utils.serializable import Serializable
from meta_mb.utils import compile_function
from meta_mb.logger import logger
from meta_mb.dynamics.mlp_dynamics import MLPDynamicsModel
import time
from collections import OrderedDict
from meta_mb.dynamics.utils import normalize, denormalize, train_test_split

class MLPDynamicsEnsemble(MLPDynamicsModel):
    """
    Class for MLP continous dynamics model
    """

    def __init__(self,
                 name,
                 env,
                 num_stack,
                 encoder,
                 input_is_img,
                 latent_dim,
                 buffer,
                 model_grad_thru_enc,
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
                 cpc_loss_weight=0.,
                 cpc_model=None,
                 ):

        Serializable.quick_init(self, locals())

        max_logvar = 1
        min_logvar = 0.1
        
        self.num_stack = num_stack
        self.encoder = encoder
        self.input_is_img = input_is_img
        self.cpc_model = cpc_model
        self.cpc_loss_weight = cpc_loss_weight
        self.latent_dim = latent_dim
        self.buffer = buffer
        self.model_grad_thru_enc = model_grad_thru_enc

        self.normalization = None
        self.normalize_input = normalize_input

        self.valid_split_ratio = valid_split_ratio
        self.rolling_average_persitency = rolling_average_persitency

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
        #     self.delta_ph = tf.placeholder(tf.float32, shape=(None, obs_space_dims))
        # 
        #     self._create_stats_vars()
        # 
        #     # concatenate action and observation --> NN input
        #     self.nn_input = tf.concat([self.obs_ph, self.act_ph], axis=1)
        # 
        #     obs_ph = tf.split(self.nn_input, self.num_models, axis=0)
        # 
        #     # create MLP
        #     mlps = []
        #     delta_preds = []
        #     self.obs_next_pred = []
        #     for i in range(num_models):
        #         with tf.variable_scope('model_{}'.format(i), reuse=tf.AUTO_REUSE):
        #             mlp = MLP(name+'/model_{}'.format(i),
        #                       output_dim=obs_space_dims,
        #                       hidden_sizes=hidden_sizes,
        #                       hidden_nonlinearity=hidden_nonlinearity,
        #                       output_nonlinearity=output_nonlinearity,
        #                       input_var=obs_ph[i],
        #                       input_dim=obs_space_dims+action_space_dims,
        #                       )
        #             mlps.append(mlp)
        # 
        #         delta_preds.append(mlp.output_var)
        # 
        #     self.delta_pred = tf.stack(delta_preds, axis=2)  # shape: (batch_size, ndim_obs, n_models)
        # 
        # 
        #     # define loss and train_op
        #     if loss_str == 'L2':
        #         self.loss = tf.reduce_mean(tf.linalg.norm(self.delta_ph[:, :, None] - self.delta_pred, axis=1))
        #     elif loss_str == 'MSE':
        #         self.loss = tf.reduce_mean((self.delta_ph[:, :, None] - self.delta_pred)**2)
        #     else:
        #         raise NotImplementedError
        # 
        #     self.optimizer = optimizer(learning_rate=self.learning_rate)
        #     self.train_op = self.optimizer.minimize(self.loss)
        # 
        #     # tensor_utils
        #     self.f_delta_pred = compile_function([self.obs_ph, self.act_ph], self.delta_pred)

        """ computation graph for inference where each of the models receives a different batch"""
        with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
            # placeholders
            if not input_is_img:
                self.obs_ph = obs = tf.placeholder(tf.float32, shape=(None, num_stack, latent_dim))
                self.act_ph = act = tf.placeholder(tf.float32, shape=(None, action_space_dims))
                self.delta_ph = delta = tf.placeholder(tf.float32, shape=(None, latent_dim))

            else:
                self.obs_ph = obs = tf.placeholder(tf.float32, shape=(None, num_stack, *obs_space_dims))
                self.act_ph = act = tf.placeholder(tf.float32, shape=(None, action_space_dims))
                self.delta_ph = delta = tf.placeholder(tf.float32, shape=(None, *obs_space_dims))
                if self.model_grad_thru_enc:
                    obs = self.encoder(tf.reshape(obs, shape=(-1, *obs_space_dims)))
                    obs = tf.reshape(obs, shape=(-1, num_stack, latent_dim))
                    delta = self.encoder(delta) - obs[:, -1]
                else:
                    obs = tf.stop_gradient(self.encoder(tf.reshape(obs, shape=(-1, *obs_space_dims))))
                    obs = tf.reshape(obs, shape=(-1, num_stack, latent_dim))
                    delta = tf.stop_gradient(self.encoder(delta) - obs[:, -1])

                if self.normalize_input:
                    obs = self.normalize(obs, self.buffer._mean_obs_var, self.buffer._std_obs_var)
                    delta = self.normalize(delta, self.buffer._mean_delta_var, self.buffer._std_delta_var)
                    act = self.normalize(act, self.buffer._mean_act_var, self.buffer._std_act_var)

            # split stack into the batches for each model --> assume each model receives a batch of the same size
            self.obs_model_batches = tf.split(obs, self.num_models, axis=0)
            self.act_model_batches = tf.split(act, self.num_models, axis=0)
            self.delta_model_batches = tf.split(delta, self.num_models, axis=0)

            # reuse previously created MLP but each model receives its own batch
            mlps = []
            delta_preds = []
            self.obs_next_pred = []
            self.model_loss = []
            # self.train_op_model_batches = []
            for i in range(num_models):
                with tf.variable_scope('model_{}'.format(i), reuse=tf.AUTO_REUSE):
                    # concatenate action and observation --> NN input
                    nn_input = tf.concat([tf.reshape(self.obs_model_batches[i], shape=(-1, num_stack * latent_dim)),
                                          self.act_model_batches[i]], axis=1)
                    mlp = MLP(name+'/model_{}'.format(i),
                              output_dim=latent_dim,
                              hidden_sizes=hidden_sizes,
                              hidden_nonlinearity=hidden_nonlinearity,
                              output_nonlinearity=output_nonlinearity,
                              input_var=nn_input,
                              input_dim=latent_dim * num_stack +action_space_dims,
                              weight_normalization=weight_normalization)
                    mlps.append(mlp)

                delta_preds.append(mlp.output_var)
                if loss_str == 'L2':
                    loss = tf.reduce_mean(tf.linalg.norm(self.delta_model_batches[i] - mlp.output_var, axis=1))
                elif loss_str == 'MSE':
                    loss = tf.reduce_mean((self.delta_model_batches[i] - mlp.output_var) ** 2)
                else:
                    raise  NotImplementedError
                self.model_loss.append(loss)

            self.total_loss = 1 / self.num_models * sum(self.model_loss)
            if cpc_loss_weight > 0:
                self.total_loss += cpc_loss_weight * self.cpc_model.loss
            self.optimizer = optimizer(learning_rate=self.learning_rate)
            self.train_op =  self.optimizer.minimize(self.total_loss)

            # self.delta_pred_model_batches_stack = tf.concat(delta_preds, axis=0) # shape: (batch_size_per_model*num_models, ndim_obs)

            # # tensor_utils
            # self.f_delta_pred_model_batches = compile_function([self.obs_ph,
            #                                                     self.act_ph],
            #                                                     self.delta_pred_model_batches_stack)

        self._networks = mlps
        # LayersPowered.__init__(self, [mlp.output_layer for mlp in mlps])
        


    def fit(self, obs, act, obs_next, cpc_train_buffer=None, cpc_val_buffer=None, epochs=1000, compute_normalization=True,
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


        epoch_times = []
        train_feed_dict_prep_time = 0
        train_op_time = 0
        train_batch_generation_time = 0

        val_feed_dict_prep_time = 0
        val_op_time = 0
        val_batch_generation_time = 0

        """ ------- Looping over training epochs ------- """
        for epoch in range(epochs):

            if self.normalize_input:
                self.buffer.update_embedding_buffer()

            dataset_size_in_trans = self.buffer.size
            val_size_in_trans = self.buffer.val_size

            # preparations for recording training stats
            epoch_start_time = time.time()
            total_losses = []
            cpc_losses = []
            cpc_rew_losses = []
            model_losses = []
            cpc_accs = []
            cpc_rew_accs = []
            cpc_grad_penalties = []

            """ ------- Looping through the shuffled and batched dataset for one epoch -------"""
            train_pb = Progbar(dataset_size_in_trans // self.batch_size)
            batch_generator = self.buffer.generate_batch()
            for _ in range(dataset_size_in_trans // self.batch_size):
                batch_time = time.time()
                obs_batch_stack, act_batch_stack, delta_batch_stack, _= next(batch_generator)
                train_batch_generation_time += time.time() - batch_time

                time_feed_dict_start = time.time()
                feed_dict = {self.obs_ph: obs_batch_stack,
                             self.act_ph: act_batch_stack,
                             self.delta_ph: delta_batch_stack}

                if self.cpc_loss_weight > 0:
                    inputs, labels = cpc_train_buffer.next()
                    feed_dict.update({self.cpc_model.x_input_ph: inputs[0],
                                      self.cpc_model.action_input_ph: inputs[1],
                                      self.cpc_model.y_input_ph: inputs[2],
                                      self.cpc_model.rew_input_ph: inputs[3],
                                      self.cpc_model.rew_labels_ph: labels,
                                      self.cpc_model.labels_ph: labels})

                train_feed_dict_prep_time += time.time() - time_feed_dict_start

                # run train op
                time_trainop_start = time.time()
                operations = [self.total_loss] + (self.model_loss + [self.cpc_model.loss, self.cpc_model.accuracy,
                                                                     self.cpc_model.rew_loss, self.cpc_model.rew_accuracy,
                                                                     self.cpc_model.gpenalty]
                                                if self.cpc_loss_weight else []) + [self.train_op]
                batch_loss_train_ops = sess.run(operations, feed_dict=feed_dict)
                train_op_time += time.time() - time_trainop_start

                total_loss = batch_loss_train_ops[0]
                prog_bar_list = [('total_loss', total_loss)]
                if self.cpc_loss_weight > 0:
                    model_loss = batch_loss_train_ops[1: self.num_models + 1]
                    model_losses.append(model_loss)
                    cpc_loss = batch_loss_train_ops[self.num_models + 1]
                    cpc_losses.append(cpc_loss)
                    cpc_acc = batch_loss_train_ops[self.num_models + 2]
                    cpc_accs.append(cpc_acc)
                    cpc_rew_loss = batch_loss_train_ops[self.num_models + 3]
                    cpc_rew_losses.append(cpc_rew_loss)
                    cpc_rew_acc = batch_loss_train_ops[self.num_models + 4]
                    cpc_rew_accs.append(cpc_rew_acc)
                    cpc_grad_penalty = batch_loss_train_ops[self.num_models + 5]
                    cpc_grad_penalties.append(cpc_grad_penalty)
                    prog_bar_list += [('model_loss', np.mean(model_loss)), ('cpc_loss', cpc_loss),
                                      ('cpc_weighted_loss', cpc_loss * self.cpc_loss_weight), ('cpc_acc', cpc_acc),
                                      ('cpc_rew_loss', cpc_rew_loss), ('cpc_rew_acc', cpc_rew_acc)]#, ('cpc_grad_penalty', cpc_grad_penalty)]
                total_losses.append(total_loss)
                train_pb.add(1,prog_bar_list)

            """ ------- Calculating validation loss ------- """
            total_losses_val = []
            cpc_losses_val = []
            cpc_accs_val = []
            cpc_rew_losses_val = []
            cpc_rew_accs_val = []
            cpc_grad_penalties_val = []
            model_losses_val = []
            val_pb = Progbar(val_size_in_trans // self.batch_size)
            batch_generator = self.buffer.generate_batch(test=True)
            for _ in range(val_size_in_trans // self.batch_size):
                batch_time = time.time()
                obs_batch_stack, act_batch_stack, delta_batch_stack, _ = next(batch_generator)
                val_batch_generation_time += time.time() - batch_time

                time_feed_dict_start = time.time()
                feed_dict = {self.obs_ph: obs_batch_stack,
                             self.act_ph: act_batch_stack,
                             self.delta_ph: delta_batch_stack}

                if self.cpc_loss_weight > 0:
                    inputs, labels = cpc_val_buffer.next()
                    feed_dict.update({self.cpc_model.x_input_ph: inputs[0],
                                      self.cpc_model.action_input_ph: inputs[1],
                                      self.cpc_model.y_input_ph: inputs[2],
                                      self.cpc_model.rew_input_ph: inputs[3],
                                      self.cpc_model.rew_labels_ph: labels,
                                      self.cpc_model.labels_ph: labels})
                val_feed_dict_prep_time += time.time() - time_feed_dict_start

                time_valop_start = time.time()
                operations = [self.total_loss] + (self.model_loss + [self.cpc_model.loss, self.cpc_model.accuracy,
                                                                     self.cpc_model.rew_loss, self.cpc_model.rew_accuracy,
                                                                     self.cpc_model.gpenalty]
                                                    if self.cpc_loss_weight else [])
                batch_loss_train_ops = sess.run(operations, feed_dict=feed_dict)
                val_op_time += time.time() - time_valop_start

                total_loss = batch_loss_train_ops[0]
                prog_bar_list = [('total_loss', total_loss)]
                if self.cpc_loss_weight > 0:
                    model_loss = batch_loss_train_ops[1:1 + self.num_models]
                    model_losses_val.append(model_loss)
                    cpc_loss = batch_loss_train_ops[1 + self.num_models]
                    cpc_losses_val.append(cpc_loss)
                    cpc_acc = batch_loss_train_ops[2 + self.num_models]
                    cpc_accs_val.append(cpc_acc)
                    cpc_rew_loss = batch_loss_train_ops[self.num_models + 3]
                    cpc_rew_losses_val.append(cpc_rew_loss)
                    cpc_rew_acc = batch_loss_train_ops[4 + self.num_models]
                    cpc_rew_accs_val.append(cpc_rew_acc)
                    cpc_grad_penalty = batch_loss_train_ops[self.num_models + 5]
                    cpc_grad_penalties_val.append(cpc_grad_penalty)
                    prog_bar_list += [('model_loss', np.mean(model_loss)), ('cpc_loss', cpc_loss),
                                      ('cpc_weighted_loss', cpc_loss * self.cpc_loss_weight), ('cpc_acc', cpc_acc),
                                      ('cpc_rew_loss', cpc_rew_loss), ('cpc_rew_acc', cpc_rew_acc)]#, ('cpc_grad_penalty', cpc_grad_penalty)]
                total_losses_val.append(total_loss)
                val_pb.add(1, prog_bar_list)


            epoch_times.append(time.time() - epoch_start_time)

        """ ------- Tabular Logging ------- """
        if log_tabular:
            logger.logkv(prefix+'AvgModelEpochTime', np.mean(epoch_times))
            logger.logkv(prefix+'TrainLoss', np.mean(total_losses))
            if self.cpc_loss_weight > 0:
                logger.logkv(prefix + 'TrainCPCLoss', np.mean(cpc_losses))
                logger.logkv(prefix + 'TrainCPCWeightedLoss', np.mean(cpc_losses) * self.cpc_loss_weight)
                logger.logkv(prefix + 'TrainCPCAcc', np.mean(cpc_accs))
                logger.logkv(prefix + 'TrainCPCRewLoss', np.mean(cpc_rew_losses))
                logger.logkv(prefix + 'TrainCPCRewAcc', np.mean(cpc_rew_accs))
                logger.logkv(prefix + 'TrainCPCGPenalty', np.mean(cpc_grad_penalties))
                logger.logkv(prefix + 'TrainModelLoss', np.mean(model_losses))
                
            logger.logkv(prefix+'AvgFinalValidLoss', np.mean(total_losses_val))
            # logger.logkv(prefix+'AvgFinalValidLossRoll', np.mean(valid_loss_rolling_average))
            if self.cpc_loss_weight > 0:
                logger.logkv(prefix + 'ValidCPCLoss', np.mean(cpc_losses_val))
                logger.logkv(prefix + 'ValidCPCWeightedLoss', np.mean(cpc_losses_val) * self.cpc_loss_weight)
                logger.logkv(prefix + 'ValidCPCAcc', np.mean(cpc_accs_val))
                logger.logkv(prefix + 'ValidCPCRewLoss', np.mean(cpc_rew_losses_val))
                logger.logkv(prefix + 'ValidCPCRewAcc', np.mean(cpc_rew_accs_val))
                logger.logkv(prefix + 'ValidCPCGPenalty', np.mean(cpc_grad_penalties_val))
                logger.logkv(prefix + 'ValidModelLoss', np.mean(model_losses_val))

            logger.logkv(prefix + 'TimeTrainBatchGeneration', train_batch_generation_time)
            logger.logkv(prefix + 'TimeValBatchGeneration', val_batch_generation_time)
            logger.logkv(prefix + 'TimeTrainOp', train_op_time)
            logger.logkv(prefix + 'TimeValOp', val_op_time)
            logger.logkv(prefix + 'TimeTrainFeeddict', train_feed_dict_prep_time)
            logger.logkv(prefix + 'TimeValFeeddict', val_feed_dict_prep_time)


    def predict_sym(self, obs_ph, act_ph):
        """
        Same batch fed into all models. Randomly output one of the predictions for each observation.
        :param obs_ph: (batch_size, obs_space_dims)
        :param act_ph: (batch_size, act_space_dims)
        :return: (batch_size, obs_space_dims)
        """
        original_obs = obs_ph
        if self.normalize_input:
            obs_ph = (obs_ph - self.buffer._mean_obs_var)/(self.buffer._std_obs_var + 1e-8)
            act_ph = (act_ph - self.buffer._mean_act_var)/(self.buffer._std_act_var + 1e-8)

        obs_ph = tf.reshape(obs_ph, shape=(-1, self.num_stack * self.latent_dim))

        # shuffle
        perm = tf.range(0, limit=tf.shape(obs_ph)[0], dtype=tf.int32)
        perm = tf.random.shuffle(perm)
        obs_ph, act_ph = tf.gather(obs_ph, perm), tf.gather(act_ph, perm)
        obs_ph, act_ph = tf.split(obs_ph, self.num_models, axis=0), tf.split(act_ph, self.num_models, axis=0)

        delta_preds = []
        with tf.variable_scope(self.name, reuse=tf.AUTO_REUSE):
            for i in range(self.num_models):
                with tf.variable_scope('model_{}'.format(i), reuse=True):

                    input_var = tf.concat([obs_ph[i], act_ph[i]], axis=1)
                    mlp = MLP(self.name+'/model_{}'.format(i),
                              output_dim=self.latent_dim,
                              hidden_sizes=self.hidden_sizes,
                              hidden_nonlinearity=self.hidden_nonlinearity,
                              output_nonlinearity=self.output_nonlinearity,
                              input_var=input_var,
                              input_dim=self.latent_dim * self.num_stack + self.action_space_dims,
                              )
                    # denormalize delta_pred
                    if self.normalize_input:
                        delta_pred = mlp.output_var * self.buffer._std_delta_var + self.buffer._mean_delta_var
                    else:
                        delta_pred = mlp.output_var
                    delta_preds.append(delta_pred)

        delta_preds = tf.concat(delta_preds, axis=0)

        # unshuffle
        perm_inv = tf.invert_permutation(perm)
        next_obs = original_obs[:, -1, :] + tf.gather(delta_preds, perm_inv)
        next_obs = tf.clip_by_value(next_obs, -1e2, 1e2)
        return tf.concat([original_obs[:, 1:], next_obs[:, None, :]], axis=1)

    

    def predict(self, obs, act, pred_type='rand', **kwargs):
        """
        Predict the batch of next observations given the batch of current observations and actions
        :param obs: observations - numpy array of shape (n_samples, ndim_obs)
        :param act: actions - numpy array of shape (n_samples, ndim_act)
        :param pred_type:  prediction type
                   - rand: choose one of the models randomly
                   - mean: mean prediction of all models
                   - all: returns the prediction of all the models
        :return: pred_obs_next: predicted batch of next observations -
                                shape:  (n_samples, ndim_obs) - in case of 'rand' and 'mean' mode
                                        (n_samples, ndim_obs, n_models) - in case of 'all' mode
        """
        assert obs.shape[0] == act.shape[0]
        assert obs.ndim == 2 and obs.shape[1] == self.obs_space_dims
        assert act.ndim == 2 and act.shape[1] == self.action_space_dims

        obs_original = obs

        if self.normalize_input:
            obs, act = [obs for _ in range(self.num_models)], [act for _ in range(self.num_models)]
            obs, act = self._normalize_data(obs, act)
            obs, act = np.concatenate(obs, axis=0), np.concatenate(act, axis=0)
            delta = np.array(self.f_delta_pred(obs, act))
            delta = self._denormalize_data(delta)

        else:
            obs, act = [obs for _ in range(self.num_models)], [act for _ in range(self.num_models)]
            obs, act = np.concatenate(obs, axis=0), np.concatenate(act, axis=0)
            delta = np.array(self.f_delta_pred(obs, act))

        assert delta.ndim == 3

        pred_obs = obs_original[:, :, None] + delta

        batch_size = delta.shape[0]
        if pred_type == 'rand':
            # randomly selecting the prediction of one model in each row
            idx = np.random.randint(0, self.num_models, size=batch_size)
            pred_obs = np.stack([pred_obs[row, :, model_id] for row, model_id in enumerate(idx)], axis=0)
        elif pred_type == 'mean':
            pred_obs = np.mean(pred_obs, axis=2)
        elif pred_type == 'all':
            pass
        else:
            NotImplementedError('pred_type must be one of [rand, mean, all]')

        pred_obs = np.clip(pred_obs, -1e2, 1e2)
        return pred_obs


    def predict_std(self, obs, act):
        """
        Calculates the std of predicted next observations among the models
        given the batch of current observations and actions
        1:param obs: observations - numpy array of shape (n_samples, ndim_obs)
        :param act: actions - numpy array of shape (n_samples, ndim_act)
        :return: std_pred_obs: std of predicted next observatations - (n_samples, ndim_obs)
        """
        assert self.num_models > 1, "calculating the std requires at "
        pred_obs = self.predict(obs, act, pred_type='all')
        assert pred_obs.ndim == 3
        return np.std(pred_obs, axis=2)

    def reinit_model(self):
        sess = tf.get_default_session()
        if '_reinit_model_op' not in dir(self):
            self._reinit_model_op = [tf.variables_initializer(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,
                                    scope=self.name+'/model_{}'.format(i))) for i in range(self.num_models)]
        sess.run(self._reinit_model_op)




    def _denormalize_data(self, delta):
        assert delta.shape[-1] == self.num_models
        denorm_deltas = []
        for i in range(self.num_models):
            denorm_delta = denormalize(delta[..., i], self.normalization['delta'][0], self.normalization['delta'][1])
            denorm_deltas.append(denorm_delta)
        return np.stack(denorm_deltas, axis=-1)

    def get_shared_param_values(self): # to feed policy
        state = dict()
        state['normalization'] = self.normalization
        state['networks_params'] = [nn.get_param_values() for nn in self._networks]
        return state

    def set_shared_params(self, state):
        self.normalization = state['normalization']
        feed_dict = {}
        for i in range(self.num_models):
            feed_dict.update({
                self._mean_obs_ph: self.normalization['obs'][0],
                self._std_obs_ph: self.normalization['obs'][1],
                self._mean_act_ph: self.normalization['act'][0],
                self._std_act_ph: self.normalization['act'][1],
                self._mean_delta_ph: self.normalization['delta'][0],
                self._std_delta_ph: self.normalization['delta'][1],
            })
        sess = tf.get_default_session()
        sess.run(self._assignations, feed_dict=feed_dict)
        for i in range(len(self._networks)):
            self._networks[i].set_params(state['networks_params'][i])

    def normalize(self, data, mean, std):
        return (data - mean) / (std + 1e-8)

    def denormalize(self, data, mean, std):
        return data * std + mean






