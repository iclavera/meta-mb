import keras
import keras.backend as K
import os
import tensorflow as tf
import matplotlib
# Force matplotlib to not use any Xwindows backend.
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
import time

from meta_mb.logger import logger
from meta_mb.unsupervised_learning.cpc.data_utils import CPCDataGenerator, plot_seq
from meta_mb.unsupervised_learning.cpc.training_utils import SaveEncoder, cross_entropy_loss

def visualize_img(images, m, n, name='example.png'):
    plt.figure(figsize=(n, m))
    sample = images
    image_size = images.shape[-2]
    sample = np.transpose(
        np.reshape(sample, [n, image_size * m, image_size, 3]),
        [0, 2, 1, 3])
    sample = np.transpose(
        np.reshape(sample, [1, image_size * n, image_size * m, 3]),
        [0, 2, 1, 3])
    # sample = np.cast(sample, np.uint8)
    # sample = sample[:, :, :, ::-1]
    plt.imshow(sample[0])
    plt.savefig(name)


class Trainer(object):
    """
    Performs steps for MAML

    Args:
        algo (Algo) :
        env (Env) :
        sampler (Sampler) :
        sample_processor (SampleProcessor) :
        baseline (Baseline) :
        policy (Policy) :
        n_itr (int) : Number of iterations to train for
        start_itr (int) : Number of iterations policy has already trained for, if reloading
        num_inner_grad_steps (int) : Number of inner steps per maml iteration
        sess (tf.Session) : current tf session (if we loaded policy, for example)
    """
    def __init__(
            self,
            env,
            sampler,
            buffer,
            dynamics_sample_processor,
            policy,
            dynamics_model,
            n_itr,
            cpc_model,
            vae=None,
            start_itr=0,
            initial_random_samples=True,
            sess=None,
            dynamics_model_max_epochs=200,
            reward_model_max_epochs=200,
            reward_model=None,
            reward_sample_processor=None,

            cpc_terms=1,
            cpc_predict_terms=1,
            cpc_initial_epoch=30,
            cpc_initial_lr = 1e-3,
            cpc_epoch=5,
            cpc_lr=1e-4,
            cpc_batch_size=64,
            cpc_negative_samples=10,
            cpc_negative_same_traj=0,
            cpc_initial_sampler=None,
            cpc_train_interval=10,

            path_checkpoint_interval=1,
            train_emb_to_state=False,
            random_buffer=None,
            ):
        self.env = env
        self.sampler = sampler
        self.buffer = buffer
        self.random_buffer = random_buffer
        self.dynamics_sample_processor = dynamics_sample_processor
        self.dynamics_model = dynamics_model
        self.reward_sample_processor = reward_sample_processor
        self.reward_model = reward_model
        self.policy = policy
        self.n_itr = n_itr
        self.start_itr = start_itr
        self.dynamics_model_max_epochs = dynamics_model_max_epochs
        self.reward_model_max_epochs = reward_model_max_epochs

        self.cpc_model = cpc_model
        self.cpc_terms = cpc_terms
        self.cpc_predict_terms = cpc_predict_terms
        self.cpc_initial_epoch = cpc_initial_epoch
        self.cpc_initial_lr = cpc_initial_lr
        self.cpc_epoch = cpc_epoch
        self.cpc_lr = cpc_lr
        self.cpc_batch_size = cpc_batch_size
        self.cpc_negative_samples = cpc_negative_samples
        self.cpc_negative_same_traj = cpc_negative_same_traj
        self.cpc_initial_sampler = cpc_initial_sampler
        self.cpc_train_interval = cpc_train_interval

        self.vae = vae

        self.path_checkpoint_interval = path_checkpoint_interval
        self.train_emb_to_state = train_emb_to_state

        self.env.reset()
        self.state_dim = len(self.env.true_state)
        self.latent_dim = self.cpc_model.latent_dim

        self.initial_random_samples = initial_random_samples

        if sess is None:
            sess = tf.Session()
        self.sess = sess

    def train(self):
        """
        Trains policy on env using algo

        Pseudocode:
            for itr in n_itr:
                for step in num_inner_grad_steps:
                    sampler.sample()
                    algo.compute_updated_dists()
                algo.optimize_policy()
                sampler.update_goals()
        """

        with self.sess.as_default() as sess:

            # initialize uninitialized vars  (only initialize vars that were not loaded)
            name_to_var = {v.op.name: v for v in tf.global_variables() + tf.local_variables()}
            uninit_vars_b = sess.run(tf.report_uninitialized_variables())
            uninit_vars = list(name_to_var[name.decode()] for name in uninit_vars_b)
            sess.run(tf.variables_initializer(uninit_vars))

            start_time = time.time()

            ''' --------------- Pretrain CPC on exploratory data --------------- '''
            length = self.cpc_initial_sampler.max_path_length if self.env.encoder is None else \
                self.cpc_initial_sampler.max_path_length  - 1
            empty_imgs = np.zeros(shape=(0, length, *self.cpc_model.image_shape))
            empty_acs = np.zeros(shape=(0, length, *self.env.action_space.shape))
            empty_rews = np.zeros(shape=(0, length, 1))

            train_data = CPCDataGenerator(empty_imgs.copy(), empty_acs.copy(), empty_rews.copy(),
                                          self.cpc_batch_size, terms=self.cpc_terms,
                                          negative_samples=self.cpc_negative_samples,
                                          predict_terms=self.cpc_predict_terms,
                                          negative_same_traj=self.cpc_negative_same_traj,)
            validation_data = CPCDataGenerator(empty_imgs.copy(), empty_acs.copy(), empty_rews.copy(),
                                               self.cpc_batch_size, terms=self.cpc_terms,
                                               negative_samples=self.cpc_negative_samples,
                                               predict_terms=self.cpc_predict_terms,
                                               negative_same_traj=self.cpc_negative_same_traj,)

            if self.random_buffer is not None:
                env_paths = self.cpc_initial_sampler.obtain_samples(log=True, random=True, log_prefix='')
                samples_data = self.dynamics_sample_processor.process_samples(env_paths,
                                                                          log=True, log_prefix='EnvTrajs-')
                print("We have collected %d trajs to put in the random buffer" % len(env_paths))

                self.random_buffer.update_buffer(samples_data['observations'],
                                          samples_data['actions'],
                                          samples_data['next_observations'],
                                          samples_data['rewards'],
                                          samples_data['env_infos']['true_state'])
                self.random_buffer.update_embedding_buffer()

            if self.cpc_initial_epoch > 0:
                env_paths = self.cpc_initial_sampler.obtain_samples(log=True, random=True, log_prefix='')
                samples_data = self.dynamics_sample_processor.process_samples(env_paths,
                                                                              log=True, log_prefix='EnvTrajs-')


                if self.vae is None:
                    train_img, val_img, train_action, val_action, train_rew, val_rew = self.get_seqs(env_paths)
                    train_data.update_dataset(train_img, train_action, train_rew)
                    validation_data.update_dataset(val_img, val_action, val_rew)

                    # Train the model
                    self.cpc_model.fit_generator(
                        generator=train_data,
                        steps_per_epoch=len(train_data),
                        validation_data=validation_data,
                        validation_steps=len(validation_data),
                        epochs=self.cpc_initial_epoch,
                        patience=2
                    )

                else:
                    # train VAE instead of CPC
                    train_img = np.concatenate([path['observations'] for path in env_paths])
                    self.vae.train(train_img, epoch=self.cpc_initial_epoch)

            if self.train_emb_to_state:
                emb_input = keras.layers.Input(shape=(self.latent_dim, ))
                hidden = keras.layers.Dense(256, activation='relu')(emb_input)
                hidden = keras.layers.Dense(256, activation='relu')(hidden)
                state_output = keras.layers.Dense(self.state_dim, activation='linear')(hidden)
                emb2state_model = keras.Model(inputs=emb_input, outputs=state_output)


                emb2state_model.save_weights(os.path.join(logger.get_dir(), 'emb2state_model.h5'))


            for itr in range(self.start_itr, self.n_itr):
                # for point mass

                itr_start_time = time.time()
                logger.log("\n ---------------- Iteration %d ----------------" % itr)

                time_env_sampling_start = time.time()

                if self.initial_random_samples and itr == 0:
                    logger.log("Obtaining random samples from the environment...")
                    env_paths = self.sampler.obtain_samples(log=True, random=True, log_prefix='')
                else:
                    logger.log("Obtaining samples from the environment using the policy...")
                    env_paths = self.sampler.obtain_samples(log=True, log_prefix='')

                if itr % self.path_checkpoint_interval == 0:
                    imgs = np.concatenate(self.get_seqs(env_paths)[:1], axis=0)
                    imgs = imgs[:, :min(250, imgs.shape[1])]
                    n, m = imgs.shape[:2]
                    imgs = imgs.reshape(-1, *imgs.shape[2:])
                    visualize_img(imgs, m, n,
                                  name=os.path.join(logger.get_dir(), 'path_itr%d.png' % itr))


                train_img, val_img, train_action, val_action, train_rew, val_rew = \
                    self.get_seqs(env_paths, test_size=0.25 if itr == 0 else 0.1)
                train_data.update_dataset(train_img, train_action, train_rew)
                validation_data.update_dataset(val_img, val_action, val_rew)

                logger.record_tabular('Time-EnvSampling', time.time() - time_env_sampling_start)
                logger.log("Processing environment samples...")

                # first processing just for logging purposes
                time_env_samp_proc = time.time()
                samples_data = self.dynamics_sample_processor.process_samples(env_paths,
                                                                              log=True, log_prefix='EnvTrajs-')
                if self.buffer is not None:
                    self.buffer.update_buffer(samples_data['observations'],
                                            samples_data['actions'],
                                            samples_data['next_observations'],
                                            samples_data['rewards'],
                                            samples_data['env_infos']['true_state'])

                logger.record_tabular('Time-EnvSampleProc', time.time() - time_env_samp_proc)

                ''' --------------- finetune cpc --------------- '''
                time_cpc_start = time.time()
                if self.cpc_epoch > 0 and itr % self.cpc_train_interval == 0 and itr > 0:
                    # Train the model
                    self.cpc_model.fit_generator(
                        generator=train_data,
                        steps_per_epoch=len(train_data),
                        validation_data=validation_data,
                        validation_steps=len(validation_data),
                        epochs=self.cpc_epoch,
                        patience=2
                    )

                    # K.set_learning_phase(0)
                    # self.env._wrapped_env._vae = CPCEncoder(path=os.path.join(logger.get_dir(), 'encoder.h5'))

                logger.record_tabular('Time-CPCModelFinetune', time.time() - time_cpc_start)
                if not self.buffer is None: #and not self.dynamics_model.input_is_img:
                    self.buffer.update_embedding_buffer()

                ''' --------------- fit dynamics model --------------- '''

                time_fit_start = time.time()

                logger.log("Training dynamics model for %i epochs ..." % (self.dynamics_model_max_epochs))
                self.dynamics_model.fit(samples_data['observations'],
                                        samples_data['actions'],
                                        samples_data['next_observations'],
                                        cpc_train_buffer=train_data,
                                        cpc_val_buffer=validation_data,
                                        epochs=self.dynamics_model_max_epochs, verbose=False, log_tabular=True)

                logger.record_tabular('Time-ModelFit', time.time() - time_fit_start)

                ''' --------------- fit reward model --------------- '''
                time_fitrew_start = time.time()

                if self.reward_model is not None:
                    if self.reward_sample_processor is not None:
                        samples_data = self.reward_sample_processor.process_samples(env_paths, log=False)
                    logger.log("Training reward model for %i epochs ..." % (self.reward_model_max_epochs))
                    self.reward_model.fit(samples_data['observations'],
                                            samples_data['actions'],
                                            samples_data['next_observations'],
                                            samples_data['rewards'],
                                            epochs=self.reward_model_max_epochs, verbose=False, log_tabular=True,
                                            prefix='Rew')

                    logger.record_tabular('Time-RewardModelFit', time.time() - time_fitrew_start)


                """ --------------------Train embedding to true state -------------"""
                if self.train_emb_to_state and itr % 10 == 0:
                    assert self.random_buffer is not None
                    latent_noises = [0., 0.3, 0.5, 0.7, 1.]

                    for noise in latent_noises:
                        # reinitialize weights
                        emb2state_model.load_weights(os.path.join(logger.get_dir(), 'emb2state_model.h5'))
                        test_traj = 5
                        # save 3 trajectories to test on
                        emb2state_model.compile(
                            optimizer=keras.optimizers.Adam(lr=1e-3),
                            loss='mean_squared_error',
                            metrics=['mean_squared_error'], )

                        callbacks = [
                            keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=1 / 3, patience=2, min_lr=1e-5,
                                                              verbose=1, min_delta=0.01),
                            keras.callbacks.CSVLogger(os.path.join(logger.get_dir(), 'emb2state.log'), append=True),
                            keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)]

                        latent_x = self.random_buffer._embedding_dataset['obs'][:-test_traj].reshape((-1, self.latent_dim))
                        latent_x += np.random.normal(scale = noise, size=latent_x.shape)

                        emb2state_model.fit\
                            (x=latent_x,
                             y=self.random_buffer._dataset['true_state'][:-test_traj].reshape((-1, self.state_dim)),
                             epochs=30,
                             validation_split=0.1,
                             callbacks=callbacks
                             )

                        num_stack = self.dynamics_model.num_stack


                        if noise == 0.:
                            self.random_buffer.update_embedding_buffer()
                            one_step_error = []

                            for buffer in [self.random_buffer, self.buffer]:
                                # record the one step prediction error in state space
                                data_act, data_obs = buffer._embedding_dataset['act'][-test_traj:], \
                                                                 buffer._embedding_dataset['obs'][-test_traj:]

                                data_nextstate = buffer._dataset['true_state'][-test_traj:, 1:]
                                data_obs = np.concatenate(
                                    [np.zeros((data_obs.shape[0], num_stack - 1, *data_obs.shape[2:])), data_obs],
                                    axis=1)
                                obs_stack = np.stack([data_obs[:, offset: data_obs.shape[1] + offset - num_stack + 1]
                                                      for offset in range(num_stack)], axis=2)[:, :-1]
                                states = obs_stack.reshape(-1, num_stack, self.latent_dim)
                                actions = data_act.reshape(-1, data_act.shape[-1])
                                predicted_next_latent = sess.run(self.dynamics_model.predict_sym(states.astype(np.float32),
                                                                                                 actions.astype(np.float32))[:, -1])
                                predicted_next_states = emb2state_model.predict(predicted_next_latent)

                                actual_next_states = data_nextstate.reshape(-1, data_nextstate.shape[-1])
                                error = predicted_next_states - actual_next_states
                                one_step_error.append(error)

                            logger.logkv('OneStepErrorRandomTraj', np.mean(np.linalg.norm(one_step_error[0], axis=1)))
                            logger.logkv('OneStepErrorMPCTraj', np.mean(np.linalg.norm(one_step_error[1], axis=1)))



                        for test_noise in latent_noises:
                            for prefix, buffer in [('random', self.random_buffer), ('mpc', self.buffer)]:
                                # plot the state predictions along time
                                latents = buffer._embedding_dataset['obs'][-test_traj:].reshape((-1, self.latent_dim))
                                predicted_states = emb2state_model.predict(latents + np.random.normal(
                                    scale = test_noise, size=latents.shape))
                                predicted_states = predicted_states.reshape(test_traj, -1, self.state_dim)
                                true_states = buffer._dataset['true_state'][-test_traj:]

                                # initial_latent = self.random_buffer._embedding_dataset['obs'][-test_traj:, :num_stack]
                                # actions_to_take = self.random_buffer._dataset['act'][-test_traj:, num_stack - 1 : num_stack + 49]
                                #
                                # openloop_predicted_latents = self.dynamics_model.openloop_rollout(initial_latent, actions_to_take)
                                # openloop_predicted_states = emb2state_model.predict\
                                #     (openloop_predicted_latents.reshape(-1, self.latent_dim)).reshape(test_traj, -1, self.state_dim)
                                #
                                for counter, (true_traj, predicted_traj) in \
                                        enumerate(zip(true_states[:2], predicted_states[:2])):#, openloop_predicted_states[:2])):
                                    num_plots = self.state_dim
                                    fig = plt.figure(figsize=(num_plots, 6))
                                    for i in range(num_plots):
                                        ax = plt.subplot(num_plots // 6 + 1, 6, i + 1)
                                        plt.plot(true_traj[num_stack:num_stack+50, i], label='true state')
                                        plt.plot(predicted_traj[num_stack:num_stack+50, i], label='predicted state')
                                    plt.savefig(os.path.join(logger.get_dir(), '%s_itr%d_trnoise=% 3.1f_tenoise=% 3.1f_%d.png' \
                                                             % (prefix, itr, noise, test_noise, counter)))

                                    # plt.clf()
                                    # for i in range(num_plots):
                                    #     ax = plt.subplot(num_plots // 6 + 1, 6, i + 1)
                                    #     plt.plot(true_traj[num_stack:num_stack+50, i], label='true state')
                                    #     plt.plot(predicte_openloop_traj[:, i], label='openloop')
                                    # plt.savefig(os.path.join(logger.get_dir(), 'openloop_itr%d_noise=% 3.1f_%d.png' \
                                    #                          % (itr, noise, counter)))


                """ ------------------- Logging Stuff --------------------------"""
                logger.logkv('Itr', itr)
                logger.logkv('n_timesteps', self.sampler.total_timesteps_sampled)

                logger.logkv('Time', time.time() - start_time)
                logger.logkv('ItrTime', time.time() - itr_start_time)

                logger.logkv('ActionMean', np.mean([np.mean(np.absolute(path['actions'])) for path in env_paths]))
                logger.logkv('LatentMean', np.mean(self.buffer._embedding_dataset['obs']))
                logger.logkv('LatentStd', np.mean(np.std(self.buffer._embedding_dataset['obs'], axis=-1)))

                # logger.log("Saving snapshot...")
                # params = self.get_itr_snapshot(itr)
                self.log_diagnostics(env_paths, '')
                # logger.save_itr_params(itr, params)
                # logger.log("Saved")

                logger.dumpkvs()
                #if itr == 0:
                #    sess.graph.finalize()

        logger.log("Training finished")
        self.sess.close()

    def get_itr_snapshot(self, itr):
        """
        Gets the current policy and env for storage
        """
        return dict(itr=itr, policy=self.policy, env=self.env, dynamics_model=self.dynamics_model, reward_model=self.reward_model)

    def log_diagnostics(self, paths, prefix):
        if hasattr(self.env, 'log_diagnostics'):
            self.env.log_diagnostics(paths, prefix)
        self.policy.log_diagnostics(paths, prefix)

    def get_seqs(self, env_paths, test_size=0.1):
        if self.env.encoder is not None:
            img_seqs = np.stack([path['env_infos']['image'] for path in env_paths])[:, :-1]  # N x T x (img_shape)
            action_seqs = np.stack([path['actions'] for path in env_paths])[:, 1:]
            rew_seqs = np.stack([path['rewards'] for path in env_paths])[:, 1:]
        else:
            img_seqs = np.stack([path['observations'] for path in env_paths])  # N x T x (img_shape)
            action_seqs = np.stack([path['actions'] for path in env_paths])
            rew_seqs = np.stack([path['rewards'] for path in env_paths])

        rew_seqs = rew_seqs[:, :, None]

        train_img, val_img, train_action, val_action, train_rew, val_rew  = \
            train_test_split(img_seqs, action_seqs, rew_seqs, test_size=test_size)

        return train_img, val_img, train_action, val_action, train_rew, val_rew

