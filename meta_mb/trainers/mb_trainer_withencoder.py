import keras
import os
import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split
import time

from meta_mb.logger import logger
from meta_mb.unsupervised_learning.cpc.data_utils import CPCDataGenerator

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
            dynamics_sample_processor,
            policy,
            dynamics_model,
            n_itr,
            cpc_model,
            start_itr=0,
            initial_random_samples=True,
            sess=None,
            dynamics_model_max_epochs=200,
            reward_model_max_epochs=200,
            reward_model=None,
            reward_sample_processor=None,
            cpc_model_epoch=5,
            cpc_model_lr=1e-4,
            cpc_batch_size=32,
            cpc_negative_same_traj=0,
            ):
        self.env = env
        self.sampler = sampler
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
        self.cpc_model_epoch = cpc_model_epoch
        self.cpc_batch_size = cpc_batch_size
        self.cpc_negative_same_traj = cpc_negative_same_traj
        self.cpc_model_lr = cpc_model_lr

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
            for itr in range(self.start_itr, self.n_itr):
                itr_start_time = time.time()
                logger.log("\n ---------------- Iteration %d ----------------" % itr)

                time_env_sampling_start = time.time()

                if self.initial_random_samples and itr == 0:
                    logger.log("Obtaining random samples from the environment...")
                    env_paths = self.sampler.obtain_samples(log=True, random=True, log_prefix='')
                else:
                    logger.log("Obtaining samples from the environment using the policy...")
                    env_paths = self.sampler.obtain_samples(log=True, log_prefix='')

                if self.cpc_model_epoch > 0:
                    img_seqs = np.stack([path['env_infos']['image'] for path in env_paths])[:-1]  # N x T x (img_shape)
                    action_seqs = np.stack([path['actions'] for path in env_paths])[1:]
                    train_img, val_img, train_action, val_action = train_test_split(img_seqs, action_seqs)
                    print(train_img.shape, val_img.shape)

                    if itr == 0: # create the iterator for the first time
                        terms = self.cpc_model.get_layer('x_input').input_shape[1]
                        predict_terms = self.cpc_model.get_layer('y_input').input_shape[1]
                        negative_samples = self.cpc_model.get_layer('y_input').input_shape[2] - 1
                        train_data = CPCDataGenerator(train_img, train_action, self.cpc_batch_size, terms=terms,
                                                      negative_samples=negative_samples,
                                                      predict_terms=predict_terms,
                                                      negative_same_traj=self.cpc_negative_same_traj)
                        validation_data = CPCDataGenerator(val_img, val_action, self.cpc_batch_size, terms=terms,
                                                           negative_samples=negative_samples,
                                                           predict_terms=predict_terms,
                                                           negative_same_traj=self.cpc_negative_same_traj)
                    else: # amend to the dataset instead if it's already created
                        train_data.update_dataset(train_img, train_action)
                        validation_data.update_dataset(val_img, val_action)


                logger.record_tabular('Time-EnvSampling', time.time() - time_env_sampling_start)
                logger.log("Processing environment samples...")

                # first processing just for logging purposes
                time_env_samp_proc = time.time()
                samples_data = self.dynamics_sample_processor.process_samples(env_paths,
                                                                              log=True, log_prefix='EnvTrajs-')

                logger.record_tabular('Time-EnvSampleProc', time.time() - time_env_samp_proc)

                ''' --------------- fit dynamics model --------------- '''

                time_fit_start = time.time()

                logger.log("Training dynamics model for %i epochs ..." % (self.dynamics_model_max_epochs))
                self.dynamics_model.fit(samples_data['observations'],
                                        samples_data['actions'],
                                        samples_data['next_observations'],
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

                ''' --------------- finetune cpc --------------- '''
                time_cpc_start = time.time()
                if self.cpc_model_epoch > 0 and train_data.n_seqs > 5:

                    callbacks = [keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0.01, patience=2, verbose=1),
                                 keras.callbacks.LearningRateScheduler(lambda epoch, lr: self.cpc_model_lr / (10 ** (epoch // 3)), verbose=1), # TODO: better lr schedule
                                 #keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=1 / 3, patience=2, min_lr=1e-5, verbose=1, min_delta=0.001),
                                 # SaveEncoder(output_dir),
                                 keras.callbacks.CSVLogger(os.path.join(logger.get_dir(), 'cpc.log'), append=True)]

                    # Train the model
                    self.cpc_model.fit_generator(
                        generator=train_data,
                        steps_per_epoch=len(train_data),
                        validation_data=validation_data,
                        validation_steps=len(validation_data),
                        epochs=self.cpc_model_epoch,
                        verbose=1,
                        callbacks=callbacks
                    )

                logger.record_tabular('Time-CPCModelFinetune', time.time() - time_cpc_start)


                """ ------------------- Logging Stuff --------------------------"""
                logger.logkv('Itr', itr)
                logger.logkv('n_timesteps', self.sampler.total_timesteps_sampled)

                logger.logkv('Time', time.time() - start_time)
                logger.logkv('ItrTime', time.time() - itr_start_time)

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
