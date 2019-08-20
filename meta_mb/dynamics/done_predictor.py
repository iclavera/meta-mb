from meta_mb.dynamics import nn

import tensorflow as tf
import numpy as np
from pdb import set_trace as st

class CoreModel(object):
    def __init__(self,
                 name,
                 obs_dim,
                 action_dim,
                 aux_hidden_dim=64,
                 transition_hidden_dim=64,
                 learning_rate = 1e-3,
                 ensemble_num = 1):
        self.name = name
        self.aux_hidden_dim = aux_hidden_dim
        self.transition_hidden_dim = transition_hidden_dim
        self.learning_rate = learning_rate
        self.ensemble_num = ensemble_num
        self.obs_dim = obs_dim
        self.action_dim = action_dim

        with tf.variable_scope(self.name):
            self.epoch_n = tf.get_variable('epoch_n', [], initializer=tf.constant_initializer(0), dtype=tf.int64, trainable=False)
            self.update_n = tf.get_variable('update_n', [], initializer=tf.constant_initializer(0), dtype=tf.int64, trainable=False)
            self.frame_n = tf.get_variable('frame_n', [], initializer=tf.constant_initializer(0), dtype=tf.int64, trainable=False)
            self.epoch_n_placeholder = tf.placeholder(tf.int64, [])
            self.update_n_placeholder = tf.placeholder(tf.int64, [])
            self.frame_n_placeholder = tf.placeholder(tf.int64, [])
        self.assign_epoch_op = [tf.assign(self.epoch_n, self.epoch_n_placeholder), tf.assign(self.update_n, self.update_n_placeholder), tf.assign(self.frame_n, self.frame_n_placeholder)]

        self.create_params()
        self.model_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.name)


    def create_params(self):
        raise Exception("unimplemented")

    def update_epoch(self, sess, epoch, updates, frames):
        sess.run(self.assign_epoch_op, feed_dict={self.epoch_n_placeholder: int(epoch), self.update_n_placeholder: int(updates), self.frame_n_placeholder: int(frames)})

    def save(self, sess, path, epoch=None):
        if epoch is None:  self.saver.save(sess, path + "/%s.params" % self.saveid)
        else:              self.saver.save(sess, path + "/%09d_%s.params" % (epoch, self.saveid))

    def load(self, sess, path, epoch=None):
        if epoch is None:  self.saver.restore(sess, path + "/%s.params" % self.saveid)
        else:              self.saver.restore(sess, path + "/%09d_%s.params" % (epoch, self.saveid))


class DonePredictor(CoreModel):
    def create_params(self):
        with tf.variable_scope(self.name):
            if self.ensemble_num > 1:
                self.done_predictor = nn.EnsembleFeedForwardNet('done_predictor', self.obs_dim + self.obs_dim + self.action_dim, [], layers=4, hidden_dim=self.aux_hidden_dim, get_uncertainty=True)
            else:
                self.done_predictor = nn.FeedForwardNet('done_predictor', self.obs_dim + self.obs_dim + self.action_dim, [], layers=4, hidden_dim=self.aux_hidden_dim, get_uncertainty=True)
            self.optimizer = tf.train.AdamOptimizer(self.learning_rate)
            self.input = tf.placeholder(tf.float32, shape=(None, self.obs_dim + self.obs_dim + self.action_dim))
            self.done_prediction = tf.placeholder(tf.float32, shape=(None))
            self.input = tf.placeholder(tf.float32, shape=(None, self.obs_dim + self.obs_dim + self.action_dim))
            self.dones = tf.placeholder(tf.float32, shape=(None))
        self.fit(self.input, self.dones)

    def fit(self, input, output):
        # info = tf.concat([obs, actions], -1)
        # next_info = tf.concat([next_obs, info], -1)
        self.done_prediction = self.done_predictor(input, is_eval=False, reduce_mode="random")
        output = tf.cast(output, tf.float32)
        done_losses = tf.nn.sigmoid_cross_entropy_with_logits(labels=output, logits=self.done_prediction)
        done_loss = tf.reduce_mean(done_losses)
        reg_loss = .0001 * (self.done_predictor.l2_loss())
        self.loss = reg_loss+done_loss
        traininig_op = self.optimizer.minimize(self.loss)


    def tf_predict(self, obs, actions, next_obs):
        info = tf.concat([obs, actions], -1)
        next_info = tf.concat([next_obs, info], -1)
        return self.done_predictor(next_info, is_eval=False, reduce_mode="random")

    def predict(self, obs, actions, next_obs):
        info = np.concatenate([obs, actions], -1)
        next_info = np.concatenate([next_obs, info], -1)
        sess = tf.get_default_session()
        return sess.run(self.done_prediction, feed_dict=dict({self.input: next_info}))

    def transition(self, obs, action, next_obs):
        info = tf.concat([obs, action], -1)
        next_info = tf.concat([next_obs, info], -1)
        done = tf.nn.sigmoid(self.done_predictor(next_info, reduce_mode="none", pre_expanded=True))
        return done
