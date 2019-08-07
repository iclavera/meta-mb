import numpy as np
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import time

class CPCDataGenerator(object):
    def __init__(self, img_seqs, action_seqs, batch_size, terms, negative_samples=1, predict_terms=1, negative_same_traj=0,
                 max_num_seq=512, predict_action=False, no_neg=False):
        self.batch_size = batch_size
        self.images = img_seqs
        self.actions = action_seqs
        self.negative_samples = negative_samples
        self.negative_same_traj = negative_same_traj
        self.predict_terms = predict_terms
        self.terms = terms

        self.n_seqs = self.images.shape[0]
        self.n_step = self.images.shape[1]
        self.n_chunks = self.images.shape[1] - self.terms - self.predict_terms + 1 # number of chunks in each time sequence
        self.n_samples = self.n_seqs * self.n_chunks

        self.max_num_seq = max_num_seq
        self.predict_action = predict_action
        self.no_neg = no_neg
        if self.predict_action:
            self.y = np.zeros(dtype=np.float32, shape=(self.batch_size, self.predict_terms, self.negative_samples + 1,
                                                       action_seqs.shape[-1]))
        else:
            self.y = np.zeros(dtype=np.float32, shape=(self.batch_size, self.predict_terms, self.negative_samples+1,
                                                       *img_seqs.shape[-3:]))

        assert self.negative_same_traj < self.negative_samples

    def __iter__(self):
        return self

    #
    def __next__(self):
        return self.next()

    def __len__(self):
        return self.n_samples // self.batch_size

    def update_dataset(self, img_seqs, action_seqs):
        self.images = np.concatenate([self.images, img_seqs])
        self.actions = np.concatenate([self.actions, action_seqs])

        if self.images.shape[0] > self.max_num_seq:
            self.images = self.images[-self.max_num_seq:]
            self.actions = self.actions[-self.max_num_seq:]

        self.n_seqs = self.images.shape[0]
        self.n_step = self.images.shape[1]
        self.n_samples = self.n_seqs * self.n_chunks

    def next(self):
        """
        :return: [x_images, y_images], labels
        x_images shape: batch_size x terms
        y_images and labels shape: batch_size x predict_terms x (negative_samples + 1)
        """
        # get the starting index of x_images
        t0 = time.time()
        idx_n = np.random.randint(0, self.n_seqs, self.batch_size)
        start_idx_t = np.random.randint(0, self.n_chunks, (self.batch_size, 1))
        x_idx_t = np.array([], dtype=np.int32).reshape((self.batch_size, 0))
        for _ in range(self.terms):
            x_idx_t = np.concatenate([x_idx_t, start_idx_t], axis=-1)
            start_idx_t += 1
        if self.predict_action:
            x_idx_t = np.concatenate([x_idx_t, start_idx_t + self.predict_terms - 1], axis=-1)

        # gather the x_images
        x_images = self.images[idx_n[:, None], x_idx_t]
        t1 = time.time()

        # print('gather x images', t1 - t0)


        y_idx_t = np.array([], dtype=np.int32).reshape((self.batch_size, 0))
        if self.predict_action:
            # get the positive samples for actions
            start_idx_t -= 1
            for _ in range(self.predict_terms):
                y_idx_t = np.concatenate([y_idx_t, start_idx_t], axis=-1)
                start_idx_t += 1

            y_pos = self.actions[idx_n[:, None], y_idx_t]

        else:
            # get tht positive samples for y_images
            for _ in range(self.predict_terms):
                y_idx_t = np.concatenate([y_idx_t, start_idx_t], axis=-1)
                start_idx_t += 1

            y_pos = self.images[idx_n[:, None], y_idx_t]
            # gather the actions
            actions = self.actions[idx_n[:, None], y_idx_t - 1]

        if self.no_neg and self.predict_action:
            return x_images, y_pos.reshape((self.batch_size, -1))

        t2 = time.time()
        #
        # print('gather y positives', t2 - t1)

        # get the negative samples (batch_size x predict_terms x negative_samples)
        seq_index = np.arange(self.n_seqs)
        neg_idx_n = np.stack([np.random.choice(seq_index[seq_index!=i], size=(self.predict_terms, self.negative_samples - self.negative_same_traj))
                              for i in idx_n])
        neg_idx_t = np.random.randint(0, self.n_step, (self.batch_size, self.predict_terms, self.negative_samples - self.negative_same_traj))

        if self.negative_same_traj > 0:
            neg_idx_n2 = np.stack([i * np.ones(shape=(self.predict_terms, self.negative_same_traj), dtype=int) for i in idx_n])
            neg_idx_t2 = np.random.randint(0, self.n_step, (self.batch_size, self.predict_terms, self.negative_same_traj))
            equal_positive = neg_idx_t2 == y_idx_t[:, :, None]
            neg_idx_t2[equal_positive] = np.mod(neg_idx_t2[equal_positive] + np.random.randint(1, self.n_step - 1, size=neg_idx_t2[equal_positive].shape),
                                                self.n_step)

            neg_idx_n = np.concatenate([neg_idx_n, neg_idx_n2], axis = -1)
            neg_idx_t = np.concatenate([neg_idx_t, neg_idx_t2], axis=-1)

        if self.predict_action:
            # y_neg = self.actions[neg_idx_n, neg_idx_t]
            self.y[:, :, 0, ...] = y_pos
            self.y[:, :, 1:, ...] = self.actions[neg_idx_n, neg_idx_t]
            # y = np.concatenate([np.expand_dims(y_pos, 2), self.actions[neg_idx_n, neg_idx_t]], axis=2)
        else:
            # y_neg = self.images[neg_idx_n, neg_idx_t]
            self.y[:, :, 0, ...] = y_pos
            self.y[:, :, 1:, ...] = self.images[neg_idx_n, neg_idx_t]
            # y = np.concatenate([np.expand_dims(y_pos, 2), self.images[neg_idx_n, neg_idx_t]], axis=2)

        t3 = time.time()
        # print('gather y negatives', t3 - t2)


        # concatenate positive samples with negative ones


        pos_neg_label = np.zeros((self.batch_size, self.predict_terms, self.negative_samples + 1)).astype('int32')
        pos_neg_label[:, :, 0] = 1


        # permute the batch so that positive samples are at random places
        # rand_idx_n = np.arange(self.batch_size)[:, None, None]
        # rand_idx_t = np.arange(self.predict_terms)[None, :, None]
        # rand_idx_neg = np.stack([np.stack([np.random.permutation(self.negative_samples + 1)
        #                                    for i in range(self.predict_terms)]) for j in range(self.batch_size)])

        # idxs = np.random.choice(pos_neg_label.shape[2], pos_neg_label.shape[2], replace=False)

        # return [x_images, actions, y[rand_idx_n, rand_idx_t, rand_idx_neg, ...]], pos_neg_label[rand_idx_n, rand_idx_t, rand_idx_neg]
        return [x_images, actions, self.y], pos_neg_label

def plot_seq(x, y, labels, name=''):
    """
    :param x: terms x image_size
    :param y: predict_terms x (negative_samples + 1) x image_size
    :param labels: predict_terms x (negative_samples + 1) x 1
    """
    n_batches = x.shape[0]
    n_terms = x.shape[1]
    predict_terms = 1
    pos_neg_count = y.shape[2]
    counter = 1
    for n_b in range(n_batches):
        for n_t in range(n_terms):
            plt.subplot(n_batches, n_terms, counter)
            plt.imshow(x[n_b, n_t, ...])
            plt.axis('off')
            counter += 1
    plt.savefig(name + '_x')

    plt.clf()

    counter=1
    for n_b in range(n_batches):
        for n_neg in range(pos_neg_count):
            ax = plt.subplot(n_batches, pos_neg_count, counter)
            plt.imshow(y[n_b, 0, n_neg, ...])
            plt.axis('off')
            ax.set_title(labels[n_b, 0, n_neg])
            counter += 1
    plt.savefig(name+"_y")


if __name__ == "__main__":
    img_seqs = np.random.uniform(size=(20, 16, 64, 64, 3))
    data = CPCDataGenerator(img_seqs, 32, 1, negative_samples=3, predict_terms=1)
    for (x, y), labels in data:
        plot_seq(x, y, labels, name='point_mass_seq')
        break










