# @Author: Jinyu Zhang
# @Time: 2022/8/24 9:31
# @E-mail: JinyuZ1996@outlook.com

import os
import random
import numpy as np
import scipy.sparse as sp
import tensorflow as tf
from LEA_Setting import *

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
seed = 2022
random.seed(seed)
np.random.seed(seed)
tf.set_random_seed(seed)
args = Settings()

def _convert_sp_mat_to_sp_tensor(X):
    coo = X.tocoo().astype(np.float32)
    indices = np.mat([coo.row, coo.col]).transpose()
    return tf.SparseTensor(indices, coo.data, coo.shape)


class LEA_GCN:
    def __init__(self, n_items_A, n_items_B, n_users, graph_matrix):
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_num
        self.config = tf.ConfigProto()
        self.config.gpu_options.allow_growth = True
        self.n_items_A = n_items_A
        self.n_items_B = n_items_B
        self.n_users = n_users
        self.graph_matrix = graph_matrix
        self.embedding_size = args.embedding_size
        self.n_fold = args.n_fold
        self.alpha = args.alpha
        self.layer_size = args.layer_size
        self.beta = args.beta
        self.regular_rate_att = args.regular_rate_att
        self.num_heads = args.num_heads
        self.n_layers = args.num_layers
        self.lr_A = args.lr_A
        self.lr_B = args.lr_B
        self.l2_regular_rate = args.l2_regular_rate
        self.dim_coefficient = args.dim_coefficient
        self.batch_size = args.batch_size
        self.weight_size = eval(self.layer_size)
        self.is_training = True
        self.graph = tf.Graph()
        with self.graph.as_default():
            with tf.name_scope('inputs'):
                self.uid, self.seq_A, self.seq_B, self.len_A, self.len_B, self.pos_A, self.pos_B, self.target_A, \
                self.target_B, self.dropout_rate, self.keep_prob = self.get_inputs()

            with tf.name_scope('encoder'):
                self.all_weights = self._init_weights()
                self.i_embeddings_A, self.u_embeddings, self.i_embeddings_B = \
                    self.graph_encoder(self.n_items_A, self.n_users, self.n_items_B, self.graph_matrix)
                self.seq_emb_A_output, self.seq_emb_B_output = self.seq_encoder(self.uid, self.seq_A, self.seq_B,
                                                                                self.pos_A, self.pos_B,
                                                                                self.dropout_rate, self.i_embeddings_A,
                                                                                self.u_embeddings, self.i_embeddings_B)
            with tf.name_scope('prediction_A'):
                self.pred_A = self.prediction_A(self.n_items_A, self.seq_emb_B_output, self.seq_emb_A_output,
                                                self.keep_prob)
            with tf.name_scope('prediction_B'):
                self.pred_B = self.prediction_B(self.n_items_B, self.seq_emb_A_output, self.seq_emb_B_output,
                                                self.keep_prob)
            with tf.name_scope('loss'):
                self.loss_A, self.loss_B = self.cal_loss(self.target_A, self.pred_A, self.target_B, self.pred_B)
            with tf.name_scope('optimizer'):
                self.train_op_A = self.optimizer(self.loss_A, self.lr_A)
                self.train_op_B = self.optimizer(self.loss_B, self.lr_B)

    def get_inputs(self):
        uid = tf.placeholder(dtype=tf.int32, shape=[None, ], name='uid')
        seq_A = tf.placeholder(dtype=tf.int32, shape=[None, None], name='seq_A')
        seq_B = tf.placeholder(dtype=tf.int32, shape=[None, None], name='seq_B')
        len_A = tf.placeholder(dtype=tf.int32, shape=[None, 1], name='len_A')
        len_B = tf.placeholder(dtype=tf.int32, shape=[None, 1], name='len_B')
        pos_A = tf.placeholder(dtype=tf.int32, shape=[None, None], name="pos_A")
        pos_B = tf.placeholder(dtype=tf.int32, shape=[None, None], name="pos_B")
        target_A = tf.placeholder(dtype=tf.int32, shape=[None, ], name='target_A')
        target_B = tf.placeholder(dtype=tf.int32, shape=[None, ], name='target_B')
        dropout_rate = tf.placeholder(dtype=tf.float32, shape=[], name='dropout_rate')
        keep_prob = tf.placeholder(dtype=tf.float32, shape=[], name='keep_prob')
        return uid, seq_A, seq_B, len_A, len_B, pos_A, pos_B, target_A, target_B, dropout_rate, keep_prob

    def _init_weights(self):
        all_weights = dict()
        initializer = tf.contrib.layers.xavier_initializer()
        all_weights['user_embedding'] = tf.Variable(initializer([self.n_users, self.embedding_size]))
        all_weights['item_embedding_A'] = tf.Variable(initializer([self.n_items_A, self.embedding_size]))
        all_weights['item_embedding_B'] = tf.Variable(initializer([self.n_items_B, self.embedding_size]))
        all_weights['pos_embedding_A'] = tf.Variable(initializer([self.n_items_A, self.embedding_size]))
        all_weights['pos_embedding_B'] = tf.Variable(initializer([self.n_items_B, self.embedding_size]))
        # parameters for domain A
        all_weights['W_att_A'] = tf.Variable(initializer([self.embedding_size, self.weight_size[0]]), dtype=tf.float32)
        all_weights['b_att_A'] = tf.Variable(initializer([1, self.weight_size[0]]), dtype=tf.float32)
        all_weights['h_att_A'] = tf.Variable(tf.ones([self.weight_size[0], 1]), dtype=tf.float32)
        # parameters for domain B
        all_weights['W_att_B'] = tf.Variable(initializer([self.embedding_size, self.weight_size[0]]), dtype=tf.float32)
        all_weights['b_att_B'] = tf.Variable(initializer([1, self.weight_size[0]]), dtype=tf.float32)
        all_weights['h_att_B'] = tf.Variable(tf.ones([self.weight_size[0], 1]), dtype=tf.float32)
        return all_weights

    def unzip_laplace(self, X):
        unzip_info = []
        fold_len = (X.shape[0]) // self.n_fold
        for i_fold in range(self.n_fold):
            start = i_fold * fold_len
            if i_fold == self.n_fold - 1:
                end = X.shape[0]
            else:
                end = (i_fold + 1) * fold_len

            unzip_info.append(_convert_sp_mat_to_sp_tensor(X[start:end]))
        return unzip_info

    def graph_encoder(self, n_items_A, n_users, n_items_B, graph_matrix):
        # Generate a set of adjacency sub-matrix.
        graph_info = self.unzip_laplace(graph_matrix)

        ego_embeddings = tf.concat([self.all_weights['item_embedding_A'], self.all_weights['user_embedding'],
                                    self.all_weights['item_embedding_B']], axis=0)  # 基本结点表示
        all_embeddings = [ego_embeddings]
        for k in range(0, self.n_layers):
            temp_embed = []
            for f in range(args.n_fold):
                temp_embed.append(tf.sparse_tensor_dense_matmul(graph_info[f], ego_embeddings))  # 把结点表示加上额外的大阵信息

            # sum messages of neighbors.
            side_embeddings = tf.concat(temp_embed, 0)

            all_embeddings += [side_embeddings]

        all_embeddings = tf.stack(all_embeddings, 1)  # layer-wise aggregation
        all_embeddings = tf.reduce_mean(all_embeddings, axis=1, keepdims=False)  # sum the layer aggregation and the normalizer
        g_embeddings_A, u_g_embeddings, g_embeddings_B = tf.split(all_embeddings, [n_items_A, n_users, n_items_B], 0)

        return g_embeddings_A, u_g_embeddings, g_embeddings_B

    def seq_encoder(self, uid, seq_A, seq_B, pos_A, pos_B, dropout_rate, i_embeddings_A, u_embeddings, i_embeddings_B):
        with tf.variable_scope('seq_encoder'):
            self.user_embed = tf.nn.embedding_lookup(u_embeddings, uid)
            ##### domain A:
            item_embed_A = tf.nn.embedding_lookup(i_embeddings_A, seq_A)
            pos_embed_A = tf.nn.embedding_lookup(self.all_weights['pos_embedding_A'], pos_A, name="ebd_pos_A")
            item_pos_A = item_embed_A + self.alpha * pos_embed_A

            # 1 simple max_pooling
            # seq_embed_A_state = tf.reduce_max((seq_emb_A_output), 1)

            # 2 using EA_channel_1 for collaborative filtering signals
            ext_embed_A1 = self.ext_attention_encoder_1(item_embed_A, is_A=True)

            # 3 using EA_channel_2 for positional encoding and sequential pattern
            ext_embed_A2 = self.ext_attention_encoder_2(item_pos_A, dim_coefficient=self.dim_coefficient)

            self.seq_embed_A = ext_embed_A1 + ext_embed_A2

            seq_emb_A_output = tf.concat([self.seq_embed_A, self.user_embed], axis=1)
            seq_emb_A_output = tf.layers.dropout(seq_emb_A_output, rate=dropout_rate,
                                                 training=tf.convert_to_tensor(self.is_training))
            ##### domain B
            item_embed_B = tf.nn.embedding_lookup(i_embeddings_B, seq_B)
            pos_embed_B = tf.nn.embedding_lookup(self.all_weights['pos_embedding_B'], pos_B, name="ebd_pos_B")
            item_pos_B = item_embed_B + self.alpha * pos_embed_B
            # 1 simple max_pooling
            # seq_embed_B_state = tf.reduce_max((seq_emb_B_output), 1)

            # 2 using EA_channel_1 for collaborative filtering signals
            ext_embed_B1 = self.ext_attention_encoder_1(item_embed_B, is_A=False)

            # 3 using EA_channel_2 for positional encoding and sequential pattern
            ext_embed_B2 = self.ext_attention_encoder_2(item_pos_B, dim_coefficient=self.dim_coefficient)

            self.seq_embed_B = ext_embed_B1 + ext_embed_B2

            seq_emb_B_output = tf.concat([self.seq_embed_B, self.user_embed], axis=1)
            seq_emb_B_output = tf.layers.dropout(seq_emb_B_output, rate=dropout_rate,
                                                 training=tf.convert_to_tensor(self.is_training))
            print(seq_emb_A_output)
            print(seq_emb_B_output)

        return seq_emb_A_output, seq_emb_B_output

    def prediction_A(self, n_items_A, seq_emb_B_output, seq_emb_A_output, keep_prob):
        with tf.variable_scope('prediction_A'):
            concat_output = tf.concat([seq_emb_B_output, seq_emb_A_output], axis=-1)
            print(concat_output)
            concat_output = tf.nn.dropout(concat_output,
                                          keep_prob)
            pred_A = tf.layers.dense(concat_output, n_items_A, activation=None,
                                     kernel_initializer=tf.contrib.layers.xavier_initializer(
                                         uniform=False))
            print(pred_A)

            return pred_A

    def prediction_B(self, n_items_B, seq_emb_A_output, seq_emb_B_output, keep_prob):

        with tf.variable_scope('prediction_B'):
            concat_output = tf.concat([seq_emb_A_output, seq_emb_B_output], axis=-1)
            print(concat_output)
            concat_output = tf.nn.dropout(concat_output,
                                          keep_prob)
            pred_B = tf.layers.dense(concat_output, n_items_B, activation=None,
                                     kernel_initializer=tf.contrib.layers.xavier_initializer(
                                         uniform=False))
            print(pred_B)

            return pred_B

    def cal_loss(self, target_A, pred_A, target_B, pred_B):

        loss_A1 = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=target_A, logits=pred_A)
        loss_A1 = tf.reduce_mean(loss_A1, name='loss_A')
        loss_A2 = self.l2_regular_rate * tf.reduce_sum(tf.square(self.seq_embed_A)) + \
                  self.l2_regular_rate * tf.reduce_sum(tf.square(self.user_embed))
        loss_A = loss_A1 + loss_A2

        loss_B1 = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=target_B, logits=pred_B)
        loss_B1 = tf.reduce_mean(loss_B1, name='loss_B')
        loss_B2 = self.l2_regular_rate * tf.reduce_sum(tf.square(self.seq_embed_B)) + \
                  self.l2_regular_rate * tf.reduce_sum(tf.square(self.user_embed))
        loss_B = loss_B1 + loss_B2

        return loss_A, loss_B

    def optimizer(self, loss, learning_rate):
        optimizer = tf.train.AdamOptimizer(learning_rate)
        gradients = optimizer.compute_gradients(loss)
        capped_gradients = [(tf.clip_by_value(grad, -5., 5.), var) for grad, var in gradients if
                            grad is not None]
        train_op = optimizer.apply_gradients(capped_gradients)
        return train_op

    def _si_norm_lap(self, adj):
        rowsum = np.array(adj.sum(1))

        d_inv = np.power(rowsum, -1).flatten()
        d_inv[np.isinf(d_inv)] = 0.
        d_mat_inv = sp.diags(d_inv)

        norm_adj = d_mat_inv.dot(adj)
        return norm_adj.tocoo()

    def ext_attention_encoder_1(self, seq_ebd, is_A):
        with tf.variable_scope("ex_att_sum"):
            if is_A:
                shape_0 = tf.shape(seq_ebd)[0]  # [?]
                shape_1 = tf.shape(seq_ebd)[1]  # [?]
                meo_ebd_1 = tf.matmul(tf.reshape(seq_ebd, [-1, self.embedding_size]),
                                      self.all_weights['W_att_A']) + self.all_weights['b_att_A']  # [?, 16]
                meo_ebd_1 = tf.nn.relu(meo_ebd_1)  # [?, 16]
                dim_trans_1 = tf.reshape(tf.matmul(meo_ebd_1, self.all_weights['h_att_A']),
                                         [shape_0, shape_1])  # [?, ?]
                dim_trans_1 = tf.exp(dim_trans_1)  # [?, ?]
                mask_index_A = tf.reduce_sum(self.len_A, 1)  # [?, ]
                mask_matrix_A = tf.sequence_mask(mask_index_A, maxlen=shape_1, dtype=tf.float32)  # [?, ?]
                masked_ebd_A = mask_matrix_A * dim_trans_1  # [?, ?]
                exp_sum_A = tf.reduce_sum(masked_ebd_A, 1, keepdims=True)  # [?, 1]
                exp_sum_A = tf.pow(exp_sum_A, tf.constant(self.beta, tf.float32, [1]))  # [?, 1]

                score_A = tf.expand_dims(tf.div(masked_ebd_A, exp_sum_A), 2)  # [?, ?, 1]

                return tf.reduce_sum(score_A * seq_ebd, 1)
            else:
                shape_0 = tf.shape(seq_ebd)[0]
                shape_1 = tf.shape(seq_ebd)[1]
                mlp_output_B = tf.matmul(tf.reshape(seq_ebd, [-1, self.embedding_size]),
                                         self.all_weights['W_att_B']) + self.all_weights['b_att_B']
                mlp_output_B = tf.nn.tanh(mlp_output_B)
                d_trans_B = tf.reshape(tf.matmul(mlp_output_B, self.all_weights['h_att_B']), [shape_0, shape_1])
                d_trans_B = tf.exp(d_trans_B)
                mask_index_B = tf.reduce_sum(self.len_B, 1)
                mask_mat_B = tf.sequence_mask(mask_index_B, maxlen=shape_1, dtype=tf.float32)
                d_trans_B = mask_mat_B * d_trans_B
                exp_sum_B = tf.reduce_sum(d_trans_B, 1, keepdims=True)
                exp_sum_B = tf.pow(exp_sum_B, tf.constant(self.beta, tf.float32, [1]))

                score_B = tf.expand_dims(tf.div(d_trans_B, exp_sum_B), 2)

                return tf.reduce_sum(score_B * seq_ebd, 1)

    def ext_attention_encoder_2(self, seq_ebd, dim_coefficient):  # seq_ebd[?, ?, 16] dim_coef = 4
        input_dim_0 = tf.shape(seq_ebd)[0]  # ?
        input_dim_1 = tf.shape(seq_ebd)[1]  # ?
        dense_layer_1 = tf.layers.dense(seq_ebd, self.embedding_size * dim_coefficient, activation=None,
                                        kernel_initializer=tf.contrib.layers.xavier_initializer(
                                            uniform=False))  # [?, ?, 64]
        reshape_1 = tf.reshape(dense_layer_1, shape=(input_dim_0, input_dim_1, self.num_heads,
                                                     self.embedding_size * dim_coefficient // self.num_heads))  # [?, ?, 2, 32]
        reorder_1 = tf.transpose(reshape_1, perm=[0, 2, 1, 3])  # [?, 2, ?, 32]
        # a linear layer for key_vectors
        meo_key_vec = tf.layers.dense(reorder_1, self.batch_size // dim_coefficient, activation=None,
                                      kernel_initializer=tf.contrib.layers.xavier_initializer(
                                          uniform=False))  # [?, 2, ?, 64]
        # normalize attention map
        meo_key_vec = tf.nn.softmax(meo_key_vec, axis=2)  # [?, 2, ?, 64]
        # dobule-normalization
        meo_key_vec = meo_key_vec / (
                self.regular_rate_att + tf.reduce_sum(meo_key_vec, axis=-1, keepdims=True))  # [?, 2, ?, 64]
        drop_layer_1 = tf.nn.dropout(meo_key_vec, self.keep_prob)  # [?, 2, ?, 64]

        # a linear layer for value_vectors
        meo_value_vec = tf.layers.dense(drop_layer_1, self.embedding_size * dim_coefficient // self.num_heads,
                                        activation=None, kernel_initializer=tf.contrib.layers.xavier_initializer(
                uniform=False))  # [?, 2, ?, 32]
        reorder_2 = tf.transpose(meo_value_vec, perm=[0, 2, 1, 3])  # [?, ?, 2, 32]
        reorder_2 = tf.reduce_sum(reorder_2, axis=2)  # 将多头注意力合并 [?, ?, 32]
        # x = tf.reshape(reorder_2, [input_dim_0, input_dim_1, self.embedding_size * dim_coefficient])  # [?, ?, 64]
        # x = tf.reduce_sum(x, axis=0)  # [?, 64]
        user_ebd = tf.reduce_sum(reorder_2, axis=1)  # [?, 32]
        # a linear layer to project original dim
        out_put_ebd = tf.layers.dense(user_ebd, self.embedding_size, activation=None,
                                      kernel_initializer=tf.contrib.layers.xavier_initializer(uniform=False))  # [?, 16]
        out_put_ebd = tf.nn.dropout(out_put_ebd, self.keep_prob)

        return out_put_ebd



    def train_GCN(self, sess, uid, seq_A, seq_B, len_A, len_B, pos_A, pos_B, target_A, target_B,
                  dropout_rate, keep_prob):

        feed_dict = {self.uid: uid, self.seq_A: seq_A, self.seq_B: seq_B, self.len_A: len_A, self.len_B: len_B,
                     self.pos_A: pos_A, self.pos_B: pos_B,
                     self.target_A: target_A, self.target_B: target_B,
                     self.dropout_rate: dropout_rate, self.keep_prob: keep_prob}

        return sess.run([self.loss_A, self.loss_B, self.train_op_A, self.train_op_B], feed_dict)

    def evaluate_gcn(self, sess, uid, seq_A, seq_B, len_A, len_B, pos_A, pos_B, target_A, target_B,
                     dropout_rate, keep_prob):
        feed_dict = {self.uid: uid, self.seq_A: seq_A, self.seq_B: seq_B,
                     self.len_A: len_A, self.len_B: len_B,
                     self.pos_A: pos_A, self.pos_B: pos_B,
                     self.target_A: target_A, self.target_B: target_B, self.dropout_rate: dropout_rate,
                     self.keep_prob: keep_prob}
        return sess.run([self.pred_A, self.pred_B], feed_dict)
