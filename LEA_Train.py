# @Author: Jinyu Zhang
# @Time: 2022/8/24 15:25
# @E-mail: JinyuZ1996@outlook.com

# coding: utf-8
from LEA_Evaluate import *
from LEA_Model import *

args = Settings()

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # log
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_num  # cuda device number


def train(sess, GCN_net, batches_train):
    # define the buffer that accepts variables.
    uid_all, seq_A_all, seq_B_all, len_A_all, len_B_all, pos_A_all, pos_B_all, target_A_all, target_B_all, train_batch_num = (
        batches_train[0], batches_train[1], batches_train[2], batches_train[3],
        batches_train[4], batches_train[5], batches_train[6], batches_train[7], batches_train[8], batches_train[9])

    shuffled_batch_indexes = np.random.permutation(train_batch_num)
    avg_loss_A = 0
    avg_loss_B = 0

    for batch_index in shuffled_batch_indexes:
        uid = uid_all[batch_index]
        seq_A = seq_A_all[batch_index]
        seq_B = seq_B_all[batch_index]
        len_A = len_A_all[batch_index]
        len_B = len_B_all[batch_index]
        pos_A = pos_A_all[batch_index]
        pos_B = pos_B_all[batch_index]
        target_A = target_A_all[batch_index]
        target_B = target_B_all[batch_index]

        train_loss_A, train_loss_B, _, _ = GCN_net.train_GCN(sess=sess, uid=uid, seq_A=seq_A, seq_B=seq_B, len_A=len_A,
                                                             len_B=len_B,
                                                             pos_A=pos_A, pos_B=pos_B,
                                                             target_A=target_A, target_B=target_B,
                                                             dropout_rate=args.dropout_rate, keep_prob=args.keep_prob)
        avg_loss_A += train_loss_A
        avg_loss_B += train_loss_B

    rec_loss_A = avg_loss_A / train_batch_num
    rec_loss_B = avg_loss_B / train_batch_num
    return rec_loss_A, rec_loss_B


def evaluate_module(sess, GCN_net, test_batches, test_len):
    # define the buffer that accepts variables.
    uid_all, seq_A_all, seq_B_all, len_A_all, len_B_all, pos_A_all, pos_B_all, target_A_all, target_B_all, test_batch_num \
        = (test_batches[0], test_batches[1], test_batches[2], test_batches[3], test_batches[4],
           test_batches[5], test_batches[6], test_batches[7], test_batches[8], test_batches[9])

    return evaluate_ratings(sess=sess, GCN_net=GCN_net, uid=uid_all, seq_A=seq_A_all, seq_B=seq_B_all,
                            len_A=len_A_all, len_B=len_B_all,
                            pos_A=pos_A_all, pos_B=pos_B_all,
                            target_A=target_A_all, target_B=target_B_all,
                            test_batch_num=test_batch_num, test_length=test_len)
