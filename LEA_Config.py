# @Author: Jinyu Zhang
# @Time: 2022/8/24 9:33
# @E-mail: jinyuZ1996@outlook.com

# coding: utf-8

import scipy.sparse as sp
import pandas as pd
import numpy as np
import random
import os

random.seed(2022)
np.random.seed(2022)

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

seed = 2022


def build_dict(dict_path):
    # build the dict from dictionary files generated from original datasets
    element_dict = {}
    with open(dict_path, 'r') as file_object:
        elements = file_object.readlines()
    for e in elements:
        e = e.strip().split('\t')
        element_dict[e[1]] = int(e[0])
    return element_dict


def build_mixed_sequences(path, dict_A, dict_B, dict_U):
    # read the data from original dataset and transform them into mixed sequential input
    with open(path, 'r') as file_object:
        mixed_sequences = []
        lines = file_object.readlines()
        for e in lines:
            e = e.strip().split('\t')
            temp_sequence = []
            user_id = e[0]  # the first element in each line is the user's id
            temp_sequence.append(dict_U[user_id])
            for item in e[1:]:
                # from the second element to the last of this line are the interacted items of this user
                if item in dict_A:
                    temp_sequence.append(dict_A[item])
                else:
                    temp_sequence.append(dict_B[item] + len(dict_A))  # to distinguish the items from different domains
            mixed_sequences.append(temp_sequence)
    return mixed_sequences


def data_generation(mixed_sequence, dict_A):
    # to transform the mixed sequence into the individual form:
    # sequence_A, sequence_B, length_A, length_B, target_item_A, target_item_B
    data_outputs = []
    for sequence_index in mixed_sequence:
        temp = []
        pos_mix = 0
        seq_A, seq_B = [], []
        pos_A, pos_B = [], []
        len_A, len_B = 0, 0
        uid = sequence_index[0]  # the first element is uid
        seq_A.append(uid)
        seq_B.append(uid)
        for item_id in sequence_index[1:]:
            # the first element is uid
            if item_id < len(dict_A):
                seq_A.append(item_id)
                pos_A.append(pos_mix)
                pos_mix += 1
                len_A += 1
            else:
                seq_B.append(item_id - len(dict_A))
                pos_B.append(pos_mix)
                pos_mix += 1
                len_B += 1
        target_A = seq_A[-1]
        target_B = seq_B[-1]
        seq_A.pop()
        seq_B.pop()  # pop the last item as the target_item
        pos_A.pop()
        pos_B.pop()
        temp.append(seq_A)
        temp.append(seq_B)
        temp.append(len_A - 1)  # because the last one is using as target item
        temp.append(len_B - 1)
        temp.append(pos_A)
        temp.append(pos_B)
        temp.append(target_A)
        temp.append(target_B)
        # extract target items from mixed sequences for B
        data_outputs.append(temp)

    return data_outputs


def get_rating_matrix(all_data_input):
    L_uid_itemA, L_uid_itemB, L_itemA_uid, L_itemB_uid, L_neighbor_item_A, L_neighbor_item_B, L_uid_uid \
        = [], [], [], [], [], [], []
    output_ratings = []
    for data_unit in all_data_input:
        seq_A = data_unit[0]
        seq_B = data_unit[1]
        items_A = [int(i) for i in seq_A[1:]]  # [0]是Uid
        items_B = [int(j) for j in seq_B[1:]]

        uid = int(seq_A[0])
        for item_A in items_A:
            L_uid_itemA.append([uid, item_A])  # 第1个list存的是uid和item_A的对儿[0]
            L_itemA_uid.append([item_A, uid])  # 第3个list存的是item_A和uid的对儿[2]

        for item_B in items_B:
            L_uid_itemB.append([uid, item_B])  # 第2个list存的是uid, item_B[1]
            L_itemB_uid.append([item_B, uid])  # 第4个list存的是item_B, uid[3]

        for item_index_A in range(0, len(items_A) - 1):
            item_temp_A = items_A[item_index_A]
            next_item_A = items_A[item_index_A + 1]
            L_neighbor_item_A.append([item_temp_A, item_temp_A])
            L_neighbor_item_A.append([item_temp_A, next_item_A])

        for item_index_B in range(0, len(items_B) - 1):
            item_temp_B = items_B[item_index_B]
            next_item_B = items_B[item_index_B + 1]
            L_neighbor_item_B.append([item_temp_B, item_temp_B])
            L_neighbor_item_B.append([item_temp_B, next_item_B])

        L_uid_uid.append([uid, uid])  # [6]

    matrix_U_A = duplicate_matrix(L_uid_itemA)
    matrix_U_B = duplicate_matrix(L_uid_itemB)
    matrix_A_U = duplicate_matrix(L_itemA_uid)
    matrix_B_U = duplicate_matrix(L_itemB_uid)
    matrix_A_neighbor = duplicate_matrix(L_neighbor_item_A)
    matrix_B_neighbor = duplicate_matrix(L_neighbor_item_B)
    matrix_U_U = duplicate_matrix(L_uid_uid)

    output_ratings.append(np.array(matrix_U_A))  # [0]
    output_ratings.append(np.array(matrix_U_B))  # [1]
    output_ratings.append(np.array(matrix_A_U))  # [2]
    output_ratings.append(np.array(matrix_B_U))  # [3]
    output_ratings.append(np.array(matrix_A_neighbor))  # [4]
    output_ratings.append(np.array(matrix_B_neighbor))  # [5]
    output_ratings.append(np.array(matrix_U_U))  # [6]

    return output_ratings


def duplicate_matrix(matrix_in):
    frame_temp = pd.DataFrame(matrix_in, columns=['row', 'column'])
    frame_temp.duplicated()
    frame_temp.drop_duplicates(inplace=True)
    return frame_temp.values.tolist()


def matrix2inverse(array_in, row_pre, col_pre, len_all):  # np.matrix转换为sparse_matrix;
    matrix_rows = array_in[:, 0] + row_pre  # X[:,0]表示对一个二维数组train_data取所有行的第一列数据;是numpy中数组的一种写法，
    matrix_columns = array_in[:, 1] + col_pre  # X[:,1]就是取所有行的第2列数据；类型为 ndarray;
    matrix_value = [1.] * len(matrix_rows)  # 只对交互过的（user,item）赋值 1.0；类型为list;
    inverse_matrix = sp.coo_matrix((matrix_value, (matrix_rows, matrix_columns)),
                                   shape=(len_all, len_all))  # shape=(129955,129955); dtype=float64;
    return inverse_matrix


def get_laplace_list(ratings, dict_A, dict_B, dict_U):
    adj_mat_list = []  # 定义一个列表；

    num_items_A = len(dict_A)
    num_items_B = len(dict_B)
    num_users = len(dict_U)
    num_all = num_items_A + num_users + num_items_B
    print("The dimension of all matrix is: {}".format(num_all))

    # 2021-11-18 重新按照在大矩阵中的位置给他们排一下序：A-U-B-tA-tB
    # 1: [item_A, next_item_A] + [item_A, item_A]
    inverse_matrix_A_A = matrix2inverse(ratings[4], row_pre=0, col_pre=0, len_all=num_all)
    # 2: [item_A, uid]
    inverse_matrix_A_U = matrix2inverse(ratings[2], row_pre=0, col_pre=num_items_A, len_all=num_all)
    # 3: [uid, item_A]
    inverse_matrix_U_A = matrix2inverse(ratings[0], row_pre=num_items_A, col_pre=0, len_all=num_all)
    # 4: [uid, uid]
    inverse_matrix_U_U = matrix2inverse(ratings[6], row_pre=num_items_A, col_pre=num_items_A,
                                        len_all=num_all)
    # 5: [uid, item_B]
    inverse_matrix_U_B = matrix2inverse(ratings[1], row_pre=num_items_A, col_pre=num_items_A + num_users,
                                        len_all=num_all)
    # 6: [item_B, uid]
    inverse_matrix_B_U = matrix2inverse(ratings[3], row_pre=num_items_A + num_users, col_pre=num_items_A,
                                        len_all=num_all)
    # 7: [item_B, next_item_B] + [item_B, item_B]
    inverse_matrix_B_B = matrix2inverse(ratings[5], row_pre=num_items_A + num_users,
                                        col_pre=num_items_A + num_users, len_all=num_all)

    print('Already convert the rating matrix into adjusted matrix.')
    adj_mat_list.append(inverse_matrix_U_A)
    adj_mat_list.append(inverse_matrix_U_B)
    adj_mat_list.append(inverse_matrix_A_U)
    adj_mat_list.append(inverse_matrix_B_U)
    adj_mat_list.append(inverse_matrix_A_A)
    adj_mat_list.append(inverse_matrix_B_B)
    adj_mat_list.append(inverse_matrix_U_U)

    # 将矩阵转为坐标格式，再存进list里
    laplace_list = [adj.tocoo() for adj in adj_mat_list]  # 拉普拉斯矩阵；

    return laplace_list


def get_batches(input_data, batch_size, padding_num_A, padding_num_B, isTrain):
    uid_all, seq_A_list, seq_B_list, len_A_list, len_B_list, pos_A_list, pos_B_list, \
    target_A_list, target_B_list = [], [], [], [], [], [], [], [], []
    num_batches = int(len(input_data) / batch_size)

    if isTrain is True:
        random.shuffle(input_data)

    for batch_index in range(num_batches):
        start_index = batch_index * batch_size
        batch = input_data[start_index:start_index + batch_size]
        uid, seq_A, seq_B, len_A, len_B, pos_A, pos_B, target_A, target_B = \
            batch_to_input(batch=batch, padding_num_A=padding_num_A, padding_num_B=padding_num_B)
        uid_all.append(uid)
        seq_A_list.append(seq_A)
        seq_B_list.append(seq_B)

        len_A_list.append(len_A)
        len_B_list.append(len_B)

        pos_A_list.append(pos_A)
        pos_B_list.append(pos_B)

        target_A_list.append(target_A)
        target_B_list.append(target_B)

    return list((uid_all, seq_A_list, seq_B_list, len_A_list, len_B_list, pos_A_list, pos_B_list,
                 target_A_list, target_B_list, num_batches))


def batch_to_input(batch, padding_num_A, padding_num_B):
    uid, seq_A, seq_B, len_A, len_B, pos_A, pos_B, target_A, target_B = \
        [], [], [], [], [], [], [], [], []
    for data_index in batch:
        len_A.append(data_index[2])
        len_B.append(data_index[3])
    maxlen_A = max(len_A)
    maxlen_B = max(len_B)
    i = 0
    for data_index in range(len(batch)):
        uid.append(batch[data_index][0][0])
        seq_A.append(batch[data_index][0][1:] + [padding_num_A] * (maxlen_A - len_A[i]))
        seq_B.append(batch[data_index][1][1:] + [padding_num_B] * (maxlen_B - len_B[i]))
        pos_A.append(batch[data_index][4][0:] + [padding_num_A] * (maxlen_A - len_A[i]))
        pos_B.append(batch[data_index][5][0:] + [padding_num_B] * (maxlen_B - len_B[i]))
        target_A.append(batch[data_index][6])
        target_B.append(batch[data_index][7])
        i += 1

    return np.array(uid), np.array(seq_A), np.array(seq_B), np.array(len_A).reshape(len(len_A), 1), np.array(
        len_B).reshape(len(len_B), 1), np.array(pos_A), np.array(pos_B), np.array(target_A), np.array(target_B)
