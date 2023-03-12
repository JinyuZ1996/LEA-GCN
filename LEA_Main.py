# @Author: Jinyu Zhang
# @Time: 2022/8/24 8:52
# @E-mail: JinyuZ1996@outlook.com

# coding: utf-8

from time import time
from LEA_Config import *
from LEA_Printer import *
from LEA_Train import *

np.seterr(all='ignore')
args = Settings()

if __name__ == '__main__':

    print_config()

    print("Loading dictionary for data generation...")
    dict_A = build_dict(dict_path=args.path_dict_A)
    dict_B = build_dict(dict_path=args.path_dict_B)
    dict_U = build_dict(dict_path=args.path_dict_U)
    num_items_A = len(dict_A)
    num_items_B = len(dict_B)
    num_users = len(dict_U)

    print("Loading the mixed data from datasets...")
    mixed_seq_train = build_mixed_sequences(path=args.path_train, dict_A=dict_A, dict_B=dict_B, dict_U=dict_U)
    mixed_seq_test = build_mixed_sequences(path=args.path_test, dict_A=dict_A, dict_B=dict_B, dict_U=dict_U)

    if args.fast_running is True:
        print("This is a fast running task:")
        len_train = len(mixed_seq_train)
        mixed_train = mixed_seq_train[:int(len_train * args.fast_ratio)]

    print("Transforming the data...")
    train_data = data_generation(mixed_sequence=mixed_seq_train, dict_A=dict_A)
    test_data = data_generation(mixed_sequence=mixed_seq_test, dict_A=dict_A)

    output_ratings = get_rating_matrix(train_data)
    print("Already load the ratings.")
    laplace_list = get_laplace_list(output_ratings, dict_A, dict_B, dict_U)
    print("Already load the adj_list.")
    matrix_form_graph = sum(laplace_list)

    print("Already finished the process of all_data.")

    train_batches = get_batches(input_data=train_data, batch_size=args.batch_size, padding_num_A=args.padding_int,
                                padding_num_B=args.padding_int, isTrain=True)
    test_batches = get_batches(input_data=test_data, batch_size=args.batch_size, padding_num_A=args.padding_int,
                               padding_num_B=args.padding_int, isTrain=False)
    print("Already load the batches.")

    n_items_A = len(dict_A)
    n_items_B = len(dict_B)
    n_users = len(dict_U)

    model = LEA_GCN(n_items_A=n_items_A, n_items_B=n_items_B, n_users=n_users, graph_matrix=matrix_form_graph)
    print("Already initialize the Light-CSR.")
    with tf.Session(graph=model.graph, config=model.config) as sess:

        model.sess = sess
        saver = tf.train.Saver()
        sess.run(tf.global_variables_initializer())
        # set the score of default best score for test
        best_score_domain_A = -1
        best_score_domain_B = -1

        for epoch in range(args.epochs):
            # training
            rec_pre_begin_time = time()
            rec_loss_A, rec_loss_B = train(sess=sess, GCN_net=model, batches_train=train_batches)
            rec_pre_time = time() - rec_pre_begin_time

            epoch_to_print = epoch + 1
            print_rec_message(epoch=epoch_to_print, rec_loss_A=rec_loss_A, rec_loss_B=rec_loss_B,
                              rec_pre_time=rec_pre_time)
            # The parameter verbose controls how many training sessions to evaluate once.
            if epoch_to_print % args.verbose == 0:
                rec_test_begin_time = time()
                [RC_5_A, RC_10_A, RC_20_A, MRR_5_A, MRR_10_A, MRR_20_A,
                 RC_5_B, RC_10_B, RC_20_B, MRR_5_B, MRR_10_B, MRR_20_B,
                 NDCG_5_A, NDCG_10_A, NDCG_20_A, NDCG_5_B, NDCG_10_B, NDCG_20_B] = \
                    evaluate_module(sess=sess, GCN_net=model, test_batches=test_batches, test_len=len(test_data))
                rec_test_time = time() - rec_test_begin_time

                print_recommender_train(epoch_to_print, RC_5_A, RC_10_A, RC_20_A, MRR_5_A, MRR_10_A, MRR_20_A, RC_5_B,
                                        RC_10_B, RC_20_B, MRR_5_B, MRR_10_B, MRR_20_B, NDCG_10_A, NDCG_10_B, rec_test_time)

                if RC_5_A >= best_score_domain_A or RC_5_B >= best_score_domain_B:
                    best_score_domain_A = RC_5_A
                    best_score_domain_B = RC_5_B
                    saver.save(sess, args.checkpoint, global_step=epoch_to_print, write_meta_graph=False)
                    print("Recommender performs better, saving current model....")
                    logging.info("Recommender performs better, saving current model....")

            train_batches = get_batches(input_data=train_data, batch_size=args.batch_size,
                                        padding_num_A=args.padding_int,
                                        padding_num_B=args.padding_int, isTrain=True)

        print("Recommender training finished.")
        logging.info("Recommender training finished.")

        print("All process finished.")
