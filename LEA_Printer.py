# @Author: Jinyu Zhang
# @Time: 2022/8/24 15:22
# @E-mail: JinyuZ1996@outlook.com

# coding: utf-8

import logging
from LEA_Setting import *

args = Settings()


def print_config():
    print("Model Configs: \n"
          "Fast-Running: {}, lr_A: {}, lr_B: {}, epochs: {}, embedding-size: {}, \n"
          "Alpha: {}, Beta: {}, num_heads: {}, dim_coefficient: {}. ".format
          (args.fast_running, args.lr_A, args.lr_B, args.epochs, args.embedding_size,
           args.alpha, args.beta, args.num_heads, args.dim_coefficient))


def print_rec_message(epoch, rec_loss_A, rec_loss_B, rec_pre_time):
    print('Epoch {} - Training Loss A: {:.5f} Training Loss B: {:.5f} - Training time: {:.3}'.
          format(epoch, rec_loss_A, rec_loss_B, rec_pre_time))
    logging.info('Epoch {} - Training Loss A: {:.5f} Training Loss B: {:.5f} - Training time: {:.3}'.
                 format(epoch, rec_loss_A, rec_loss_B, rec_pre_time))


def print_recommender_train(epoch, RC_5_A, RC_10_A, RC_20_A, MRR_5_A, MRR_10_A, MRR_20_A, RC_5_B, RC_10_B,
                            RC_20_B, MRR_5_B, MRR_10_B, MRR_20_B, NDCG_10_A, NDCG_10_B, rec_test_time):
    # print(
    #     "Evaluate on Domain-A, Epoch %d : RC5 = %.4f, RC10 = %.4f, RC20 = %.4f, MRR5 = %.4f, MRR10 = %.4f, MRR20 = %.4f" % (
    #         epoch, RC_5_A, RC_10_A, RC_20_A, MRR_5_A, MRR_10_A, MRR_20_A))
    # print(
    #     "Evaluate on Domain-B, Epoch %d : RC5 = %.4f, RC10 = %.4f, RC20 = %.4f, MRR5 = %.4f, MRR10 = %.4f, MRR20 = %.4f" % (
    #         epoch, RC_5_B, RC_10_B, RC_20_B, MRR_5_B, MRR_10_B, MRR_20_B))
    # logging.info(
    #     "Evaluate on Domain-A, Epoch %d : RC5 = %.4f, RC10 = %.4f, RC20 = %.4f, MRR5 = %.4f, MRR10 = %.4f, MRR20 = %.4f" % (
    #         epoch, RC_5_A, RC_10_A, RC_20_A, MRR_5_A, MRR_10_A, MRR_20_A))
    # logging.info(
    #     "Evaluate on Domain-B, Epoch %d : RC5 = %.4f, RC10 = %.4f, RC20 = %.4f, MRR5 = %.4f, MRR10 = %.4f, MRR20 = %.4f" % (
    #         epoch, RC_5_B, RC_10_B, RC_20_B, MRR_5_B, MRR_10_B, MRR_20_B))
    # print("Epoch: {}, Recommender evaluate time: {:.3f}".format(epoch, rec_test_time))
    # logging.info("Epoch: {}, Recommender evaluate time: {:.3f}".format(epoch, rec_test_time))
    print(
        "Evaluate on Domain-A, Epoch %d : RC10 = %.4f, MRR10 = %.4f, NDCG10 = %.4f" % (
            epoch, RC_10_A, MRR_10_A, NDCG_10_A))
    print(
        "Evaluate on Domain-B, Epoch %d : RC10 = %.4f, MRR10 = %.4f, NDCG10 = %.4f" % (
            epoch, RC_10_B, MRR_10_B, NDCG_10_B))
    print("Epoch: {}, Recommender evaluate time: {:.3f}".format(epoch, rec_test_time))
