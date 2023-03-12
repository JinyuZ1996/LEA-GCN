# @Author: Jinyu Zhang
# @Time: 2022/8/24 14:27
# @E-mail: JinyuZ1996@outlook.com

# coding: utf-8

class Settings:
    def __init__(self):
        self.code_path = "https://github.com/JinyuZ1996/LEA-GCN/"

        '''
            block 1: the draw_figures parameters for model training
        '''
        self.lr_A = 2e-3
        self.lr_B = 4e-3
        self.keep_prob = 0.9
        self.dropout_rate = 0.1
        self.batch_size = 256
        self.epochs = 100   # 100 for DOUBAN 80 for AMAZON
        self.verbose = 10
        self.gpu_num = '0'

        '''
            block 2: the draw_figures parameters for LEA_Model.py
        '''
        self.embedding_size = 16
        self.n_fold = self.embedding_size
        self.layer_size = '['+str(self.embedding_size)+']'
        self.num_layers = 3
        self.padding_int = 0

        self.alpha = 0.2
        # FOR EA-SUM
        self.beta = 0.5
        # FOR EA-A
        self.num_heads = 2
        self.dim_coefficient = 2 * self.num_heads
        self.regular_rate_att = 1e-7
        self.l2_regular_rate = 1e-7

        '''
            block 3: the draw_figures parameters for file paths
        '''
        self.dataset = 'Douban'  # Douban or Amazon
        self.path_train = 'data/' + self.dataset + '/train_data.txt'
        self.path_test = 'data/' + self.dataset + '/test_data.txt'
        self.path_dict_A = 'data/' + self.dataset + '/A_dict.txt'
        self.path_dict_B = 'data/' + self.dataset + '/B_dict.txt'
        self.path_dict_U = 'data/' + self.dataset + '/U_dict.txt'


        self.checkpoint = 'checkpoint/trained_model.ckpt'

        self.fast_running = False
        self.fast_ratio = 0.8