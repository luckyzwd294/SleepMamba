import torch
import os


class Config(object):
    def __init__(self):
        self.device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
        self.num_fold = 10
        self.num_classes = 5
        self.num_epochs = 100               
        self.batch_size = 64
        self.learning_rate = 5e-5
        self.dropout = 0.5       
        
        self.num_head = 8
        self.num_encoder = 2               
        self.num_encoder_multi = 2     

        self.d_model = 256
        self.s_model = 256
        self.d_ff = 1024
        self.drate = 0.5

        self.shuffle = True
        self.d_state = 16
        self.window = 5
        self.patience = 10
        self.dense = 142

        # Patch-Attention部分的参数
        self.patch_eln = 16
        self.stride = 8

        self.path_EDF = '/data/SleepEDF-78/'
        self.path_labels = '/data/SleepEDF-78/labels'






