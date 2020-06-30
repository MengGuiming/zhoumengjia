import config
# import models
import modelsnew as models
import numpy as np
import os
import time
import datetime
import json
from sklearn.metrics import average_precision_score
import sys
import os
import argparse
# import IPython

# sys.excepthook = IPython.core.ultratb.FormattedTB(mode='Verbose', color_scheme='Linux', call_pdb=1)


parser = argparse.ArgumentParser()
parser.add_argument('--model_name', type = str, default = 'BertEvi', help = 'name of the model')
parser.add_argument('--save_name', type = str)

parser.add_argument('--train_prefix', type = str, default = 'train')
parser.add_argument('--test_prefix', type = str, default = 'dev_test')
parser.add_argument('--input_theta', type = float, default = -1)
parser.add_argument('--two_phase', action='store_true')
# parser.add_argument('--ignore_input_theta', type = float, default = -1)


args = parser.parse_args()
model = {
    'CNN3': models.CNN3,
    'LSTM': models.LSTM,
    'BiLSTM': models.BiLSTM,
    'ContextAware': models.ContextAware,
    'BertEvi':models.BertEvi,
    'BertEviOld':models.BertEviOld,
    'XLNetEVI':models.XLNetEVI,
    'BertGcn':models.BertGcn,
    'BertMTL':models.BertMTL,
    'BertMTL1':models.BertMTL1
    # 'LSTM_SP': models.LSTM_SP
}

configs = {
    'ConfigXLNetEVI':config.ConfigXLNetEVI,
    'ConfigBertEvi':config.ConfigBertEvi,
    'ConfigBertEviOld':config.ConfigBertEviOld,
    'ConfigBertGcn':config.ConfigBertGcn,
    'ConfigBertMTL':config.ConfigBertMTL,
    'ConfigBertMTL1':config.ConfigBertMTL1,
    'ConfigBiLSTM':config.ConfigBiLSTM
}

con = configs['Config'+args.model_name](args)
#con.load_train_data()
con.load_test_data()
# con.set_train_model()
pretrain_model_name = 'checkpoint_BiLSTM_bert_relation_exist_cls'
con.testall(model[args.model_name], args.save_name, args.input_theta, args.two_phase, pretrain_model_name)#, args.ignore_input_theta)
