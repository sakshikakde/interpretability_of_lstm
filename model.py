import torch
from torch import nn

from models import cnn_lstm_fc as cnnlstm
# from models import cnn_lstm_with_attention as cnnlstm
# from models import custom_feat_extractor_lstn_fc as cnnlstm
def generate_model(opt, device):
	assert opt.model in [
		'cnnlstm'
	]

	if opt.model == 'cnnlstm':
		model = cnnlstm.CNNLSTM(num_classes=opt.n_classes)
		# model = cnnlstm.CNNLSTM(num_channels = 3,
		# 						num_out_features = 300,
		# 						num_classes=opt.n_classes)
		# model = cnnlstm.CNNLSTM(lstm_input_sz = 300, lstm_hidden_sz = 256, r = 20, d_a = 50, num_classes=opt.n_classes)
	return model.to(device)