import torch
import torch.nn as nn
from .pyraformer.Layers import EncoderLayer, Decoder, Predictor
from .pyraformer.Layers import Bottleneck_Construct, Conv_Construct, MaxPooling_Construct, AvgPooling_Construct
from .pyraformer.Layers import get_mask, get_subsequent_mask, refer_points, get_k_q, get_q_k
from .pyraformer.embed import DataEmbedding, CustomEmbedding


class Encoder(nn.Module):
    """ A encoder model with self attention mechanism. """

    def __init__(self, configs):
        super().__init__()

        self.d_model = configs.d_model
        self.model_type = "Pyraformer"
        self.window_size = configs.detail_freq_list
        self.truncate = configs.truncate
        if configs.decoder == 'attention':
            self.mask, self.all_size = get_mask(configs.seq_len, self.window_size, configs.d_ff, configs.device)
        else:
            self.mask, self.all_size = get_mask(configs.seq_len+1, configs.detail_freq_list, configs.d_ff, configs.device)
        self.decoder_type = configs.decoder
        if configs.decoder == 'FC':
            self.indexes = refer_points(self.all_size, configs.detail_freq_list, configs.device)

        self.layers = nn.ModuleList([
            EncoderLayer(self.d_model, configs.d_ff, configs.n_heads, configs.factor, configs.factor, dropout=configs.dropout, \
                normalize_before=False) for i in range(configs.e_layers)
            ])

        self.enc_embedding = DataEmbedding(configs.enc_in, configs.d_model, configs.dropout)

        self.conv_layers = Bottleneck_Construct(configs.d_model, configs.detail_freq_list, configs.factor)

    def forward(self, x_enc, x_mark_enc):

        seq_enc = self.enc_embedding(x_enc, x_mark_enc)

        mask = self.mask.repeat(len(seq_enc), 1, 1).to(x_enc.device)
        seq_enc = self.conv_layers(seq_enc)

        for i in range(len(self.layers)):
            seq_enc, _ = self.layers[i](seq_enc, mask)

        if self.decoder_type == 'FC':
            indexes = self.indexes.repeat(seq_enc.size(0), 1, 1, seq_enc.size(2)).to(seq_enc.device)
            indexes = indexes.view(seq_enc.size(0), -1, seq_enc.size(2))
            all_enc = torch.gather(seq_enc, 1, indexes)
            seq_enc = all_enc.view(seq_enc.size(0), self.all_size[0], -1)
        elif self.decoder_type == 'attention' and self.truncate:
            seq_enc = seq_enc[:, :self.all_size[0]]

        return seq_enc


class Model(nn.Module):
    """ A sequence to sequence model with attention mechanism. """

    def __init__(self, configs):
        super().__init__()

        self.predict_step = configs.pred_len
        self.d_model = configs.d_model
        self.input_size = configs.seq_len
        self.channels = configs.enc_in

        configs.truncate = False
        configs.decoder = "attention" # or "FC"
        configs.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        configs.detail_freq_list = [int(x) for x in configs.detail_freq[1:-1].split(',')]

        self.decoder_type = configs.decoder

        self.encoder = Encoder(configs)
        if configs.decoder == 'attention':
            mask = get_subsequent_mask(self.input_size, configs.detail_freq_list, self.predict_step, configs.truncate)
            self.decoder = Decoder(configs, mask)
            self.predictor = Predictor(self.d_model, self.channels)
        elif configs.decoder == 'FC':
            self.predictor = Predictor(4 * self.d_model, self.predict_step * self.channels)

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, v4=None, v5=None, v6=None): #, batch_x_mark, dec_inp, batch_y_mark):
        
        """
        Return the hidden representations and predictions.
        For a sequence (l_1, l_2, ..., l_N), we predict (l_2, ..., l_N, l_{N+1}).
        Input: event_type: batch*seq_len;
               event_time: batch*seq_len.
        Output: enc_output: batch*seq_len*model_dim;
                type_prediction: batch*seq_len*num_classes (not normalized);
                time_prediction: batch*seq_len.
        """
        if self.decoder_type == 'attention':
            enc_output = self.encoder(x_enc, x_mark_enc)
            dec_enc = self.decoder(x_dec, x_mark_dec, enc_output)

            #if pretrain:
            #    dec_enc = torch.cat([enc_output[:, :self.input_size], dec_enc], dim=1)
            #    pred = self.predictor(dec_enc)
            #else:
            pred = self.predictor(dec_enc)
        elif self.decoder_type == 'FC':
            enc_output = self.encoder(x_enc, x_mark_enc)[:, -1, :]
            pred = self.predictor(enc_output).view(enc_output.size(0), self.predict_step, -1)

        return pred
