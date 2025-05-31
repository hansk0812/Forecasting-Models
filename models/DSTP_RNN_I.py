"""
Reference  https://github.com/Zhenye-Na/DA-RNN
"""
# -*- coding: utf-8 -*-

from utils import *
from torch.autograd import Variable
import torch
from torch import cuda
torch.cuda.is_available()
import numpy as np
from torch import nn
from torch import optim
import torch.nn.functional as F
import pandas as pd
import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt

def count_values(truth,pred):
    count_avg = 0
    assert len(truth)==len(pred)
    for x in range(len(truth)):
        count_avg+=abs(truth[x]-pred[x])
    return count_avg/len(truth)


class Encoder(nn.Module):


    def __init__(self, T ,
                 input_size,
                 encoder_num_hidden):
        """Initialize an encoder in DA_RNN."""
        super(Encoder, self).__init__()
        self.encoder_num_hidden = encoder_num_hidden
        self.input_size = input_size
        self.T = T

        self.encoder_lstm = nn.LSTM(
            input_size=self.input_size, 
            hidden_size=self.encoder_num_hidden)

        self.encoder_lstm2 = nn.LSTM(
            input_size=self.input_size, 
            hidden_size=self.encoder_num_hidden)

        # Construct Input Attention Mechanism via deterministic attention model
        # Eq2: W_f[h_{t-1}; s_{t-1}] + U_f * x^k + b, init Attn I
        self.encoder_attn = nn.Linear(
            in_features=2 * self.encoder_num_hidden + self.T - 1, 
            out_features=1, bias=True)

        # W_s[h_{t-1} ; s_{t-1}] + U_s[x^k ; y^k], init Attn II (phase II attn)
        self.encoder_attnII = nn.Linear(
            in_features=2 * self.encoder_num_hidden + 2*self.T - 2, 
            out_features=1, bias=True)

    def forward(self, X):
        """forward.

        Args:
            X

        """
        X_tilde = Variable(X.data.new(
            X.size(0), self.T - 1, self.input_size).zero_())
        X_encoded = Variable(X.data.new(
            X.size(0), self.T - 1, self.encoder_num_hidden).zero_())

        X_tildeII = Variable(X.data.new(
            X.size(0), self.T - 1, self.input_size).zero_())
        X_encodedII = Variable(X.data.new(
            X.size(0), self.T - 1, self.encoder_num_hidden).zero_())


        # hidden, cell: initial states with dimention hidden_size

        h_n = self._init_states(X)
        s_n = self._init_states(X)

        hs_n = self._init_states(X)
        ss_n = self._init_states(X)

        for t in range(self.T - 1):
            #Phase one attention
            # batch_size * input_size * (2 * hidden_size + T - 1)
            
            # torch.Size([1300, 720, 1]) torch.Size([1300, 720, 1]) torch.Size([1300, 720, 1])
            print (h_n.shape, s_n.shape, X.shape)
            
            x = torch.cat((h_n.repeat(self.input_size, 1, 1).permute(1, 0, 2), 
                           s_n.repeat(self.input_size, 1, 1).permute(1, 0, 2),
                           X.permute(1, 0, 2)), dim=2)
            
            print (x.shape)
            x = self.encoder_attn(
                x.view(-1, self.encoder_num_hidden * 2 + self.T - 1))

            # get weights by softmax
            alpha = F.softmax(x.view(-1, self.input_size))

            # get new input for LSTM
            x_tilde = torch.mul(alpha, X[:, t, :]) #233x363

            self.encoder_lstm.flatten_parameters()
            
            # encoder LSTM
            _, final_state = self.encoder_lstm(x_tilde.unsqueeze(0), (h_n, s_n))
            h_n = final_state[0]
            s_n = final_state[1]


            #Phase II attention

            x2 = torch.cat((hs_n.repeat(self.input_size, 1, 1).permute(1, 0, 2), #233 363 1042
                           ss_n.repeat(self.input_size, 1, 1).permute(1, 0, 2),
                           X.permute(0, 2, 1)), dim=2)

            x2 = self.encoder_attnII(
                x2.view(-1, self.encoder_num_hidden * 2 + 2*self.T - 2))

            alpha2 = F.softmax(x2.view(-1, self.input_size))# 233x363

            X_tildeII = torch.mul(alpha2, x_tilde)


            self.encoder_lstm2.flatten_parameters()
            _, final_state2 = self.encoder_lstm2(
                X_tildeII.unsqueeze(0), (hs_n, ss_n))
            hs_n = final_state2[0]
            ss_n = final_state2[1]
            X_tildeII[:, t, :] = X_tildeII
            X_encodedII[:, t, :] = hs_n

        return X_tildeII , X_encodedII

    def _init_states(self, X):
        """Initialize all 0 hidden states and cell states for encoder.

        Args:
            X
        Returns:
            initial_hidden_states

        """
        # hidden state and cell state [num_layers*num_directions, batch_size, hidden_size]
        # https://pytorch.org/docs/master/nn.html?#lstm
        return torch.zeros_like(X)


class Decoder(nn.Module):


    def __init__(self, T, decoder_num_hidden, encoder_num_hidden):
        super(Decoder, self).__init__()
        self.decoder_num_hidden = decoder_num_hidden
        self.encoder_num_hidden = encoder_num_hidden
        self.T = T

        self.attn_layer = nn.Sequential(
            nn.Linear(2 * decoder_num_hidden + encoder_num_hidden, encoder_num_hidden),
            nn.Tanh(),
            nn.Linear(encoder_num_hidden, 1)
        )
        self.lstm_layer = nn.LSTM(
            input_size=1, 
            hidden_size=decoder_num_hidden
        )
        self.fc = nn.Linear(encoder_num_hidden + 1, 1)
        self.fc_final_price = nn.Linear(decoder_num_hidden + encoder_num_hidden, 1)

        self.fc.weight.data.normal_()

    def forward(self, X_encoed):
        """forward."""
        d_n = self._init_states(X_encoed)
        c_n = self._init_states(X_encoed)

        for t in range(self.T - 1):

            x = torch.cat((d_n.repeat(self.T - 1, 1, 1).permute(1, 0, 2),
                           c_n.repeat(self.T - 1, 1, 1).permute(1, 0, 2),
                           X_encoed), dim=2)

            beta = F.softmax(self.attn_layer(
                x.view(-1, 2 * self.decoder_num_hidden + self.encoder_num_hidden)).view(-1, self.T - 1))
            
            # Eqn. 14: compute context vector
            # batch_size * encoder_hidden_size
            context = torch.bmm(beta.unsqueeze(1), X_encoed)[:, 0, :]
            if t < self.T - 1:
                # Eqn. 15
                # batch_size * 1
                y_tilde = self.fc(
                    torch.cat((context, X_encoed[:, t].unsqueeze(1)), dim=1))
                
                # Eqn. 16: LSTM
                self.lstm_layer.flatten_parameters()
                _, final_states = self.lstm_layer(
                    y_tilde.unsqueeze(0), (d_n, c_n))
                d_n = final_states[0]
                c_n = final_states[1]
                
        # Eqn. 22: final output
        final_temp_y = torch.cat((d_n[0], context), dim=1)
        y_pred_price = self.fc_final_price(final_temp_y)

        return y_pred_price
    def _init_states(self, X):
        """Initialize all 0 hidden states and cell states for encoder.

        Args:
            X
        Returns:
            initial_hidden_states

        """
        # hidden state and cell state [num_layers*num_directions, batch_size, hidden_size]
        # https://pytorch.org/docs/master/nn.html?#lstm
        initial_states = X.data.new(
            1, X.size(0), self.decoder_num_hidden).zero_()
        return initial_states


class Model(nn.Module):
    """da_rnn."""

    def __init__(self, config):
        
        super(Model, self).__init__()
        self.encoder_num_hidden = config.d_model
        self.decoder_num_hidden = config.d_model
        self.learning_rate = config.learning_rate
        self.batch_size = config.batch_size
        self.shuffle = False
        self.epochs = config.itr
        self.T = 10

        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        print("==> Use accelerator: ", self.device)
        
        self.Encoder = Encoder(input_size=config.enc_in,
                               encoder_num_hidden=self.encoder_num_hidden,
                               T=self.T).to(self.device)
        self.Decoder = Decoder(encoder_num_hidden=self.encoder_num_hidden,
                               decoder_num_hidden=self.decoder_num_hidden,
                               T=self.T).to(self.device)
        
    def forward(self, X, V1, V2, V3):
        """
        Forward pass.
        """
        
        input_encoded = self.Encoder(X)
        y_pred_price = self.Decoder(input_encoded)

        return y_pred_price

if __name__ == "__main__":

    X, y= read_NDX('nasdaq100_padding.csv', debug=False)

    model = DSTP_rnn(X, y, 128, 128, 128, 0.001, 50)
    model.train()
    torch.save(model.state_dict(), 'dstprnn_model.pkl')
    y_pred = model.test()

    fig1 = plt.figure()
    plt.semilogy(range(len(model.iter_losses)), model.iter_losses)
    plt.savefig("1.png")
    plt.close(fig1)

    fig2 = plt.figure()
    plt.semilogy(range(len(model.epoch_losses)), model.epoch_losses)
    plt.savefig("2.png")
    plt.close(fig2)

    fig3 = plt.figure()
    plt.plot(y_pred, label='Predicted')
    plt.plot(model.y[model.train_timesteps:], label="True")
    plt.legend(loc='upper left')
    plt.savefig("3.png")
    plt.close(fig3)
    print('Finished Training')

