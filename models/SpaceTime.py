import torch
import torch.nn as nn
import einops

from .spacetime import PreprocessSSM, EncoderSSM, DecoderSSM

class Model(nn.Module):
    def __init__(self, config):
        super(Model, self).__init__()

        self.inference_only = not config.is_training

        # Embedding
        self.embedding_dim = config.d_model
        self.input_dim = config.enc_in
        
        # Preprocessing
        self.preprocess = PreprocessSSM(model_dim=config.d_model, kernel_dim=2, kernel_repeat=config.n_heads, 
                                        num_kernels=config.d_ff, kernel_init="normal", seed=43, head_dim=config.enc_in, 
                                        min_avg_window=4, max_avg_window=64)
        # Encoder SSM
        self.encoder = EncoderSSM(norm_order=1, model_dim=config.d_model, kernel_dim=config.d_ff, kernel_repeat=config.n_heads,
                                    num_kernels=config.d_ff, kernel_init="normal", num_heads=config.n_heads, head_dim=config.enc_in,
                                    skip_connection=True, seed=43)
        self.kernel_dim = self.encoder.kernel_dim
        # Encoder MLP
        self.encoder_mlp_layers = self.return_mlp_layers(config.e_layers, config.d_model, config.d_model, config.dropout) 
        # both encoder SSM and MLP layers' counts are 1

        # Decoder SSM
        self.decoder = DecoderSSM(lag=config.seq_len, horizon=config.pred_len, norm_order=1, model_dim=config.d_model, kernel_dim=config.factor, kernel_repeat=1,
                                    num_kernels=config.d_model, kernel_init="normal", num_heads=1, head_dim=1,
                                    skip_connection=True, seed=43)
        self.decoder.inference_only = not config.is_training
        # Decoder MLP
        self.output_layer = self.return_mlp_layers(config.d_layers, config.d_model, config.enc_in, config.dropout)
        # both decoder SSM and MLP layers' counts are 1

    def return_mlp_layers(self, num_layers, input_shape, output_shape, dropout):

        mlp_layers = [nn.GELU(), nn.Dropout(dropout)]
        for _ in range(num_layers):
            mlp_layers.append(nn.Linear(input_shape, output_shape))
            #mlp_layers.append(nn.Linear(output_shape, output_shape))
            mlp_layers.append(nn.GELU())
        
        return nn.Sequential(*mlp_layers)
    
    # -------------
    # Toggle things
    # -------------
    def set_inference_only(self, mode=False):
        self.inference_only = mode
        self.decoder.inference_only = mode
        
    def set_closed_loop(self, mode=True):
        self.decoder.closed_loop = mode
        
    def set_train(self):
        self.train()
        
    def set_eval(self):
        self.eval()
        self.set_inference_only(mode=True)
        
    def forward(self, x):
   
        self.set_closed_loop(True)
        # Assume u.shape is (batch x len x dim), 
        # where len = lag + horizon
        
        z = einops.repeat(x, 'b l d -> b l (r d)', 
                      r=self.embedding_dim // self.input_dim)
        
        z = self.preprocess(z)
        
        z = self.encoder(z)
        
        z = self.encoder_mlp_layers(z) + z

        y_c, _ = self.decoder(z)  
        
        y_c = self.output_layer(y_c)  # y_c is closed-loop output
        
        if not self.inference_only:  
            # Also compute outputs via open-loop
            self.set_closed_loop(False)
            y_o, z_u = self.decoder(z)
            y_o = self.output_layer(y_o)    # y_o is "open-loop" output
            # Prediction and "ground-truth" for next-time-step 
            # layer input (i.e., last-layer output)
            z_u_pred, z_u_true = z_u  
        else:
            y_o = None
            z_u_pred, z_u_true = None, None
        # Return (model outputs), (model last-layer next-step inputs)
        
        return (y_c, y_o), (z_u_pred, z_u_true)
