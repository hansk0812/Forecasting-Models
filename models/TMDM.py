import torch
from torch import nn
from torch.nn import functional as F
from layers.Embed import DataEmbedding
import yaml
import argparse

class Model(nn.Module):
    """
    Vanilla Transformer
    """

    def __init__(self, configs):
        super(Model, self).__init__()

        with open(configs.diffusion_config_dir, "r") as f:
            config = yaml.unsafe_load(f)
            diffusion_config = dict2namespace(config)

        diffusion_config.diffusion.timesteps = configs.timesteps
        
        self.args = configs
        self.device = configs.device
        self.diffusion_config = diffusion_config

        self.model_var_type = diffusion_config.model.var_type
        self.num_timesteps = diffusion_config.diffusion.timesteps
        self.vis_step = diffusion_config.diffusion.vis_step
        self.num_figs = diffusion_config.diffusion.num_figs
        self.dataset_object = None

        betas = make_beta_schedule(schedule=diffusion_config.diffusion.beta_schedule, num_timesteps=self.num_timesteps,
                                   start=diffusion_config.diffusion.beta_start, end=diffusion_config.diffusion.beta_end)
        betas = self.betas = betas.float().to(self.device)
        self.betas_sqrt = torch.sqrt(betas)
        alphas = 1.0 - betas
        self.alphas = alphas
        self.one_minus_betas_sqrt = torch.sqrt(alphas)
        alphas_cumprod = alphas.to('cpu').cumprod(dim=0).to(self.device)
        self.alphas_bar_sqrt = torch.sqrt(alphas_cumprod)
        self.one_minus_alphas_bar_sqrt = torch.sqrt(1 - alphas_cumprod)
        if diffusion_config.diffusion.beta_schedule == "cosine":
            self.one_minus_alphas_bar_sqrt *= 0.9999  # avoid division by 0 for 1/sqrt(alpha_bar_t) during inference
        alphas_cumprod_prev = torch.cat(
            [torch.ones(1, device=self.device), alphas_cumprod[:-1]], dim=0
        )
        self.alphas_cumprod_prev = alphas_cumprod_prev
        self.posterior_mean_coeff_1 = (
                betas * torch.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        )
        self.posterior_mean_coeff_2 = (
                torch.sqrt(alphas) * (1 - alphas_cumprod_prev) / (1 - alphas_cumprod)
        )
        posterior_variance = (
                betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        )
        self.posterior_variance = posterior_variance
        if self.model_var_type == "fixedlarge":
            self.logvar = betas.log()
            # torch.cat(
            # [posterior_variance[1:2], betas[1:]], dim=0).log()
        elif self.model_var_type == "fixedsmall":
            self.logvar = posterior_variance.clamp(min=1e-20).log()

        self.tau = None  # precision fo test NLL computation

        # CATE MLP
        self.diffussion_model = ConditionalGuidedModel(diffusion_config, self.args)

        self.enc_embedding = DataEmbedding(configs.enc_in, configs.CART_input_x_embed_dim, configs.embed, configs.freq,
                                           configs.dropout)

        a = 0

    def forward(self, x, x_mark, y, y_t, y_0_hat, t):
        enc_out = self.enc_embedding(x, x_mark)
        dec_out = self.diffussion_model(enc_out, y_t, y_0_hat, t)

        return dec_out

def dict2namespace(config):
    namespace = argparse.Namespace()
    for key, value in config.items():
        if isinstance(value, dict):
            new_value = dict2namespace(value)
        else:
            new_value = value
        setattr(namespace, key, new_value)
    return namespace

def make_beta_schedule(schedule="linear", num_timesteps=1000, start=1e-5, end=1e-2):
    if schedule == "linear":
        betas = torch.linspace(start, end, num_timesteps)
    elif schedule == "const":
        betas = end * torch.ones(num_timesteps)
    elif schedule == "quad":
        betas = torch.linspace(start ** 0.5, end ** 0.5, num_timesteps) ** 2
    elif schedule == "jsd":
        betas = 1.0 / torch.linspace(num_timesteps, 1, num_timesteps)
    elif schedule == "sigmoid":
        betas = torch.linspace(-6, 6, num_timesteps)
        betas = torch.sigmoid(betas) * (end - start) + start
    elif schedule == "cosine" or schedule == "cosine_reverse":
        max_beta = 0.999
        cosine_s = 0.008
        betas = torch.tensor(
            [min(1 - (math.cos(((i + 1) / num_timesteps + cosine_s) / (1 + cosine_s) * math.pi / 2) ** 2) / (
                    math.cos((i / num_timesteps + cosine_s) / (1 + cosine_s) * math.pi / 2) ** 2), max_beta) for i in
             range(num_timesteps)])
        if schedule == "cosine_reverse":
            betas = betas.flip(0)  # starts at max_beta then decreases fast
    elif schedule == "cosine_anneal":
        betas = torch.tensor(
            [start + 0.5 * (end - start) * (1 - math.cos(t / (num_timesteps - 1) * math.pi)) for t in
             range(num_timesteps)])
    return betas

class ConditionalLinear(nn.Module):
    def __init__(self, num_in, num_out, n_steps):
        super(ConditionalLinear, self).__init__()
        self.num_out = num_out
        self.lin = nn.Linear(num_in, num_out)
        self.embed = nn.Embedding(n_steps, num_out)
        self.embed.weight.data.uniform_()

    def forward(self, x, t):
        out = self.lin(x)
        gamma = self.embed(t)
        # out = gamma.view(-1, self.num_out) * out

        out = gamma.view(t.size()[0], -1, self.num_out) * out
        return out


class ConditionalGuidedModel(nn.Module):
    def __init__(self, config, MTS_args):
        super(ConditionalGuidedModel, self).__init__()
        n_steps = config.diffusion.timesteps + 1
        self.cat_x = config.model.cat_x
        self.cat_y_pred = config.model.cat_y_pred
        data_dim = 14

        self.lin1 = ConditionalLinear(data_dim, 128, n_steps)
        self.lin2 = ConditionalLinear(128, 128, n_steps)
        self.lin3 = ConditionalLinear(128, 128, n_steps)
        self.lin4 = nn.Linear(128, 7)

    def forward(self, x, y_t, y_0_hat, t):
        if self.cat_x:
            if self.cat_y_pred:
                eps_pred = torch.cat((y_t, y_0_hat), dim=-1)
            else:
                eps_pred = torch.cat((y_t, x), dim=2)
        else:
            if self.cat_y_pred:
                eps_pred = torch.cat((y_t, y_0_hat), dim=2)
            else:
                eps_pred = y_t
        if y_t.device.type == 'mps':
            eps_pred = self.lin1(eps_pred, t)
            eps_pred = F.softplus(eps_pred.cpu()).to(y_t.device)

            eps_pred = self.lin2(eps_pred, t)
            eps_pred = F.softplus(eps_pred.cpu()).to(y_t.device)

            eps_pred = self.lin3(eps_pred, t)
            eps_pred = F.softplus(eps_pred.cpu()).to(y_t.device)

        else:
            eps_pred = F.softplus(self.lin1(eps_pred, t))
            eps_pred = F.softplus(self.lin2(eps_pred, t))
            eps_pred = F.softplus(self.lin3(eps_pred, t))
        eps_pred = self.lin4(eps_pred)
        return eps_pred

def extract(input, t, x):
    shape = x.shape
    out = torch.gather(input, 0, t.to(input.device))
    reshape = [t.shape[0]] + [1] * (len(shape) - 1)
    return out.reshape(*reshape)

# Forward functions
def q_sample(y, y_0_hat, alphas_bar_sqrt, one_minus_alphas_bar_sqrt, t, noise=None):
    """
    y_0_hat: prediction of pre-trained guidance model; can be extended to represent
        any prior mean setting at timestep T.
    """
    if noise is None:
        noise = torch.randn_like(y).to(y.device)
    sqrt_alpha_bar_t = extract(alphas_bar_sqrt, t, y)
    sqrt_one_minus_alpha_bar_t = extract(one_minus_alphas_bar_sqrt, t, y)
    # q(y_t | y_0, x)
    y_t = sqrt_alpha_bar_t * y + (1 - sqrt_alpha_bar_t) * y_0_hat + sqrt_one_minus_alpha_bar_t * noise
    return y_t


# Reverse function -- sample y_{t-1} given y_t
def p_sample(model, x, x_mark, y, y_0_hat, y_T_mean, t, alphas, one_minus_alphas_bar_sqrt):
    """
    Reverse diffusion process sampling -- one time step.

    y: sampled y at time step t, y_t.
    y_0_hat: prediction of pre-trained guidance model.
    y_T_mean: mean of prior distribution at timestep T.
    We replace y_0_hat with y_T_mean in the forward process posterior mean computation, emphasizing that 
        guidance model prediction y_0_hat = f_phi(x) is part of the input to eps_theta network, while 
        in paper we also choose to set the prior mean at timestep T y_T_mean = f_phi(x).
    """
    device = next(model.parameters()).device
    z = torch.randn_like(y)  # if t > 1 else torch.zeros_like(y)
    t = torch.tensor([t]).to(device)
    alpha_t = extract(alphas, t, y)
    sqrt_one_minus_alpha_bar_t = extract(one_minus_alphas_bar_sqrt, t, y)
    sqrt_one_minus_alpha_bar_t_m_1 = extract(one_minus_alphas_bar_sqrt, t - 1, y)
    sqrt_alpha_bar_t = (1 - sqrt_one_minus_alpha_bar_t.square()).sqrt()
    sqrt_alpha_bar_t_m_1 = (1 - sqrt_one_minus_alpha_bar_t_m_1.square()).sqrt()
    # y_t_m_1 posterior mean component coefficients
    gamma_0 = (1 - alpha_t) * sqrt_alpha_bar_t_m_1 / (sqrt_one_minus_alpha_bar_t.square())
    gamma_1 = (sqrt_one_minus_alpha_bar_t_m_1.square()) * (alpha_t.sqrt()) / (sqrt_one_minus_alpha_bar_t.square())
    gamma_2 = 1 + (sqrt_alpha_bar_t - 1) * (alpha_t.sqrt() + sqrt_alpha_bar_t_m_1) / (
        sqrt_one_minus_alpha_bar_t.square())
    eps_theta = model(x, x_mark, 0, y, y_0_hat, t).to(device).detach()
    # y_0 reparameterization
    y_0_reparam = 1 / sqrt_alpha_bar_t * (
            y - (1 - sqrt_alpha_bar_t) * y_T_mean - eps_theta * sqrt_one_minus_alpha_bar_t)
    # posterior mean
    y_t_m_1_hat = gamma_0 * y_0_reparam + gamma_1 * y + gamma_2 * y_T_mean
    # posterior variance
    beta_t_hat = (sqrt_one_minus_alpha_bar_t_m_1.square()) / (sqrt_one_minus_alpha_bar_t.square()) * (1 - alpha_t)
    y_t_m_1 = y_t_m_1_hat.to(device) + beta_t_hat.sqrt().to(device) * z.to(device)
    return y_t_m_1


if __name__ == "__main__":

    class Config:
        enc_in = 7
        dec_in = 7
        c_out = 7
        e_layers = 1
        d_layers = 1
        d_model = 256
        factor = 64
        seq_len = 720
        label_len = 360
        pred_len = 720
        
        activation = "gelu"
        d_ff = 2048
        n_heads = 4
        moving_avg = [24]
        output_attention = False
        embed = "timeF"
        freq = 'm'
        dropout = 0.2
        
        k_cond = 1
        k_z = 1e-2
        p_hidden_layers = 2
        p_hidden_dims = [64, 64]
        timesteps = 1000
        diffusion_config_dir = "utils/TMDM_config.yml"
        CART_input_x_embed_dim = 32

        device = torch.device('cpu')

    args = Config()
    
    from exp.exp_main import log_normal
    from models.diffusion import Autoformer as ns_Autoformer
    from models.diffusion import Transformer as ns_Transformer
    import numpy as np

    model = Model(args)
    #cond_pred_model = ns_Autoformer.Model(args).float()
    #cond_pred_model_train = ns_Autoformer.Model(args).float()
    cond_pred_model = ns_Transformer.Model(args).float()
    cond_pred_model_train = ns_Transformer.Model(args).float()

    batch_x = torch.ones((1,720,7))
    batch_x_mark = torch.ones((1,720,1))
    dec_inp = torch.ones((1,args.pred_len+args.label_len,7))
    batch_y = torch.ones((1,args.pred_len+args.label_len,7))
    batch_y_mark = torch.ones((1,args.pred_len+args.label_len,1))
    #print (model(batch_x).shape)
    
    n = batch_x.size(0)
    t = torch.randint(
        low=0, high=args.timesteps, size=(n // 2 + 1,)
    ).to(args.device)
    t = torch.cat([t, args.timesteps - 1 - t], dim=0)[:n]

    _, y_0_hat_batch, KL_loss, z_sample = cond_pred_model(batch_x, batch_x_mark, dec_inp,
                                                         batch_y_mark)

    loss_vae = log_normal(batch_y, y_0_hat_batch, torch.from_numpy(np.array(1)))

    loss_vae_all = loss_vae + args.k_z * KL_loss
    # y_0_hat_batch = z_sample

    y_T_mean = y_0_hat_batch
    e = torch.randn_like(batch_y).to(args.device)

    y_t_batch = q_sample(batch_y, y_T_mean, model.alphas_bar_sqrt,
                         model.one_minus_alphas_bar_sqrt, t, noise=e)

    output = model(batch_x, batch_x_mark, batch_y, y_t_batch, y_0_hat_batch, t)

    # loss = (e[:, -self.args.pred_len:, :] - output[:, -self.args.pred_len:, :]).square().mean()
    loss = (e - output).square().mean() + args.k_cond*loss_vae_all
    
    print ('normal dist', output.shape, 'forecasts', y_0_hat_batch.shape)
    print ('loss', loss.item())

