import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
from torch.nn import LayerNorm

import copy
import math
import utils
from encoder import make_encoder
from utils import PositionalEmbedding
LOG_FREQ = 10000


def gaussian_logprob(noise, log_std):
    """Compute Gaussian log probability."""
    residual = (-0.5 * noise.pow(2) - log_std).sum(-1, keepdim=True)
    return residual - 0.5 * np.log(2 * np.pi) * noise.size(-1)

def squash(mu, pi, log_pi):
    """Apply squashing function.
    See appendix C from https://arxiv.org/pdf/1812.05905.pdf.
    """
    mu = torch.tanh(mu)
    if pi is not None:
        pi = torch.tanh(pi)
    if log_pi is not None:
        log_pi -= torch.log(F.relu(1 - pi.pow(2)) + 1e-6).sum(-1, keepdim=True)
    return mu, pi, log_pi

def weight_init(m):
    """Custom weight init for Conv2D and Linear layers."""
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight.data)
        m.bias.data.fill_(0.0)
    elif isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        # delta-orthogonal init from https://arxiv.org/pdf/1806.05393.pdf
        assert m.weight.size(2) == m.weight.size(3)
        m.weight.data.fill_(0.0)
        m.bias.data.fill_(0.0)
        mid = m.weight.size(2) // 2
        gain = nn.init.calculate_gain('relu')
        nn.init.orthogonal_(m.weight.data[:, :, mid, mid], gain)


class Actor(nn.Module):

    def __init__(self, obs_shape, action_shape, hidden_dim, encoder_type,
                 encoder_feature_dim, log_std_min, log_std_max, num_layers, num_filters):
        super().__init__()
        self.encoder = make_encoder(
            encoder_type, obs_shape, encoder_feature_dim, num_layers,
            num_filters, output_logits=True
        )

        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        self.trunk = nn.Sequential(
            nn.Linear(self.encoder.feature_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, 2 * action_shape[0])
        )
        self.outputs = dict()
        self.apply(weight_init)

    def forward(
            self, obs, compute_pi=True, compute_log_pi=True, detach_encoder=False
    ):
        obs = self.encoder(obs, detach=detach_encoder)

        mu, log_std = self.trunk(obs).chunk(2, dim=-1)

        # constrain log_std inside [log_std_min, log_std_max]
        log_std = torch.tanh(log_std)
        log_std = self.log_std_min + 0.5 * (
                self.log_std_max - self.log_std_min
        ) * (log_std + 1)

        self.outputs['mu'] = mu
        self.outputs['std'] = log_std.exp()

        if compute_pi:
            std = log_std.exp()
            noise = torch.randn_like(mu)
            pi = mu + noise * std
        else:
            pi = None
            entropy = None

        if compute_log_pi:
            log_pi = gaussian_logprob(noise, log_std)
        else:
            log_pi = None

        mu, pi, log_pi = squash(mu, pi, log_pi)

        return mu, pi, log_pi, log_std

    def log(self, L, step, log_freq=LOG_FREQ):
        if step % log_freq != 0:
            return

        for k, v in self.outputs.items():
            L.log_histogram('train_actor/%s_hist' % k, v, step)

        L.log_param('train_actor/fc1', self.trunk[0], step)
        L.log_param('train_actor/fc2', self.trunk[2], step)
        L.log_param('train_actor/fc3', self.trunk[4], step)

class QFunction(nn.Module):

    def __init__(self, obs_dim, action_dim, hidden_dim):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(obs_dim + action_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, obs, action):
        assert obs.size(0) == action.size(0)
        obs_action = torch.cat([obs, action], dim=1)
        return self.trunk(obs_action)

class Critic(nn.Module):

    def __init__(
            self, obs_shape, action_shape, hidden_dim, encoder_type,
            encoder_feature_dim, num_layers, num_filters
    ):
        super().__init__()
        self.encoder = make_encoder(
            encoder_type, obs_shape, encoder_feature_dim, num_layers,
            num_filters, output_logits=True
        )
        self.Q1 = QFunction(
            self.encoder.feature_dim, action_shape[0], hidden_dim,
        )
        self.Q2 = QFunction(
            self.encoder.feature_dim, action_shape[0], hidden_dim,
        )
        self.outputs = dict()
        self.apply(weight_init)

    def forward(self, obs, action, detach_encoder=False):
        # detach_encoder allows to stop gradient propogation to encoder
        obs = self.encoder(obs, detach=detach_encoder)

        q1 = self.Q1(obs, action)
        q2 = self.Q2(obs, action)

        self.outputs['q1'] = q1
        self.outputs['q2'] = q2

        return q1, q2

    def log(self, L, step, log_freq=LOG_FREQ):
        if step % log_freq != 0:
            return

        self.encoder.log(L, step, log_freq)

        for k, v in self.outputs.items():
            L.log_histogram('train_critic/%s_hist' % k, v, step)

        for i in range(3):
            L.log_param('train_critic/q1_fc%d' % i, self.Q1.trunk[i * 2], step)
            L.log_param('train_critic/q2_fc%d' % i, self.Q2.trunk[i * 2], step)


class MultiheadAttention(nn.Module):

    def __init__(self, embed_dim, num_heads=1, dropout=0):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads

        self.head_dim = embed_dim // num_heads
        assert embed_dim % num_heads == 0
        self.scaling = self.head_dim ** -0.5
        self.in_proj = nn.Linear(embed_dim, 3 * embed_dim)
        self.dropout = dropout

    def forward(self, x):
        tgt_len, bsz, embed_dim = x.size()
        x = self.in_proj(x)
        q, k, v = x.chunk(3, dim=-1)
        q *= self.scaling

        q = q.contiguous().view(tgt_len, bsz * self.num_heads, self.head_dim).transpose(0, 1)
        k = k.contiguous().view(-1, bsz * self.num_heads, self.head_dim).transpose(0, 1)
        v = v.contiguous().view(-1, bsz * self.num_heads, self.head_dim).transpose(0, 1)

        # attn weight [bsz * num_heads, tgt_len, src_len]
        attn_weights = torch.bmm(q, k.transpose(1, 2))
        attn_weights = attn_weights - torch.max(attn_weights, dim=-1, keepdim=True)[0]
        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights = F.dropout(attn_weights, p=self.dropout, training=self.training)
        attn = torch.bmm(attn_weights, v)
        attn = attn.transpose(0, 1).contiguous().view(tgt_len, bsz, embed_dim)
        return attn

class AttnLayer(nn.Module):

    def __init__(self, embed_dim, normalize_before=False,
                 relu_dropout=0.,
                 attention_dropout=0.,
                 dropout=0.,
                 ):
        super().__init__()
        self.self_attn = MultiheadAttention(embed_dim, dropout=attention_dropout)
        self.attn_layer_norm = LayerNorm(embed_dim)
        self.fc1 = nn.Linear(embed_dim, 2 * embed_dim)
        self.fc2 = nn.Linear(2 * embed_dim, embed_dim)
        self.final_layer_norm = LayerNorm(embed_dim)
        self.normalize_before = normalize_before
        self.dropout = dropout
        self.relu_dropout = relu_dropout

    def forward(self, x):
        residual = x
        x = self.maybe_layer_norm(self.attn_layer_norm, x, before=True)
        x = self.self_attn(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = residual + x
        x = self.maybe_layer_norm(self.attn_layer_norm, x, after=True)

        residual = x
        x = self.maybe_layer_norm(self.final_layer_norm, x, before=True)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=self.relu_dropout, training=self.training)
        x = self.fc2(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = residual + x
        x = self.maybe_layer_norm(self.final_layer_norm, x, after=True)

        return x

    def maybe_layer_norm(self, layer_norm, x, before=False, after=False):
        assert before ^ after
        if after ^ self.normalize_before:
            return layer_norm(x)
        else:
            return x

class CTMR(nn.Module):

    def __init__(self, z_dim,  critic, critic_target,
                 output_type="continuous", num_attn_layer=2,
                 mtm_bsz=64, normalize_before=False,
                 relu_dropout=0.,
                 attention_dropout=0.,
                 dropout=0.,
                 ):
        super().__init__()
        self.mtm_bsz = mtm_bsz
        self.encoder = critic.encoder
        self.encoder_target = critic_target.encoder
        self.position = PositionalEmbedding(z_dim)
        self.attn_layers = nn.ModuleList([AttnLayer(z_dim, normalize_before=normalize_before,
                                                    dropout=dropout, attention_dropout=attention_dropout,
                                                    relu_dropout=relu_dropout)
                                          for _ in range(num_attn_layer)])

    def encode_obs(self, obs, detach=False, ema=False):
        x = obs.view(-1, *obs.shape[2:])
        if ema:
            with torch.no_grad():
                z_out = self.encoder_target(x)
        else:
            z_out = self.encoder(x)

        if detach:
            z_out = z_out.detach()
        return z_out.view(self.mtm_bsz, -1, *z_out.shape[1:])



    def forward(self, obs, mtm=False, ema=False, detach=False):
        obs = self.encode_obs(obs, detach=detach, ema=ema)
        if not mtm:
            return obs.reshape(-1, *obs.shape[2:])
        length = obs.shape[1]
        position = self.position(length)
        x = obs + position
        x = x.transpose(0, 1)
        for i in range(len(self.attn_layers)):
            x = self.attn_layers[i](x)
        x = x.transpose(0, 1)
        return x.reshape(-1, *obs.shape[2:])


    def compute_logits(self, z_a, z_pos, phrase=2):
        logits = torch.matmul(z_a, z_pos.T)
        logits = logits - torch.max(logits, 1, keepdim=True)[0]
        return logits


class InverseSquareRootSchedule(object):

    def __init__(self, warmup_step=4e4):
        if warmup_step is None:
            self.warmup_step = warmup_step
        else:
            warmup_step = int(warmup_step)
            assert warmup_step > 0 and isinstance(warmup_step, int)
            self.warmup_step = warmup_step
            init = 5e-4
            end = 1
            self.init_lr = init
            self.lr_step = (end - init) / warmup_step
            self.decay = warmup_step ** 0.5

    def step(self, step):
        if self.warmup_step is None:
            return  1
        else:
            if step < self.warmup_step:
                return self.init_lr + self.lr_step * step
            else:
                return self.decay * (step ** -0.5)


class AnneallingSchedule(object):
    def __init__(self, warmup_step=4e4):
        if warmup_step is None:
            self.warmup_step = warmup_step
        else:
            warmup_step = int(warmup_step)
            assert warmup_step > 0 and isinstance(warmup_step, int)
            self.warmup_step = warmup_step
            self.decay = warmup_step ** 0.5

    def step(self, step):
        if self.warmup_step is None:
            return  1
        else:
            if step < self.warmup_step:
                return 1
            else:
                return self.decay * (step ** -0.5)


class CtmrSacAgent(object):

    def __init__(self,
                 obs_shape,
                 action_shape,
                 device,
                 hidden_dim=256,
                 discount=0.99,
                 init_temperature=0.01,
                 alpha_lr=1e-3,
                 alpha_beta=0.9,
                 actor_lr=1e-3,
                 actor_beta=0.9,
                 actor_log_std_min=-10,
                 actor_log_std_max=2,
                 actor_update_freq=2,
                 critic_lr=1e-3,
                 critic_beta=0.9,
                 critic_tau=0.005,
                 critic_target_update_freq=2,
                 encoder_type='pixel',
                 encoder_feature_dim=50,
                 encoder_lr=1e-3,
                 encoder_tau=0.005,
                 num_layers=4,
                 num_filters=32,
                 cpc_update_freq=1,
                 log_interval=100,
                 detach_encoder=False,
                 curl_latent_dim=128,
                 num_attn_layer=2,
                 actor_attach_encoder=False,
                 actor_coeff=1.,
                 adam_warmup_step=None,
                 encoder_annealling=False,
                 mtm_bsz=64,
                 mtm_ema=True,
                 normalize_before=False,
                 relu_dropout=0.,
                 attention_dropout=0.,
                 dropout=0.,
                 ):
        self.device = device
        self.discount = discount
        self.critic_tau = critic_tau
        self.encoder_tau = encoder_tau
        self.actor_update_freq = actor_update_freq
        self.critic_target_update_freq = critic_target_update_freq
        self.cpc_update_freq = cpc_update_freq
        self.log_interval = log_interval
        self.image_size = obs_shape[-1]
        self.curl_latent_dim = curl_latent_dim
        self.detach_encoder = detach_encoder
        self.encoder_type = encoder_type
        self.actor_attach_encoder = actor_attach_encoder
        self.actor_coeff = actor_coeff
        self.mtm_ema = mtm_ema
        self.mtm_bsz = mtm_bsz

        self.actor = Actor(obs_shape, action_shape, hidden_dim, encoder_type,
            encoder_feature_dim, actor_log_std_min, actor_log_std_max,
            num_layers, num_filters).to(device)
        self.critic = Critic(
            obs_shape, action_shape, hidden_dim, encoder_type,
            encoder_feature_dim, num_layers, num_filters
        ).to(device)
        self.critic_target = Critic(
            obs_shape, action_shape, hidden_dim, encoder_type,
            encoder_feature_dim, num_layers, num_filters
        ).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.actor.encoder.copy_conv_weights_from(self.critic.encoder)
        self.log_alpha = torch.tensor(np.log(init_temperature)).to(device)
        self.log_alpha.requires_grad = True
        self.target_entropy = -np.prod(action_shape)

        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(), lr=actor_lr, betas=(actor_beta, 0.999)
        )
        self.critic_optimizer = torch.optim.Adam(
            self.critic.parameters(), lr=critic_lr, betas=(critic_beta, 0.999)
        )
        self.log_alpha_optimizer = torch.optim.Adam(
            [self.log_alpha], lr=alpha_lr, betas=(alpha_beta, 0.999)
        )
        if self.encoder_type == 'pixel':
            self.CTMR = CTMR(encoder_feature_dim,  self.critic, self.critic_target,
                             output_type='continuous', num_attn_layer=num_attn_layer,
                             mtm_bsz=mtm_bsz, normalize_before=normalize_before,
                             dropout=dropout, attention_dropout=attention_dropout,
                             relu_dropout=relu_dropout).to(self.device)
            self.encoder_optimizer = torch.optim.Adam(
                self.critic.encoder.parameters(), lr=2*encoder_lr
            )

            self.cpc_optimizer = torch.optim.Adam(
                [v for k,v in self.CTMR.named_parameters() if 'encoder' not in k],
                lr=encoder_lr,
            )
            if adam_warmup_step:
                lrscheduler = InverseSquareRootSchedule(adam_warmup_step)
                lrscheduler_lambda = lambda x: lrscheduler.step(x)
                self.cpc_lrscheduler = torch.optim.lr_scheduler.LambdaLR(self.cpc_optimizer, lrscheduler_lambda)
                if encoder_annealling:
                    lrscheduler2 = AnneallingSchedule(adam_warmup_step)
                    lrscheduler_lambda2 = lambda  x: lrscheduler2.step(x)
                    self.encoder_lrscheduler = torch.optim.lr_scheduler.LambdaLR(self.encoder_optimizer, lrscheduler_lambda2)
                else:
                    self.encoder_lrscheduler = None

            else:
                self.cpc_lrscheduler = None
            self.cross_entropy_loss = nn.CrossEntropyLoss()

            self.train()
            self.critic_target.train()

    def train(self, training=True):
        self.training = training
        self.actor.train(training)
        self.critic.train(training)
        if self.encoder_type == 'pixel':
            self.CTMR.train(training)

    @property
    def alpha(self):
        return self.log_alpha.exp()

    def select_action(self, obs):
        with torch.no_grad():
            obs = torch.FloatTensor(obs).to(self.device)
            obs = obs.unsqueeze(0)
            mu, _, _, _ = self.actor(
                obs, compute_pi=False, compute_log_pi=False
            )
            return mu.cpu().data.numpy().flatten()

    def sample_action(self, obs):
        if obs.shape[-1] != self.image_size:
            obs = utils.center_crop_image(obs, self.image_size)

        with torch.no_grad():
            obs = torch.FloatTensor(obs).to(self.device)
            obs = obs.unsqueeze(0)
            mu, pi, _, _ = self.actor(obs, compute_log_pi=False)
            return pi.cpu().data.numpy().flatten()

    def update_critic(self, obs, action, reward, next_obs, not_done, L, step):
        with torch.no_grad():
            _, policy_action, log_pi, _ = self.actor(next_obs)
            target_Q1, target_Q2 = self.critic_target(next_obs, policy_action)
            target_V = torch.min(target_Q1,
                                 target_Q2) - self.alpha.detach() * log_pi
            target_Q = reward + (not_done * self.discount * target_V)

        # get current Q estimates
        current_Q1, current_Q2 = self.critic(
            obs, action, detach_encoder=self.detach_encoder)
        critic_loss = F.mse_loss(current_Q1,
                                 target_Q) + F.mse_loss(current_Q2, target_Q)
        if step % self.log_interval == 0:
            L.log('train_critic/loss', critic_loss, step)


        # Optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        self.critic.log(L, step)

    def update_actor_and_alpha(self, obs, L, step):
        # detach encoder, so we don't update it with the actor loss
        _, pi, log_pi, log_std = self.actor(obs, detach_encoder=(not self.actor_attach_encoder))
        actor_Q1, actor_Q2 = self.critic(obs, pi, detach_encoder=(not self.actor_attach_encoder))

        actor_Q = torch.min(actor_Q1, actor_Q2)
        actor_loss = (self.alpha.detach() * log_pi - actor_Q).mean()

        if step % self.log_interval == 0:
            L.log('train_actor/loss', actor_loss, step)
            L.log('train_actor/target_entropy', self.target_entropy, step)
        entropy = 0.5 * log_std.shape[1] * \
            (1.0 + np.log(2 * np.pi)) + log_std.sum(dim=-1)
        if step % self.log_interval == 0:
            L.log('train_actor/entropy', entropy.mean(), step)

        # optimize the actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        self.actor.log(L, step)

        self.log_alpha_optimizer.zero_grad()
        alpha_loss = (self.alpha *
                      (-log_pi - self.target_entropy).detach()).mean()
        if step % self.log_interval == 0:
            L.log('train_alpha/loss', alpha_loss, step)
            L.log('train_alpha/value', self.alpha, step)
        alpha_loss.backward()
        self.log_alpha_optimizer.step()

    def update_cpc(self, cpc_kwargs, L, step):

        obses = cpc_kwargs['obses']
        obses_label = cpc_kwargs['obses_label']
        non_masked = cpc_kwargs['non_masked']
        non_masked = non_masked.reshape(-1)
        x = self.CTMR(obses, mtm=True)
        label = self.CTMR(obses_label, ema=True)
        true_idx = torch.arange(x.shape[0] ).long().to(label.device)
        x = x[non_masked]
        true_idx = true_idx[non_masked]
        logits = self.CTMR.compute_logits(x, label)
        loss =  self.cross_entropy_loss(logits, true_idx)

        self.encoder_optimizer.zero_grad()
        self.cpc_optimizer.zero_grad()
        L.log('train/ctmr_loss', loss, step)
        loss.backward()
        self.encoder_optimizer.step()
        self.cpc_optimizer.step()
        if self.cpc_lrscheduler is not None:
            self.cpc_lrscheduler.step()
            L.log('train/ctmr_cpc_lr', self.cpc_optimizer.param_groups[0]['lr'], step)
            if self.encoder_lrscheduler is not None:
                self.encoder_lrscheduler.step()
                L.log('train/ctmr_encoder_lr', self.encoder_optimizer.param_groups[0]['lr'], step)





    def update(self, replay_buffer, L, step):
        if self.encoder_type == 'pixel':
            obs, action, reward, next_obs, not_done, cpc_kwargs = replay_buffer.sample_ctmr()
        else:
            obs, action, reward, next_obs, not_done = replay_buffer.sample_proprio()

        if step % self.log_interval == 0:
            L.log('train/batch_reward', reward.mean(), step)


        self.update_critic(obs, action, reward, next_obs, not_done, L, step)

        if step % self.actor_update_freq == 0:
            self.update_actor_and_alpha(obs, L, step)

        if step % self.critic_target_update_freq == 0:
            utils.soft_update_params(
                self.critic.Q1, self.critic_target.Q1, self.critic_tau
            )
            utils.soft_update_params(
                self.critic.Q2, self.critic_target.Q2, self.critic_tau
            )
            utils.soft_update_params(
                self.critic.encoder, self.critic_target.encoder,
                self.encoder_tau
            )

        if step % self.cpc_update_freq == 0 and self.encoder_type == 'pixel':
            self.update_cpc(cpc_kwargs, L, step)


    def save(self, model_dir, step):
        torch.save(
            self.actor.state_dict(), '%s/actor_%s.pt' % (model_dir, step)
        )
        torch.save(
            self.critic.state_dict(), '%s/critic_%s.pt' % (model_dir, step)
        )

    def save_curl(self, model_dir, step):
        torch.save(
            self.CTMR.state_dict(), '%s/ctmr_%s.pt' % (model_dir, step)
        )

    def load(self, model_dir, step):
        self.actor.load_state_dict(
            torch.load('%s/actor_%s.pt' % (model_dir, step))
        )
        self.critic.load_state_dict(
            torch.load('%s/critic_%s.pt' % (model_dir, step))
        )