from typing import List, Optional, Type, Tuple
import torch
import torch.nn as nn
from stable_baselines3 import TD3
from stable_baselines3.td3.policies import TD3Policy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from gymnasium import spaces
import numpy as np
from stable_baselines3.common.preprocessing import get_action_dim,get_obs_shape


import torch as th
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.policies import BaseModel
from stable_baselines3.td3.policies import Actor, CnnPolicy, MlpPolicy, MultiInputPolicy, TD3Policy




class CustomActor(Actor):
    """
    Actor network (policy) for TD3.
    """
    def __init__(self, *args, **kwargs):
        super(CustomActor, self).__init__(*args, **kwargs)
        action_dim = get_action_dim(self.action_space)
        obs_dim = get_obs_shape(self.observation_space)
        num_agents, individual_action_dim = self.action_space.shape
        individual_obs_dim = np.prod(obs_dim)//num_agents


        self.mu = AttentionNetwork(individual_obs_dim,individual_action_dim, num_agents)

class AvgContinuousCritic(BaseModel):

    features_extractor: BaseFeaturesExtractor

    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Box,
        net_arch: list[int],
        features_extractor: BaseFeaturesExtractor,
        features_dim: int,
        activation_fn: type[nn.Module] = nn.ReLU,
        normalize_images: bool = True,
        n_critics: int = 2,
        share_features_extractor: bool = True,
    ):
        super().__init__(
            observation_space,
            action_space,
            features_extractor=features_extractor,
            normalize_images=normalize_images,
        )

        action_dim = get_action_dim(self.action_space)

        self.share_features_extractor = share_features_extractor
        self.n_critics = n_critics
        num_agents, individual_action_dim = self.action_space.shape
        self.q_networks: list[nn.Module] = []
        for idx in range(n_critics):
            q_net = AttentionNetwork(features_dim // num_agents + individual_action_dim,
                                        1,
                                        num_agents,
                                        squash_output=False,
                                        hidden_size = 64,
                                        aggregate_output='mean'
                                    )
            self.add_module(f"qf{idx}", q_net)
            self.q_networks.append(q_net)

    def forward(self, obs: th.Tensor, actions: th.Tensor) -> tuple[th.Tensor, ...]:
        # Learn the features extractor using the policy loss only
        # when the features_extractor is shared with the actor
        with th.set_grad_enabled(not self.share_features_extractor):
            features = self.extract_features(obs, self.features_extractor)
        qvalue_input = th.cat([features, actions], dim=1)
        return tuple(q_net(qvalue_input) for q_net in self.q_networks)

    def q1_forward(self, obs: th.Tensor, actions: th.Tensor) -> th.Tensor:
        """
        Only predict the Q-value using the first network.
        This allows to reduce computation when all the estimates are not needed
        (e.g. when updating the policy in TD3).
        """
        with th.no_grad():
            features = self.extract_features(obs, self.features_extractor)
        return self.q_networks[0](th.cat([features, actions], dim=1))

class CustomTD3Policy(TD3Policy):
    def __init__(self, *args, **kwargs):
        super(CustomTD3Policy, self).__init__(*args, **kwargs)

    def make_actor(self, features_extractor: Optional[BaseFeaturesExtractor] = None) -> CustomActor:
        actor_kwargs = self._update_features_extractor(self.actor_kwargs, features_extractor)
        return CustomActor(**actor_kwargs).to(self.device)
    
    def make_critic(self, features_extractor: Optional[BaseFeaturesExtractor] = None) -> AvgContinuousCritic:
        critic_kwargs = self._update_features_extractor(self.critic_kwargs, features_extractor)
        return AvgContinuousCritic(**critic_kwargs).to(self.device)


class DistQNetwork(nn.Module):
    def __init__(self, 
                    input_dim,
                    output_dim, 
                    n_agents,
                    hidden_size = 64,
                    squash_output = True,
                    aggregate_output = None
                ):
        super().__init__()
        self.input_dim = input_dim
        self.n_agents = n_agents
        self.output_dim = output_dim
        self.aggregate_output = aggregate_output

        self.fc =   nn.Sequential(
                        nn.Linear(self.input_dim*self.n_agents, hidden_size),
                        nn.ReLU(),
                        nn.Linear(hidden_size, hidden_size),
                        nn.ReLU(),
                        nn.Linear(hidden_size, self.output_dim),
                    )
        if squash_output:
            self.fc = nn.Sequential(*self.fc, nn.Tanh())
    
    def forward(self, x):
        return self.fc(x)
    

class AttentionNetwork(nn.Module):
    def __init__(self, 
                    input_dim,
                    output_dim, 
                    n_agents, 
                    hidden_size = 64,
                    final_msg_dim = 16,
                    key_dim = 4,
                    squash_output = True,
                    aggregate_output = None,
                ):
        super().__init__()
        self.input_dim = input_dim
        self.n_agents = n_agents
        self.output_dim = output_dim
        self.scale = 1/np.sqrt(key_dim)
        self.aggregate_output = aggregate_output

        self.fc =   nn.Sequential(
                        nn.Linear(self.input_dim + final_msg_dim, hidden_size),
                        nn.ReLU(),
                        nn.Linear(hidden_size, hidden_size),
                        nn.ReLU(),
                        nn.Linear(hidden_size, output_dim),
                    )
        if squash_output:
            self.fc = nn.Sequential(*self.fc, nn.Tanh())
    
        self.K = nn.Linear(self.input_dim, key_dim)
        self.Q = nn.Linear(self.input_dim, key_dim)
        self.V = nn.Linear(self.input_dim, final_msg_dim)
        
        if self.aggregate_output == 'mean':
            self.agg = lambda x: torch.mean(x,dim=-2)
        elif self.aggregate_output == 'sum':
            self.agg = lambda x: torch.sum(x,dim=-2)
        elif self.aggregate_output == 'mlp':
            self.mixer = nn.Linear(self.n_agents*self.output_dim, 1)
            self.agg = lambda x: self.mixer(x.transpose(-1, -2)).unsqueeze(-1)
        else:
            self.agg = lambda x: torch.flatten(x,start_dim=-2)
        
        # LSTM for recurrency
        # self.lstm = nn.LSTM( rnn_hidden_dim, rnn_hidden_dim, batch_first=True)

        # # Final layer to produce Q-values
        # self.q_net = nn.Linear( rnn_hidden_dim, 5)

        # # Hidden state initialization
        # self.hidden_state = None  # (h_n, c_n) for LSTM

    def forward(self, observations):
        obs_shape = observations.shape
        observations = observations.reshape(obs_shape[:-1]+(self.n_agents,-1))
        k = self.K(observations)
        q = self.Q(observations)
        v = self.V(observations)

        kq_prod = (k.unsqueeze(-2)*q.unsqueeze(-3)).sum(-1)*self.scale
        attention = torch.softmax(kq_prod,dim=-1)
        cumulated = torch.bmm(attention, v)

        fc_input = torch.cat([cumulated,observations],dim=-1)

        x = self.fc(fc_input)

        # # Ensure LSTM state is maintained
        # if self.hidden_state is None:
        # 	self.hidden_state = (torch.zeros(1, x.shape[0], 128).to(x.device),
        # 						torch.zeros(1, x.shape[0], 128).to(x.device))

        # x, self.hidden_state = self.lstm(x.unsqueeze(0), self.hidden_state)
        # x = self.q_net(x.squeeze(0))  # Remove batch dim  

        # print("x shape:\t",x.shape)
        
        out = self.agg(x)

        return out
