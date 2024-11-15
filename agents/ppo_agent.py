import numpy as np
import torch
import os
import glob
import time
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import MultivariateNormal
from torch.distributions import Categorical

import numpy as np

# set device to cpu or cuda
device = torch.device('cpu')

if(torch.cuda.is_available()):
    device = torch.device('cuda:0')
    torch.cuda.empty_cache()
    print("Device set to : " + str(torch.cuda.get_device_name(device)))
else:
    print("Device set to : cpu")


cards = ["A", "2", "3", "4", "5", "6", "7", "8", "9", "10", "J", "Q", "K"]

# PPO code adapted from PPO-Pytorch
# added compatibility with action masks with minor changes
class RolloutBuffer:
    def __init__(self):
        self.actions = []
        self.states = []
        self.logprobs = []
        self.rewards = []
        self.state_values = []
        self.is_terminals = []
        self.action_masks = []


    def clear(self):
        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.state_values[:]
        del self.is_terminals[:]
        del self.action_masks[:]


class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, has_continuous_action_space, action_std_init):
        super(ActorCritic, self).__init__()

        self.has_continuous_action_space = has_continuous_action_space

        if has_continuous_action_space:
            self.action_dim = action_dim
            self.action_var = torch.full((action_dim,), action_std_init * action_std_init).to(device)

        # actor
        if has_continuous_action_space :
            self.actor = nn.Sequential(
                            nn.Linear(state_dim, 64),
                            nn.Tanh(),
                            nn.Linear(64, 64),
                            nn.Tanh(),
                            nn.Linear(64, action_dim),
                            nn.Tanh()
                        )
        else:
            self.actor = nn.Sequential(
                            nn.Linear(state_dim, 64),
                            nn.Tanh(),
                            nn.Linear(64, 64),
                            nn.Tanh(),
                            nn.Linear(64, action_dim),
                        )


        # critic
        self.critic = nn.Sequential(
                        nn.Linear(state_dim, 64),
                        nn.Tanh(),
                        nn.Linear(64, 64),
                        nn.Tanh(),
                        nn.Linear(64, 1)
                    )

    def set_action_std(self, new_action_std):

        if self.has_continuous_action_space:
            self.action_var = torch.full((self.action_dim,), new_action_std * new_action_std).to(device)
        else:
            print("--------------------------------------------------------------------------------------------")
            print("WARNING : Calling ActorCritic::set_action_std() on discrete action space policy")
            print("--------------------------------------------------------------------------------------------")


    def forward(self):
        raise NotImplementedError


    def act(self, state, action_mask=None):
        
        if self.has_continuous_action_space:
            action_mean = self.actor(state)
            cov_mat = torch.diag(self.action_var).unsqueeze(dim=0)
            dist = MultivariateNormal(action_mean, cov_mat)
        else:
            logits = self.actor(state)
            if action_mask is not None:
                # mask out invalid actions
                logits = logits + (action_mask == 0).float() * -1e10

            action_probs = torch.softmax(logits, dim=-1)
            dist = Categorical(action_probs)

        action = dist.sample()
        action_logprob = dist.log_prob(action)
        state_val = self.critic(state)

        return action.detach(), action_logprob.detach(), state_val.detach()


    def evaluate(self, state, action, action_mask=None):

        if self.has_continuous_action_space:
            action_mean = self.actor(state)
            action_var = self.action_var.expand_as(action_mean)
            cov_mat = torch.diag_embed(action_var).to(device)
            dist = MultivariateNormal(action_mean, cov_mat)

            # for single action continuous environments
            if self.action_dim == 1:
                action = action.reshape(-1, self.action_dim)

        else:
            logits = self.actor(state)
            if action_mask is not None:
                # mask out invalid actions
                logits = logits + (action_mask == 0).float() * -1e10

            action_probs = torch.softmax(logits, dim=-1)
            dist = Categorical(action_probs)

        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()
        state_values = self.critic(state)

        return action_logprobs, state_values, dist_entropy


class PPO:
    def __init__(self, state_dim, action_dim, lr_actor, lr_critic, gamma, K_epochs, eps_clip, has_continuous_action_space, action_std_init=0.6):

        self.has_continuous_action_space = has_continuous_action_space

        if has_continuous_action_space:
            self.action_std = action_std_init

        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs

        self.buffer = RolloutBuffer()

        self.policy = ActorCritic(state_dim, action_dim, has_continuous_action_space, action_std_init).to(device)
        self.optimizer = torch.optim.Adam([
                        {'params': self.policy.actor.parameters(), 'lr': lr_actor},
                        {'params': self.policy.critic.parameters(), 'lr': lr_critic}
                    ])

        self.policy_old = ActorCritic(state_dim, action_dim, has_continuous_action_space, action_std_init).to(device)
        self.policy_old.load_state_dict(self.policy.state_dict())

        self.MseLoss = nn.MSELoss()


    def set_action_std(self, new_action_std):

        if self.has_continuous_action_space:
            self.action_std = new_action_std
            self.policy.set_action_std(new_action_std)
            self.policy_old.set_action_std(new_action_std)

        else:
            print("--------------------------------------------------------------------------------------------")
            print("WARNING : Calling PPO::set_action_std() on discrete action space policy")
            print("--------------------------------------------------------------------------------------------")


    def decay_action_std(self, action_std_decay_rate, min_action_std):
        print("--------------------------------------------------------------------------------------------")

        if self.has_continuous_action_space:
            self.action_std = self.action_std - action_std_decay_rate
            self.action_std = round(self.action_std, 4)
            if (self.action_std <= min_action_std):
                self.action_std = min_action_std
                print("setting actor output action_std to min_action_std : ", self.action_std)
            else:
                print("setting actor output action_std to : ", self.action_std)
            self.set_action_std(self.action_std)

        else:
            print("WARNING : Calling PPO::decay_action_std() on discrete action space policy")

        print("--------------------------------------------------------------------------------------------")


    def select_action(self, state, action_mask):

        if self.has_continuous_action_space:
            with torch.no_grad():
                state = torch.FloatTensor(state).to(device)
                action, action_logprob, state_val = self.policy_old.act(state, action_mask)

            self.buffer.states.append(state)
            self.buffer.actions.append(action)
            self.buffer.logprobs.append(action_logprob)
            self.buffer.state_values.append(state_val)

            return action.detach().cpu().numpy().flatten()

        else:
            with torch.no_grad():
                state = torch.FloatTensor(state).to(device)
                action, action_logprob, state_val = self.policy_old.act(state, action_mask)
                
            self.buffer.action_masks.append(action_mask)
            self.buffer.states.append(state)
            self.buffer.actions.append(action)
            self.buffer.logprobs.append(action_logprob)
            self.buffer.state_values.append(state_val)

            return action.item()


    def update(self):

        # Monte Carlo estimate of returns
        rewards = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(self.buffer.rewards), reversed(self.buffer.is_terminals)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)

        # Normalizing the rewards
        rewards = torch.tensor(rewards, dtype=torch.float32).to(device)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-7)

        # convert list to tensor
        old_states = torch.squeeze(torch.stack(self.buffer.states, dim=0)).detach().to(device)
        old_actions = torch.squeeze(torch.stack(self.buffer.actions, dim=0)).detach().to(device)
        old_logprobs = torch.squeeze(torch.stack(self.buffer.logprobs, dim=0)).detach().to(device)
        old_state_values = torch.squeeze(torch.stack(self.buffer.state_values, dim=0)).detach().to(device)
        old_action_masks = torch.squeeze(torch.stack(self.buffer.action_masks, dim=0)).detach().to(device)

        # calculate advantages
        advantages = rewards.detach() - old_state_values.detach()


        # Optimize policy for K epochs
        for _ in range(self.K_epochs):

            # Evaluating old actions and values
            logprobs, state_values, dist_entropy = self.policy.evaluate(old_states, old_actions,old_action_masks)

            # match state_values tensor dimensions with rewards tensor
            state_values = torch.squeeze(state_values)

            # Finding the ratio (pi_theta / pi_theta__old)
            ratios = torch.exp(logprobs - old_logprobs.detach())

            # Finding Surrogate Loss
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1-self.eps_clip, 1+self.eps_clip) * advantages

            # final loss of clipped objective PPO
            loss = -torch.min(surr1, surr2) + 0.5 * self.MseLoss(state_values, rewards) - 0.01 * dist_entropy

            # take gradient step
            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()

        # Copy new weights into old policy
        self.policy_old.load_state_dict(self.policy.state_dict())

        # clear buffer
        self.buffer.clear()


    def save(self, checkpoint_path):
        torch.save(self.policy_old.state_dict(), checkpoint_path)


    def load(self, checkpoint_path):
        self.policy_old.load_state_dict(torch.load(checkpoint_path, map_location=lambda storage, loc: storage))
        self.policy.load_state_dict(torch.load(checkpoint_path, map_location=lambda storage, loc: storage))

class PPOAgent():
    """ BS Agent class implemented for the PPO algorithm. Can only handle 4 players and 1 deck atm."""
    def __init__(self, my_index, num_players):
        self.my_index = my_index
        self.num_players = num_players

        # keep track of cards
        self.track_pile = {card : 0 for card in cards}
        self.track_pile_list = []

        self.track_player_hands = [{card : 0 for card in cards} for _ in range(self.num_players)]
        self.hand_sizes = [13] * self.num_players

        # ppo agent
        has_continuous_action_space = False
        max_training_timesteps = int(1e5)
        max_ep_len = 400
        update_timestep = max_ep_len * 4
        K_epochs = 40
        eps_clip = 0.2
        gamma = 0.99
        lr_actor = 0.0003
        lr_critic = 0.001
        action_std = None

        action_dim = 13*4+2 # 13 card types * 4 (choose card amount) + is_bs (0,1)
        self.state_dim = 426

        self.ppo_agent = PPO(self.state_dim, action_dim, lr_actor, lr_critic, gamma, K_epochs, eps_clip, has_continuous_action_space, action_std)
        self.current_ep_reward = 0
        self.state = None
        self.previous_hand_size = 0
        self.time_step = 0

    @staticmethod
    def parse_action(index):
        """Parses out the action in the action encoding given an action index. 4x13 + 2"""
        if index < 52:
            card_type_index = index // 4  # Determine card type (0-12)
            card_amount_index = index % 4 + 1  # Determine amount (1-4)
            return {"play_card": (cards[card_type_index], card_amount_index), "is_bs": None}

        else:
            is_bs_flag = index - 52  # 52 -> 0 (no bs), 53 -> 1 (bs)
            return {"play_card": None, "is_bs": is_bs_flag}

    @staticmethod
    def get_card_encoding(card_dict):
        """Encodes a card dictionary by one hot encoding."""
        one_hot_encoding = []

        for card, count in card_dict.items():
            one_hot_vector = [1 if i == count else 0 for i in range(5)]
            one_hot_encoding.extend(one_hot_vector)

        return torch.tensor(one_hot_encoding, dtype=torch.long)


    def generate_state(self, hand, intended_card= 0, card = 0, card_amt = 0, player_to_bid_bs_on=0):
        """Generates the state based on the tracks and scenerio.

        State Encoding Overview:

        General Card Info
        my_hand_encoding: 13x5
        other_players_hand_encoding: 3x13x5
        other_player_hand_amount: 3
        pile: 13x5

        Card Play Info
        intended_card: 13+1 (0 is for not relevant)

        Call BS Info
        card: 13+1 (0 is for not relevant)
        card_amt: 4+1 (0 is for not relevant)
        player_playing_hand: 13x5 (0 is for not relevant)

        for a total of... 426 features
        """
        my_hand_encoding = self.get_card_encoding(hand)
        # encode other players hands (one-hot)
        other_players_hand_encoding = []
        for i, track_hand in enumerate(self.track_player_hands):
            if i !=self.my_index:
                other_players_hand_encoding.extend(self.get_card_encoding(track_hand))
        other_players_hand_encoding = torch.tensor(other_players_hand_encoding)
        # encode other player hand amounts by divide by 4
        hand_amount = []
        for i, hand_amt in enumerate(self.hand_sizes):
            if i !=self.my_index:
                hand_amount.append(hand_amt)
        other_players_hand_amt_encoding = torch.tensor(hand_amount)/4

        # encode the pile
        pile_encoding = self.get_card_encoding(self.track_pile)

        # encode all the action specific stuff (one-hot)
        intended_card_encoding = F.one_hot(torch.tensor(intended_card),14)
        card_encoding = F.one_hot(torch.tensor(card),14)
        card_amt_encoding = F.one_hot(torch.tensor(card_amt),5)

        if player_to_bid_bs_on != -1:
            player_playing_hand = self.get_card_encoding(self.track_player_hands[player_to_bid_bs_on])
        else:
            player_playing_hand = torch.zeros(65)

        # concat them all together
        state = torch.concat([my_hand_encoding,
                        other_players_hand_encoding,
                        other_players_hand_amt_encoding,
                        pile_encoding,
                        intended_card_encoding,
                        card_encoding,
                        card_amt_encoding, 
                        player_playing_hand])
  

        assert state.shape[0] == self.state_dim

        if self.state == None:
            # this is the beginning of the game so no reward
            self.previous_hand_size = len(hand)
            return state, 0

        # reward based on a reduction of previous hand size
        reward = len(self.previous_hand_size) - len(hand)
        self.previous_hand_size = len(hand)

        return state, reward


    def update_model(self, reward, done):
        self.ppo_agent.buffer.rewards.append(reward)
        self.ppo_agent.buffer.is_terminals.append(done)
        self.current_ep_reward += reward
        self.time_step+=1

        # update agent every 200 steps
        if self.time_step % 200 == 0:
            self.ppo_agent.update()

        # make sure we didn't do anything stupid
        assert(len(self.ppo_agent.buffer.rewards) == len(self.ppo_agent.buffer.actions))

    def get_card(self, intended_card, hand):
        # populate every key in hand :(
        [hand[card] for card in cards]
        state, reward = self.generate_state(hand,intended_card=cards.index(intended_card))

        # mask out invalid actions
        action_mask = torch.zeros(54)
        for index in range(52):
            card_type_index = index // 4
            card_amount_index = index % 4 +1
            
            card_type = cards[card_type_index]  
            
            if hand[card_type] >= card_amount_index:
                action_mask[index] = 1 

    
        action_index = self.ppo_agent.select_action(state, action_mask)
        action = self.parse_action(action_index)
        self.update_model(reward, False)

        return action['play_card']

    def get_call_bs(self, player_index, card, card_amt, hand):
        # populate every key in hand :(
        [hand[card] for card in cards]
        # update our card tracks
        self.hand_sizes[player_index] -= card_amt
        self.track_pile[card]+=card_amt
        self.track_pile_list.extend([card]*card_amt)

        state, reward = self.generate_state(hand,card=cards.index(card),card_amt=card_amt,player_to_bid_bs_on=player_index)    
        action_mask = torch.zeros(54)
        action_mask[52:] = 1
        action_index = self.ppo_agent.select_action(state, action_mask)
        action = self.parse_action(action_index)
        self.update_model(reward, False)

        return action['is_bs']

    def give_info(self, player_indexes_picked_up):
        # update the player hand tracking
        loser_indexes = player_indexes_picked_up

        pile_size = len(self.track_pile_list)
        for i in range(pile_size):
            if len(self.track_pile_list) == 0:
                break
            self.track_player_hands[loser_indexes[i % len(loser_indexes)]][self.track_pile_list.pop()] += 1
            self.hand_sizes[loser_indexes[i % len(loser_indexes)]] +=1

    def is_finished(self, winner_index):
        # replace the last action reward with the win/lose reward
        if len(self.ppo_agent.buffer.rewards) == 0: 
            return
        if self.my_index == winner_index:
            self.ppo_agent.buffer.rewards[-1] = 500
            self.ppo_agent.buffer.is_terminals[-1] = True
        else:
            self.ppo_agent.buffer.rewards[-1] = -500
            self.ppo_agent.buffer.is_terminals[-1] = -500

    def give_full_info(self, was_bs):
        pass
    
    def reset(self):
        pass

    def give_winner(self, winner):
        pass

