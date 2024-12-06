import os
import glob
import time
from datetime import datetime

import torch
import torch.nn as nn
from torch.distributions import MultivariateNormal
from torch.distributions import Categorical

import numpy as np

# set device to cpu or cuda
device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

if(torch.cuda.is_available()):
    device = torch.device('cuda:0')
    torch.cuda.empty_cache()
    print("Device set to : " + str(torch.cuda.get_device_name(device)))
else:
    print("Device set to : cpu")

import torch.nn.functional as F

cards = ["A", "2", "3", "4", "5", "6", "7", "8", "9", "10", "J", "Q", "K"]

# PPO code adapted from PPO-Pytorch
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
        self.imitate_optimizer = torch.optim.Adam(self.actor.parameters(), 1e-3)
        self.CrossEntropyLoss = nn.CrossEntropyLoss()
        
    def set_action_std(self, new_action_std):

        if self.has_continuous_action_space:
            self.action_var = torch.full((self.action_dim,), new_action_std * new_action_std).to(device)
        else:
            print("--------------------------------------------------------------------------------------------")
            print("WARNING : Calling ActorCritic::set_action_std() on discrete action space policy")
            print("--------------------------------------------------------------------------------------------")


    def forward(self):
        raise NotImplementedError


    def act(self, state, action_mask=None, intended_card_index=None):

        if self.has_continuous_action_space:
            action_mean = self.actor(state)
            cov_mat = torch.diag(self.action_var).unsqueeze(dim=0)
            dist = MultivariateNormal(action_mean, cov_mat)
        else:
            logits = self.actor(state).to(device)
            if action_mask is not None:
                # mask out invalid actions
                logits = logits + (action_mask + 1e-45).log()
            #print(action_mask)
            action_probs = torch.softmax(logits, dim=-1)
            #print(action_probs)
            if intended_card_index == -1:
                action_probs[0] *= 2
                action_probs = action_probs / action_probs.sum()
                
            dist = Categorical(action_probs[action_mask.to(torch.bool)])
            
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
        
        state_values = self.critic(state)
        dist_entropy = dist.entropy()

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


    def select_action(self, state, action_mask,intended_card_index=None):

        if self.has_continuous_action_space:
            with torch.no_grad():
                state = torch.FloatTensor(state).to(device)
                action, action_logprob, state_val, action_logdis = self.policy_old.act(state, action_mask)

            self.buffer.states.append(state)
            self.buffer.actions.append(action)
            self.buffer.logprobs.append(action_logprob)
            self.buffer.state_values.append(state_val)

            return action.detach().cpu().numpy().flatten()

        else:
            with torch.no_grad():
                state = torch.FloatTensor(state).to(device)
                action, action_logprob, state_val = self.policy_old.act(state, action_mask,intended_card_index=intended_card_index)

            self.buffer.action_masks.append(action_mask)
            self.buffer.states.append(state)
            self.buffer.actions.append(action)
            self.buffer.logprobs.append(action_logprob)
            self.buffer.state_values.append(state_val)

            return action.item()


    def update(self):
        logs = {}
        # Monte Carlo estimate of returns
        rewards = []
        discounted_reward = 0
        #print(self.buffer.rewards)
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
            logs['critic_loss'] = self.MseLoss(state_values, rewards)
            # take gradient step
            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()

        # Copy new weights into old policy
        self.policy_old.load_state_dict(self.policy.state_dict())
        
        # clear buffer
        self.buffer.clear()
        #print(logs['critic_loss'])
        #print(state_values[-3:-1])
        #print(rewards[-3:-1])

    def save(self, checkpoint_path):
        torch.save(self.policy_old.state_dict(), checkpoint_path)


    def load(self, checkpoint_path):
        self.policy_old.load_state_dict(torch.load(checkpoint_path, map_location=lambda storage, loc: storage))
        self.policy.load_state_dict(torch.load(checkpoint_path, map_location=lambda storage, loc: storage))
        


class PPOAgent():
    AGENT_CARDS = ["A", "2", "3", "4", "5", "6", "7", "8", "9", "10", "J", "Q", "K"]
    """ BS Agent class implemented for the PPO algorithm. Can only handle 4 players and 1 deck atm."""
    def __init__(self, my_index, num_players, agent_args = {}):
        self.my_index = my_index
        self.num_players = num_players
        self.do_training = True
        if "do_training" in agent_args and agent_args["do_training"] is not None:
            self.do_training = agent_args["do_training"]
        # keep track of cards
        self.track_pile = {card : 0 for card in self.AGENT_CARDS}
        self.track_pile_list = []

        self.track_player_hands = [{card : 0 for card in self.AGENT_CARDS} for _ in range(self.num_players)]
        self.hand_sizes = [13] * self.num_players

        # ppo agent
        has_continuous_action_space = False
        max_training_timesteps = int(1e5)
        max_ep_len = 50
        update_timestep = max_ep_len * 2
        K_epochs = 40
        eps_clip = 0.35
        gamma = 0.9999
        lr_actor = 0.003
        lr_critic = 0.01
        action_std = None

        self.bs_state_dim = 45
        self.play_state_dim = 100
        
        card_playing_action_dim = 13*4 # 13 card types * 4 (choose card amount)
        self.card_playing_ppo_agent = PPO(self.play_state_dim, card_playing_action_dim, lr_actor, lr_critic, gamma, K_epochs, eps_clip, has_continuous_action_space, action_std)
        bs_action_dim = 2 # is_bs (0,1)
        self.bs_ppo_agent = PPO(self.bs_state_dim, bs_action_dim, lr_actor, lr_critic, gamma, K_epochs, eps_clip, has_continuous_action_space, action_std)
        
        self.card_playing_ppo_agent.load('agent_weights/30000_simple_agents_play_policy.pth')
        self.bs_ppo_agent.load('agent_weights/30000_simple_agents_bs_policy.pth')

        self.state = None
        self.previous_hand_size = 0
        self.previous_hand_size_bs = 0
        self.hand_size_from_bs = 0
        self.hand_size_from_play = 0
        self.previous_bs_call = 0

        self.bs_time_step = 0
        self.play_time_step = 0
        self.bs_total_rewards = 0
        self.player_total_rewards = 0
        
    def parse_action(self, index, intended_card):        
        """Parses out the action in the action encoding given an action index. 4x13"""
        intended_card_index = self.AGENT_CARDS.index(intended_card)
        card_type_index = index // 4 - intended_card_index  # Determine card type (0-12)
        card_amount_index = index % 4 + 1  # Determine amount (1-4)
        return {"play_card": (self.AGENT_CARDS[card_type_index], card_amount_index)}


    def get_card_encoding(self, card_dict, intended_card_index):
        """Encodes a card dictionary by one hot encoding. Order is based on intended card."""
        one_hot_encoding = torch.zeros((13*5))
        
        for i, card in enumerate(self.AGENT_CARDS):
            assert card_dict[card] < 5
            one_hot_index = (i-intended_card_index)*5 + card_dict[card]

            one_hot_encoding[one_hot_index] = 1
            
        return one_hot_encoding

    def get_size_encoding(self, x):
        """Returns a tensor of size 6. One-hot, indicating what range the size is in."""
        encoding = torch.zeros(6)
        if x > 10:
            encoding[0] = 1
        elif 5 < x <= 10:
            encoding[1] = 1
        elif 3 < x <= 5:
            encoding[2] = 1
        elif x == 2:
            encoding[3] = 1
        elif x == 1:
            encoding[4] = 1
        elif x == 0:
            encoding[5] = 1
        return encoding
        

    def generate_state(self, hand, intended_card= "2", card_amt = 0, player_to_bid_bs_on=-1):
        """Generates the state based on the tracks and scenerio.

        State Encoding Overview:

        General Card Info
        my_hand_encoding: 13x5 (in order of mod) = 45
        my_hand_size_encoding: 6
        other_player_hand_amount: (x > 10, 5 < x < 10, 3 < x < 5, x = 2, x = 1) * 3 = 18
        pile_size: x > 10, 5 < x < 10, 3 < x < 5, x = 2, x = 1, x = 0 = 6
        player_playing_hand_size_encoding: (how much of the card I have) = 5
        
        Call BS Info:
        my_hand_size_encoding: 6
        card_amt: 4
        player_playing_hand_size: x > 10, 5 < x < 10, 3 < x < 5, x = 2, x = 1, x = 0 = 6
        player_playing_hand_size_encoding: (how much of the card I have) = 5
        
        for a total of... 82 features for playing card agent, and 153 features for bs agent
        """        
        my_hand_encoding = self.get_card_encoding(hand, self.AGENT_CARDS.index(intended_card))
        my_hand_size_encoding = self.get_size_encoding(self.hand_sizes[self.my_index])
        
        # encode other player hand amounts
        hand_amount = []
        for i, hand_amt in enumerate(self.hand_sizes):
            if i !=self.my_index:
                hand_amount.append(self.get_size_encoding(hand_amt))
        other_players_hand_amt_encoding = torch.concat(hand_amount,dim=0)

        # encode the pile        
        pile_size =  self.get_size_encoding(len(self.track_pile_list))
        cards_in_my_hand_encoding = F.one_hot(torch.tensor(hand[intended_card]),5)

        # encode extra for bs agent
        if player_to_bid_bs_on != -1:
            card_amt_encoding = F.one_hot(torch.tensor(card_amt-1),4)
            player_playing_hand_size = self.get_size_encoding(self.hand_sizes[player_to_bid_bs_on])
            
            # how many of the card in my hand
            cards_in_my_hand_encoding = F.one_hot(torch.tensor(hand[intended_card]),5)

            state = torch.concat([my_hand_size_encoding,
                            other_players_hand_amt_encoding,
                            pile_size,
                            cards_in_my_hand_encoding,
                            card_amt_encoding,
                            player_playing_hand_size])
            
            assert state.shape[0] == self.bs_state_dim
        else:
            state = torch.concat([
                            cards_in_my_hand_encoding, my_hand_encoding, 
                            my_hand_size_encoding,
                            other_players_hand_amt_encoding,
                            pile_size])
            assert state.shape[0] == self.play_state_dim

        if self.state == None:
            if player_to_bid_bs_on == -1:
                self.previous_hand_size = sum(hand.values())
                self.hand_size_from_play = 0
            else:
                self.previous_hand_size_bs = sum(hand.values())
                self.hand_size_from_bs = 0

            self.state = "Training"
            # this is the beginning of the game so no reward
            return state, 0
        
        reward = 0
        if player_to_bid_bs_on == -1:
            # reward based on previous hand size
            self.previous_hand_size = sum(hand.values()) 
            reward = self.hand_size_from_play/4
            reward = max(-6,reward)
        else:
            if player_to_bid_bs_on == (self.my_index+1) % 4:
                self.hand_size_from_play = self.previous_hand_size - sum(hand.values())
                
            reward = self.previous_hand_size_bs - sum(hand.values())
            reward = max(-20,reward)
            reward = min(reward, 0)
            
            if reward == 0 and self.previous_bs_call == 1:
                if (player_to_bid_bs_on-1) % 4 != self.my_index:
                    reward+= 2/self.hand_sizes[(player_to_bid_bs_on-1) % 4]
                else:
                    reward+= 2/self.hand_sizes[(player_to_bid_bs_on-2) % 4]
                
            self.previous_hand_size_bs = sum(hand.values()) 
            
            if self.previous_bs_call == 0:
                return state, 1

        return state, reward

    def update_model(self, model, reward, done, is_bs):
        model.buffer.rewards.append(reward)
        model.buffer.is_terminals.append(done)

        if is_bs:
            self.bs_total_rewards += reward
        else:
            self.player_total_rewards += reward
 
        # make sure we didn't do anything stupid
        assert(len(model.buffer.rewards) == len(model.buffer.actions))

    def get_card(self, intended_card, hand):
        # populate every key in hand :(
        [hand[card] for card in self.AGENT_CARDS]
        state, reward = self.generate_state(hand,intended_card=intended_card)
        current_card = self.AGENT_CARDS.index(intended_card)
        card_cycle = [(current_card + i * self.num_players) % 13 for i in range(1,14)][::-1]
        
        action_index_map = [0]*52
        total_index = 0
        action_mask = []
        # reconstruct action cycle and masks based on avialable hand
        for card in card_cycle:
            if hand[self.AGENT_CARDS[card]] > 0:
                action_mask.extend([1]*hand[self.AGENT_CARDS[card]])
                for i in range(hand[self.AGENT_CARDS[card]]):
                    action_index_map[total_index] =(self.AGENT_CARDS[card],hand[self.AGENT_CARDS[card]]-i) 
                    total_index+=1
                    
        # pad the rest of mask since they are invalid
        action_mask+=[0]*(52-len(action_mask))
        action_mask = torch.tensor(action_mask).to(device)
        action_index = self.card_playing_ppo_agent.select_action(state, action_mask)
        card_type, card_amt = action_index_map[action_index]
                
        # only track what we played
        self.track_pile[card_type]+=card_amt
        self.track_pile_list.extend([card_type]*card_amt)
        
        should_play = None
        # add tiny punishment for playing incorrect card
        if hand[intended_card] > 0:
            if card_type == intended_card:
                reward += 2
            else:
                reward-=1
        if self.do_training:
            self.update_model(self.card_playing_ppo_agent, reward, False, False)
        return action_index_map[action_index]

    def get_call_bs(self, player_index, card, card_amt, hand):
        # populate every key in hand :(
        [hand[card] for card in self.AGENT_CARDS]
        # update our card tracks
        self.hand_sizes[player_index] -= card_amt
        
        self.track_player_hands[player_index][card]-=card_amt
        self.track_player_hands[player_index][card] = max(self.track_player_hands[player_index][card], 0)
        
        # add unk cards to track pile
        self.track_pile_list.extend(["unk"]*card_amt)

        state, reward = self.generate_state(hand,intended_card=card,card_amt=card_amt,player_to_bid_bs_on=player_index)
        action_mask = torch.ones(2).to(device)
        action_index = self.bs_ppo_agent.select_action(state, action_mask, intended_card_index=-1)
        
        self.update_model(self.bs_ppo_agent, reward, False, True)
        
        self.previous_bs_call = action_index
        
        return action_index

    def give_info(self, player_indexes_picked_up):
        # update the player hand tracking
        loser_indexes = player_indexes_picked_up

        pile_size = len(self.track_pile_list)
        for i in range(pile_size):
            if len(self.track_pile_list) == 0:
                break
            card = self.track_pile_list.pop()
            if card != "unk":
                if self.track_player_hands[loser_indexes[i % len(loser_indexes)]][card] < 4:
                    self.track_player_hands[loser_indexes[i % len(loser_indexes)]][card] += 1
                
                
            self.hand_sizes[loser_indexes[i % len(loser_indexes)]] +=1
            
        self.track_pile = {card : 0 for card in self.AGENT_CARDS}

    def is_finished(self, winner_index):
        # replace the last action reward with the win/lose reward
        if len(self.bs_ppo_agent.buffer.rewards) !=0:
            if self.my_index == winner_index:
                self.bs_ppo_agent.buffer.rewards[-1] = 200
                self.bs_ppo_agent.buffer.is_terminals[-1] = True
            else:
                self.bs_ppo_agent.buffer.rewards[-1] = -200
                self.bs_ppo_agent.buffer.is_terminals[-1] = True
                
        if len(self.card_playing_ppo_agent.buffer.rewards) !=0:
            if self.my_index == winner_index:
                self.card_playing_ppo_agent.buffer.rewards[-1] = 200
                self.card_playing_ppo_agent.buffer.is_terminals[-1] = True
            else:
                self.card_playing_ppo_agent.buffer.rewards[-1] = -200
                self.card_playing_ppo_agent.buffer.is_terminals[-1] = True
                
        self.play_time_step+=1
        if self.play_time_step % 4 == 0:
            self.card_playing_ppo_agent.update()
            self.bs_ppo_agent.update()
        
    def give_full_info(self, was_bs):
        pass

    def reset(self):
        self.state = None
        self.previous_hand_size = 0
        
        self.bs_total_rewards = 0
        self.player_total_rewards = 0

        self.track_pile = {card : 0 for card in self.AGENT_CARDS}
        self.track_pile_list = []

        self.track_player_hands = [{card : 0 for card in self.AGENT_CARDS} for _ in range(self.num_players)]
        self.hand_sizes = [13] * self.num_players
    
    def give_winner(self, winner):
        pass

