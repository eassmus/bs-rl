from agents.agent import Agent

import torch
from torch.nn import functional as F
from torch import nn
from torch import optim

import math
import random
import copy

from collections import deque
from collections import namedtuple

observation_space_size = 13 + 4 + 1 # my hand and hand_sizes and pile_sizee
action_space_size = 13 * 4 # 13 cards * 4 possibilities

def action_encode(card, num_cards):
    return card * 4 + num_cards - 1

def action_decode(action):
    return action // 4, action % 4 + 1

class q_net(nn.Module):
    def __init__(self):
        super(q_net, self).__init__()
        self.l1 = nn.Linear(observation_space_size, 256)
        self.l2 = nn.Linear(256, 256)
        self.l3 = nn.Linear(256, action_space_size)

    def forward(self, x):
        out = self.l1(x)
        out = F.relu(out)
        out = self.l2(out)
        out = F.relu(out)
        out = self.l3(out)
        return out

cards = ["A", "2", "3", "4", "5", "6", "7", "8", "9", "10", "J", "Q", "K"]

Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))

if torch.cuda.is_available():
    dev = "cuda:0"
else:
    dev = "cpu"

device = torch.device(dev)

# action_space 

class DQNAgent(Agent):
    def __init__(self, my_index, num_players, agent_args = []):
        self.model = q_net().to(device)
        self.reference_model = q_net().to(device)
        self.optimizer = optim.AdamW(self.model.parameters(), lr=agent_args["learning_rate"])
        self.my_index = my_index
        self.num_players = num_players
        self.num_decks = agent_args["num_decks"]
        self.hand_sizes = [13] * 4
        self.pile_size = 0
        self.last_hand_size = -1
        self.replay_buffer = deque(maxlen=1000)
        self.ep_decay = agent_args["ep_decay"]
        self.ep_start = 0.9
        self.ep_end = 0.01
        self.training_cycles = 0
        self.batch_size = agent_args["batch_size"]
        self.last_state = None
        self.last_action = None
        self.last_card = None
        self.last_hand = None

    def get_ep_chance(self):
        #print(math.exp(-1. * self.training_cycles / self.ep_end), self.training_cycles)
        return self.ep_end + (self.ep_start - self.ep_end) * math.exp(-1. * self.training_cycles / self.ep_decay)

    def get_card(self, intended_card, hand) -> tuple[str, int]:
        self.last_card = cards.index(intended_card)
        self.check_for_reward(len(hand), hand)
        mod_mad = {cards[i] : ((cards.index(intended_card) + i * self.num_players) % 13) for i in range(0,13)}
        rev_mod_map = {((cards.index(intended_card) + i * self.num_players) % 13) : i for i in range(0,13)}  
        mapped_hand = {mod_mad[card] : hand[card] for card in cards}
        state = torch.tensor([mapped_hand[i] for i in range(0,13)] + self.hand_sizes + [self.pile_size], dtype=torch.float32)
        values = self.model(state)
        best_action = None
        best_value = None
        self.last_state = state
        for i in range(0, 52):
            card, num_cards = action_decode(i)
            card = rev_mod_map[card]
            if hand[cards[card]] < num_cards: 
                continue
            if best_value is None or values[i] > best_value:
                best_action = (card, num_cards)
                best_value = values[i]
        #print(self.get_ep_chance())
        if best_action is None or random.random() > self.get_ep_chance():
            a = [i for i in range(0, 52)]
            random.shuffle(a)
            for i in range(52):
                action = action_decode(a[i])
                if hand[cards[action[0]]] >= action[1]:
                    best_action = action
                    break
        self.last_action = action_encode(mod_mad[cards[best_action[0]]], best_action[1])
        self.last_hand_size = self.hand_sizes[0]
        self.pile_size += best_action[1]
        self.hand_sizes[0] -= best_action[1]
        self.last_hand = copy.deepcopy(hand)
        self.last_hand[cards[best_action[0]]] -= best_action[1]
        return cards[best_action[0]], best_action[1]

    def check_for_reward(self, hand_size, hand):
        if self.last_hand_size == -1:
            return
        mod_mad = {cards[i] : ((self.last_card + (i + 1) * self.num_players) % 13) for i in range(0,13)}
        mapped_hand = {mod_mad[card] : self.last_hand[card] for card in cards}
        reward = -hand_size
        self.replay_buffer.append(Transition(self.last_state, self.last_action, torch.tensor([mapped_hand[i] for i in range(0,13)] + self.hand_sizes + [self.pile_size], dtype=torch.float32), reward))
        self.last_hand_size = -1

    def get_call_bs(self, player_index, card, card_amt, hand) -> bool:
        self.hand_sizes[(player_index - self.my_index) % 4] -= card_amt
        self.pile_size += card_amt
        self.check_for_reward(len(hand), hand)
        # calls BS if knows it is BS
        card_count = hand[card]
        if card_count + card_amt > 4:
            return True
        return False
  
    def train(self):
        if len(self.replay_buffer) < self.batch_size:
            return
        transitions = random.sample(self.replay_buffer, self.batch_size)
        for i in range(len(transitions)):
            transitions[i].action
            
        batch = Transition(*zip(*transitions))
        
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                            batch.next_state)), dtype=torch.bool, device=device)
        non_final_next_states = torch.cat([s.unsqueeze(0) for s in batch.next_state
                                                    if s is not None])

        state_batch = torch.cat([state.unsqueeze(0) for state in list(batch.state)])
        action_batch = torch.tensor(batch.action).unsqueeze(0)
        reward_batch = torch.tensor(batch.reward).unsqueeze(0)

        state_action_values = self.model(state_batch).gather(1, action_batch)

        next_state_values = torch.zeros(self.batch_size, device=device)
        with torch.no_grad():
            next_state_values[non_final_mask] = self.reference_model(non_final_next_states).max(1).values
            
        expected_state_action_values = (next_state_values * 0.99) + reward_batch

        criterion = nn.MSELoss()
        #print(state_action_values, expected_state_action_values)
        loss = criterion(state_action_values, expected_state_action_values)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.training_cycles += 1
        if self.training_cycles % 100 == 0:
            self.reference_model.load_state_dict(self.model.state_dict())

    def give_info(self, player_indexes_picked_up):
        for player_index in player_indexes_picked_up:
            self.hand_sizes[(player_index - self.my_index) % 4] += self.pile_size // len(player_indexes_picked_up)
        self.pile_size = 0

    def give_full_info(self, was_bs):
        pass

    def reset(self):
        self.train()
        self.hand_sizes = [13] * 4
        self.pile_size = 0
        self.last_hand_size = -1
        self.last_state = None
        self.last_action = None
        self.last_card = None
        self.last_hand = None
