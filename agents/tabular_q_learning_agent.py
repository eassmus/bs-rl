from agents.agent import Agent
from agents.bs_call_learning_agent import BSCallLearningAgent

from collections import deque
import math
import random

import numpy as np

cards = ["A", "2", "3", "4", "5", "6", "7", "8", "9", "10", "J", "Q", "K"]

def state_enocde(hand):
    out = 0
    for card in cards:
        out += min(hand[card], 1)
        out *= 2
    out //= 2
    return out

class TabularQLearningAgent(Agent):
    def __init__(self, my_index, num_players, agent_args = []):
        self.my_index = my_index
        self.num_players = num_players
        self.num_decks = agent_args["num_decks"]
        self.ep_decay = agent_args["ep_decay"]
        self.ep_start = 0.9
        if "ep_start" in agent_args and agent_args["ep_start"] is not None:
            self.ep_start = agent_args["ep_start"]
        self.ep_end = 0.1
        self.trajectory = []

        self.learning_rate = 0.001
        self.last_hand_size = 13
        self.alpha = 0.99
        self.training_cycles = 0

        self.q_table = np.random.random((2**13, 4, 13)) # hand, cards_placed, action

        self.bs_agent = BSCallLearningAgent(my_index, num_players, agent_args)

    def check_for_reward(self, hand):
        if self.last_hand_size == -1:
            return
        reward = self.last_hand_size - sum(hand.values())
        self.trajectory.append(reward)
        self.last_hand_size = -1

    def get_ep_threshhold(self):
        return self.ep_end + (self.ep_start - self.ep_end) * math.exp(
            -1.0 * self.training_cycles / self.ep_decay
        )
    
    def get_card(self, intended_card, hand) -> tuple[str, int]:
        self.check_for_reward(hand)

        self.bs_agent.get_card(intended_card, hand)
        
        self.last_hand_size = sum(hand.values())

        mod_mad = {
            cards[i]: ((cards.index(intended_card) + i * self.num_players) % 13)
            for i in range(0, 13)
        }
        rev_mod_map = {cards[((cards.index(intended_card) + i * self.num_players) % 13)]: i for i in range(0, 13)}
        mapped_hand = {cards[mod_mad[card]]: hand[card] for card in cards}

        state = state_enocde(mapped_hand)

        self.trajectory.append(state)

        best_play = None
        best_value = None
        for i in range(13):
            for j in range(4):
                if hand[cards[rev_mod_map[cards[i]]]] < j + 1:
                    continue
                if best_value is None or self.q_table[state][j][i] > best_value:
                    best_play = (rev_mod_map[cards[i]], j + 1)
                    best_value = self.q_table[state][j][i]
        if best_play is None or random.random() > self.get_ep_threshhold():
            a = [i for i in range(0, 52)]
            random.shuffle(a)
            for i in range(52):
                action = (a[i] // 4, a[i] % 4 + 1)
                if hand[cards[action[0]]] >= action[1]:
                    best_play = (action[0], action[1])
                    break

        self.trajectory.append(best_play)

        return cards[best_play[0]], best_play[1]

    def get_call_bs(self, player_index, card, card_amt, hand) -> bool:
        self.check_for_reward(hand)
        return self.bs_agent.get_call_bs(player_index, card, card_amt, hand)

    def give_info(self, player_indexes_picked_up):
        self.bs_agent.give_info(player_indexes_picked_up)
    
    def reset(self):
        self.trajectory = []
        self.last_hand_size = 13
        self.bs_agent.reset()
    
    def give_full_info(self, was_bs):
        self.bs_agent.give_full_info(was_bs)

    def give_winner(self, winner):
        if winner == self.my_index:
            self.trajectory.append(100)
        self.bs_agent.give_winner(winner)
        self.train()

    def train(self):
        gain = 0
        for i in range(len(self.trajectory) - 3, -1, -3):
            state = self.trajectory[i]
            action = self.trajectory[i + 1]
            reward = self.trajectory[i + 2]
            self.q_table[state][min(3,action[1])][action[0]] += self.learning_rate * (reward + gain - self.q_table[state][min(3,action[1])][action[0]])
            gain += reward
            gain *= self.alpha
        self.training_cycles += 1
