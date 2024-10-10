from agents.agent import Agent
import random

class RandomAgent(Agent):
    def __init__(self, my_index, num_players, agent_args = []):
        if len(agent_args) > 0:
            self.random_chance = float(agent_args["random_chance"]) 
        else: 
            self.random_chance = 0.5

    def get_card(self, intended_card, hand):
        return random.choice([card for card in hand if hand[card] > 0]), 1
    
    def get_call_bs(self, player_index, card, card_amt, hand):
        return random.random() < self.random_chance

    def give_info(self, player_indexes_picked_up):
        pass

    def give_full_info(self, was_bs):
        pass

    def reset(self):
        pass