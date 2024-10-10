from agents.agent import Agent
import random

class AggressiveAgent(Agent):
    def __init__(self, my_index, num_players):
        pass 

    def get_card(self, intended_card, hand):
        if hand[intended_card] > 0:
            return intended_card, hand[intended_card]
        return random.choice([card for card in hand if hand[card] > 0]), 1
    
    def get_call_bs(self, player_index, card, card_amt, hand):
        return hand[card] > 0

    def give_info(self, player_indexes_picked_up):
        pass

    def give_full_info(self, was_bs):
        pass
    
    def reset(self):
        pass