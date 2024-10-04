import env
import random

class AggressiveAgent(env.Agent):
    def __init__(self, my_index, num_players):
        pass 

    def get_card(self, intended_card, hand):
        if intended_card in hand:
            return intended_card, hand.count(intended_card)
        return random.choice(hand), 1
    
    def get_call_bs(self, player_index, card, card_amt, hand):
        if card in hand:
            return True
        return False

    def give_info(self, player_indexes_picked_up):
        pass