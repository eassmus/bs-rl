from agents.agent import Agent
import random

class SimpleAgent(Agent):
    def __init__(self, my_index, num_players, agent_args = []):
        self.num_players= num_players


    def get_card_count(self, target_card, hand):
        return hand[target_card]

    def get_card(self, intended_card, hand):
        # check if has card and if so, plays it
        card_count = self.get_card_count(intended_card, hand)
        if card_count > 0:
            return intended_card, card_count
        # play a random card if not
        return [card for card in hand if hand[card] > 0][0], 1

    def get_call_bs(self, player_index, card, card_amt, hand):
        # calls BS if knows it is BS
        card_count = self.get_card_count(card, hand)
        if card_count + card_amt > 4:
            return True
        # random BS call for fun

        return False

    def give_info(self, player_indexes_picked_up):
        pass
    
    def give_full_info(self, was_bs):
        pass

    def reset(self):
        pass

    def give_winner(self, winner):
        pass