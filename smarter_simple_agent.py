from env import Agent

cards = ["A", "2", "3", "4", "5", "6", "7", "8", "9", "10", "J", "Q", "K"]

# this agent tries to play what it won't have to play for a while 
class SmartSimpleAngent:
    def __init__(self, my_index, num_players, agent_args = []):
        self.my_index = my_index
        self.num_players = num_players

    def get_card(self, intended_card, hand):
        current_card = cards.index(intended_card)
        future_cards = [cards[(current_card + i * self.num_players) % 13] for i in range(1,14)]
        for card in future_cards[::-1]:
            if hand[card] > 0:
                return card, hand[card]
        random_chosen = [card for card in hand if hand[card] > 0][0]
        return random_chosen, hand[random_chosen]
    
    def get_call_bs(self, player_index, card, card_amt, hand):
        # calls BS if knows it is BS
        card_count = hand[card]
        return card_count + card_amt > 4
    
    def give_info(self, player_indexes_picked_up):
        pass

    def reset(self):
        pass