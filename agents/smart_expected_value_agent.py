from agents.agent import Agent

cards = ["A", "2", "3", "4", "5", "6", "7", "8", "9", "10", "J", "Q", "K"]

# this agent tries to play what it won't have to play for a while 
class SmartExpectedValueAgent:
    def __init__(self, my_index, num_players, agent_args = []):
        self.my_index = my_index
        self.num_players = num_players
        self.expected_values = None # generated later when we are given our first hand
        self.num_decks = agent_args["num_decks"]
        self.threshold = agent_args["threshold"]
        self.hand_sizes = [(52 * self.num_decks) // self.num_players] * self.num_players
 
    def gen_initial_expected_values(self, hand):
        self.expected_values = [{card : 0 for card in cards} for _ in range(self.num_players)]
        cards_left = {card: 4 * self.num_decks for card in cards}
        for card in hand:
            cards_left[card] -= 1
            self.expected_values[self.my_index][card] += 1

        for i in range(self.num_players):
            if i != self.my_index:
                self.expected_values[i] = {card : cards_left[card] / (self.num_players - 1) for card in cards}

    def update_expected_values(self, hand):
        for card in cards:
            self.expected_values[self.my_index][card] = hand[card]

    def get_card(self, intended_card, hand):
        if self.expected_values is None:
            self.gen_initial_expected_values(hand)
        self.update_expected_values(hand)
        # using get_card strategy from smater_simple_agent
        current_card = cards.index(intended_card)
        future_cards = [cards[(current_card + i * self.num_players) % 13] for i in range(1,14)]
        for card in future_cards[::-1]:
            if hand[card] > 0:
                for _  in range(hand[card]):
                    self.in_pile.append(card)
                return card, hand[card]
        random_chosen = [card for card in hand if hand[card] > 0][0]
        for _ in range(hand[random_chosen]):
            self.in_pile.append(random_chosen)
        return random_chosen, hand[random_chosen]

    def get_call_bs(self, player_index, card, card_amt, hand):
        for _ in range(card_amt):
            self.in_pile.append(card)
        if self.expected_values is None:
            self.gen_initial_expected_values(hand)
        self.update_expected_values(hand) 
        return (hand[card] > (4 - card_amt)) or card_amt - self.expected_values[player_index][card] > self.threshold
    
    def give_info(self, player_indexes_picked_up):
        for card in self.in_pile:
            for player_index in player_indexes_picked_up:
                self.expected_values[player_index][card] += 1 / len(player_indexes_picked_up)

        for player in player_indexes_picked_up:
            self.hand_sizes[player] += len(self.in_pile) / 3

        self.in_pile = []

    def give_full_info(self, was_bs):
        pass
    
    def reset(self):
        self.expected_values = None
        self.in_pile = []

    def give_winner(self, winner):
        pass
