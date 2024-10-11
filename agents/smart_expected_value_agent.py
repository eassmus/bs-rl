from agents.agent import Agent

cards = ["A", "2", "3", "4", "5", "6", "7", "8", "9", "10", "J", "Q", "K"]

# this agent tries to play what it won't have to play for a while 
class SmartExpectedValueAngent:
    def __init__(self, my_index, num_players, agent_args = []):
        self.my_index = my_index
        self.num_players = num_players
        self.expected_values = None # generated later when we are given our first hand
 
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
                return card, hand[card]
        random_chosen = [card for card in hand if hand[card] > 0][0]
        return random_chosen, hand[random_chosen]

    def get_call_bs(self, player_index, card, card_amt, hand):
        if len(self.data) > 0 and self.data[-1][0] == "data":
            self.data.pop()
        if self.expected_values is None:
            self.gen_initial_expected_values(hand)
        self.update_expected_values(hand) 
        return hand[card] > 4 - card_amt or self.expected_values[player_index][card] < -1.5
    
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



class BSCallLearningAgent(Agent):
    def __init__(self, my_index, num_players, agent_args):
        self.my_index = my_index
        self.num_players = num_players
        self.num_decks = agent_args["num_decks"]
        self.expected_values = None # generated later when we are given our first hand
        self.in_pile = []
        self.model = _Model(7, agent_args["hidden_layer_size"], 2)
        self.data = []
        self.hand_sizes = [self.num_decks * 13] * self.num_players
        self.last_caller = None
        self.criterion = nn.CrossEntropyLoss()  # Loss function for classification tasks
        self.optimizer = optim.Adam(self.model.parameters(), lr=agent_args["learning_rate"])
        self.train_every = agent_args["train_every"]
        if "load_model" in agent_args and agent_args["load_model"] is not None:
            self.load_model(agent_args["load_model"])

    def gen_initial_expected_values(self, hand):
        self.expected_values = [{card : 0 for card in cards} for _ in range(self.num_players)]
        cards_left = {card: 4 * self.num_decks for card in cards}
        for card in hand:
            cards_left[card] -= 1
            self.expected_values[self.my_index][card] += 1

        for i in range(self.num_players):
            if i != self.my_index:
                self.expected_values[i] = {card : cards_left[card] / (self.num_players - 1) for card in cards}

