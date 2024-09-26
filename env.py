from copy import deepcopy

# Player Methods
# get_call_bs
# get_card
# get_card_amt

# Env Loop
# reset
# run_game

cards = ["A", "2", "3", "4", "5", "6", "7", "8", "9", "10", "J", "Q", "K"] * 4

class Agent:
    def get_card(self):
        raise NotImplementedError
    
    def get_card_amt(self):
        raise NotImplementedError
    
    def get_call_bs(self):
        raise NotImplementedError

class BSEnv:
    def __init__(self, agents : [Agent], decks=1):
        self.num_players = len(agents)
        self.players = agents
        self.decks = decks
        self.reset()

    def reset(self):
        self.finished = False
        self.turn = 0
        self.total_turns = 0
        self.pile = []
        self.action_history = []
        deck = deepcopy(cards) * self.decks
        deck.shuffle()
        cards_per_player = (52 * self.decks) // self.num_players
        self.player_hands = [deck[i:i+cards_per_player] for i in range(0, len(deck), cards_per_player)]

    def run_game(self):
        while not self.finished:
            # get card
            card = self.players[self.turn].get_card() # TODO: figure out what info to pass in
            card_amt = self.players[self.turn].get_card_amt() #player checks it is legal for now
            self.player_hands[self.turn][card] -= card_amt
            [self.pile.append(card) for _ in range(card_amt)]
            is_bs = cards[self.total_turns % 31] == card
            bids = []
            for other_player in range(self.turn + 1, self.turn + self.num_players):
                player_index = other_player % 4
                bs_bid = self.players[player_index].get_call_bs()
                bids.append(bs_bid)

            
            
            self.turn += 1
            self.turn %= self.num_players
            self.total_turns += 1