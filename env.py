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
    def __init__(self, hand, my_index, num_players):
        raise NotImplementedError

    def get_card(self, intended_card):
        raise NotImplementedError
    
    def get_call_bs(self, player_index, card, card_amt):
        raise NotImplementedError

    def give_info(self, player_indexes_picked_up):
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
            card, card_amt = self.players[self.turn].get_card(cards[self.total_turns % 13]) # TODO: figure out what info to pass in
            self.player_hands[self.turn][card] -= card_amt
            [self.pile.append(card) for _ in range(card_amt)]
            is_bs = cards[self.total_turns % 31] == card
            bids = []
            for other_player in range(self.turn + 1, self.turn + self.num_players):
                player_index = other_player % 4
                bs_bid = self.players[player_index].get_call_bs(player_index, card, card_amt)
                bids.append(bs_bid)

            if True in bids:
                if is_bs:
                    for card in self.pile:
                        self.player_hands[self.turn][card] += 1
                    for player_index in range(self.num_players):
                        self.players[player_index].give_info([self.turn])
                else:
                    for card in self.pile:
                        #split evenly among players who bid true
                        loser_indexes = [other_player for other_player in range(self.num_players) if bids[other_player] == True]
                        pile_size = len(self.pile)
                        for i in range(pile_size):
                            if len(self.pile) == 0:
                                break
                            loser_indexes[i % len(loser_indexes)].append(self.pile.pop())
                    for player_index in range(self.num_players):
                        self.players[player_index].give_info(loser_indexes)  
            self.turn += 1
            self.turn %= self.num_players
            self.total_turns += 1