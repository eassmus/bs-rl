from copy import deepcopy
import random
# Player Methods
# get_call_bs
# get_card
# get_card_amt

# Env Loop
# reset
# run_game

cards = ["A", "2", "3", "4", "5", "6", "7", "8", "9", "10", "J", "Q", "K"] * 4

def card_to_index(card):
    match card:
        case "A":
            return 0
        case "2":
            return 1
        case "3":
            return 2
        case "4":
            return 3
        case "5":
            return 4
        case "6":
            return 5
        case "7":
            return 6
        case "8":
            return 7
        case "9":
            return 8
        case "10":
            return 9
        case "J":
            return 10
        case "Q":
            return 11
        case "K":
            return 12

class Agent:
    def __init__(self, hand, my_index, num_players):
        raise NotImplementedError

    def get_card(self, intended_card) -> tuple[str, int]:
        raise NotImplementedError
    
    def get_call_bs(self, player_index, card, card_amt) -> bool:
        raise NotImplementedError

    def give_info(self, player_indexes_picked_up):
        raise NotImplementedError

class BSEnv:
    def __init__(self, agent_types : [Agent], decks=1):
        self.num_players = len(agent_types)
        self.agent_types = agent_types
        self.players = []
        self.decks = decks
        self.reset()

    def reset(self):
        self.finished = False
        self.turn = 0
        self.total_turns = 0
        self.pile = []
        self.action_history = []
        deck = deepcopy(cards) * self.decks
        random.shuffle(deck)
        cards_per_player = (52 * self.decks) // self.num_players
        self.player_hands = [deck[i:i+cards_per_player] for i in range(0, len(deck), cards_per_player)]

        self.players = []
        for i, agent_type in enumerate(self.agent_types):
            # initialize players
            self.players.append(agent_type(self.player_hands[i], i, self.num_players))


    def run_game(self):
        while not self.finished:
            # get card
            card, card_amt = self.players[self.turn].get_card(cards[self.total_turns % 13]) # TODO: figure out what info to pass in
            self.player_hands[self.turn][card_to_index(card)] -= card_amt
            [self.pile.append(card) for _ in range(card_amt)]
            is_bs = cards[self.total_turns % 13] == card
            bids = []
            for other_player in range(self.turn + 1, self.turn + self.num_players):
                player_index = other_player % 4
                bs_bid = self.players[player_index].get_call_bs(player_index, card, card_amt)
                bids.append(bs_bid)
            if True in bids:
                if is_bs:
                    for card in self.pile:
                        self.player_hands[self.turn][card_to_index(card)] += 1
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
            for player_index in range(self.num_players):
                empty = True
                for card in cards:
                    if self.player_hands[player_index][card_to_index(card)] > 0:
                        empty = False
                        break
                if empty:
                    self.finished = True
                    break
