from copy import deepcopy
import random
import game_metrics as gm
from collections import defaultdict
# random.seed(42)

# Player Methods
# get_call_bs
# get_card
# get_card_amt

# Env Loop
# reset
# run_game

cards = ["A", "2", "3", "4", "5", "6", "7", "8", "9", "10", "J", "Q", "K"] * 4

def remove_cards(card_list, card, num):
    """Remove num cards from a card list."""
    card_count = card_list[card]

    if card_count == 0 or num > card_count:
        raise ValueError(f"Agent Invalid Action: cards played do not exist in hand")

    card_list[card] -= num

    return card_list



class Agent:
    def __init__(self, my_index, num_players, agent_args = []):
        raise NotImplementedError

    def get_card(self, intended_card, hand) -> tuple[str, int]:
        raise NotImplementedError

    def get_call_bs(self, player_index, card, card_amt, hand) -> bool:
        raise NotImplementedError

    def give_info(self, player_indexes_picked_up):
        raise NotImplementedError
    
    def reset(self):
        raise NotImplementedError

class BSEnv:
    def __init__(self, agent_types : [Agent], agent_args = [], decks=1):
        self.num_players = len(agent_types)
        self.agent_types = agent_types
        self.players = []
        self.decks = decks
        self.players = []
        for i, agent_type in enumerate(self.agent_types):
            # initialize players
            if len(agent_args) > 0 and len(agent_args[i]) > 0:
                self.players.append(agent_type(i, self.num_players, agent_args[i]))
            else:
                self.players.append(agent_type(i, self.num_players))
        self.reset()

    def sanity_check(self):
        try:
            assert sum([sum(self.player_hands[i].values()) for i in range(self.num_players)]) + len(self.pile) == 52 * self.decks
        except AssertionError:
            print("Assertion failed: Total number of cards does not equal 52.")
            print("Player Hands:")
            for i, hand in enumerate(self.player_hands):
                print(f"Player {i + 1}: {hand}")
            print("Pile:", self.pile)
            raise AssertionError("Total card count does not equal 52.")

    def reset(self):
        self.finished = False
        self.turn = 0
        self.total_turns = 0
        self.pile = []
        self.action_history = []
        deck = deepcopy(cards) * self.decks
        random.shuffle(deck)
        cards_per_player = (52 * self.decks) // self.num_players
        self.player_hands = [defaultdict(int) for _ in range(0, len(self.agent_types))]
        for i in range(len(deck)): 
            self.player_hands[i // cards_per_player][deck[i]] += 1

        for player in self.players:
            player.reset()


    def run_game(self):
        while not self.finished:
            starting_hands = deepcopy(self.player_hands)
            starting_pile = deepcopy(self.pile)

            # get card
            card, card_amt = self.players[self.turn].get_card(cards[self.total_turns % 13], self.player_hands[self.turn])

            # remove cards from hand
            self.player_hands[self.turn] = remove_cards(self.player_hands[self.turn], card, card_amt)
            
            # add card to pile
            [self.pile.append(card) for _ in range(card_amt)]

            # check if bs
            is_bs = cards[self.total_turns % 13] != card

            # collect bs bids from other players
            bids = [False]*self.num_players
            for other_player in range(self.turn + 1, self.turn + self.num_players):
                player_index = other_player % self.num_players

                bs_bid = self.players[player_index].get_call_bs(player_index, cards[self.total_turns % 13], card_amt, self.player_hands[player_index])
                if bs_bid:
                    bids[player_index] = True

            if True in bids:
                if is_bs:
                    # add pile to player hand
                    while len(self.pile) > 0:
                        self.player_hands[self.turn][self.pile.pop()] += 1

                    for player_index in range(self.num_players):
                        self.players[player_index].give_info([self.turn])

                else:
                    for card in self.pile:
                        # split evenly among players who bid true
                        loser_indexes = [other_player for other_player in range(self.num_players) if bids[other_player] == True]
                        
                        pile_size = len(self.pile)
                        for i in range(pile_size):
                            if len(self.pile) == 0:
                                break
                            self.player_hands[loser_indexes[i % len(loser_indexes)]][self.pile.pop()] += 1

                    for player_index in range(self.num_players):
                        self.players[player_index].give_info(loser_indexes)

            self.action_history.append(gm.RoundPlayed(self.turn, self.total_turns, card, card_amt, [i for i in range(self.num_players) if bids[i]], is_bs, starting_hands, deepcopy(self.player_hands), starting_pile, deepcopy(self.pile)))


            self.turn += 1
            self.turn %= self.num_players
            self.total_turns += 1

            # check if player hand is empty
            for player_hand in self.player_hands:
                if sum([player_hand[card] for card in player_hand]) == 0:
                    # end game
                    self.finished = True

            # sanity check to make sure no cards are being duplicated/deleted
            self.sanity_check()
        
        return gm.GameMetrics(self.action_history, self.num_players, self.decks, self.turn - 1)