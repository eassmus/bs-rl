from copy import deepcopy
import random
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
    count = 0
    new_card_list = []

    for item in card_list:
        if item == card and count < num:
            count += 1
        else:
            new_card_list.append(item)

    return new_card_list



class Agent:
    def __init__(self, my_index, num_players):
        raise NotImplementedError

    def get_card(self, intended_card, hand) -> tuple[str, int]:
        raise NotImplementedError

    def get_call_bs(self, player_index, card, card_amt, hand) -> bool:
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

class BSEnv:
    def __init__(self, agent_types : [Agent], decks=1):
        self.num_players = len(agent_types)
        self.agent_types = agent_types
        self.players = []
        self.decks = decks
        self.reset()

    def sanity_check(self):
        try:
            assert len(self.player_hands[0]) + len(self.player_hands[1]) + len(self.player_hands[2]) + len(self.player_hands[3]) + len(self.pile) == 52
        except AssertionError:
            print("Assertion failed: Total number of cards does not equal 52.")
            print("Player Hands:")
            for i, hand in enumerate(self.player_hands):
                print(f"Player {i + 1}: {hand}")
            print("Pile:", self.pile)
            print(len(self.player_hands[0]) + len(self.player_hands[1]) + len(self.player_hands[2]) + len(self.player_hands[3]) + len(self.pile))
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
        self.player_hands = [deck[i:i+cards_per_player] for i in range(0, len(deck), cards_per_player)]

        self.players = []
        for i, agent_type in enumerate(self.agent_types):
            # initialize players
            self.players.append(agent_type(i, self.num_players))


    def run_game(self):
        while not self.finished:
            print("-----------")
            print(f"Turn: {self.total_turns} Card to Play: {cards[self.total_turns % 13]}")
            print(f"Pile {self.pile}")

            # get card
            card, card_amt = self.players[self.turn].get_card(cards[self.total_turns % 13], self.player_hands[self.turn]) # TODO: figure out what info to pass in
            print(f"Player {self.turn} current hand {self.player_hands[self.turn]}")

            # remove cards from hand
            self.player_hands[self.turn] = remove_cards(self.player_hands[self.turn], card, card_amt)
            print(f"Player {self.turn} plays {card} {card_amt} time(s).")
            print(f"Player {self.turn} new hand {self.player_hands[self.turn]}")

            # add card to pile
            [self.pile.append(card) for _ in range(card_amt)]

            # check if bs
            is_bs = cards[self.total_turns % 13] != card

            # collect bs bids from other players
            bids = [False]*4
            for other_player in range(self.turn + 1, self.turn + self.num_players):
                player_index = other_player % 4

                bs_bid = self.players[player_index].get_call_bs(player_index, card, card_amt, self.player_hands[player_index])
                if bs_bid:
                    print(f"Player {player_index} bids BS")
                    bids[player_index] = True

            if True in bids:
                if is_bs:
                    # add pile to player hand
                    for card in self.pile:
                        self.player_hands[self.turn].append(self.pile.pop())

                    print(f"Player {self.turn} takes the pile. Their new hand is {self.player_hands[self.turn]}")
                    for player_index in range(self.num_players):
                        self.players[player_index].give_info([self.turn])

                else:
                    for card in self.pile:
                        # split evenly among players who bid true
                        loser_indexes = [other_player for other_player in range(self.num_players) if bids[other_player] == True]
                        print(f"It was not BS. Splitting the pile {self.pile}")
                        [print(f"Player {loser_index} old hand: {self.player_hands[loser_index]}") for loser_index in loser_indexes]

                        pile_size = len(self.pile)
                        for i in range(pile_size):
                            if len(self.pile) == 0:
                                break
                            self.player_hands[loser_indexes[i % len(loser_indexes)]].append(self.pile.pop())

                        [print(f"Player {loser_index} new hand: {self.player_hands[loser_index]}") for loser_index in loser_indexes]


                    for player_index in range(self.num_players):
                        self.players[player_index].give_info(loser_indexes)

            self.turn += 1
            self.turn %= self.num_players
            self.total_turns += 1

            # check if player hand is empty
            for player_hand in self.player_hands:
                if len(player_hand) == 0:
                    # end game
                    self.finished = True

            # sanity check to make sure no cards are being duplicated/deleted
            self.sanity_check()