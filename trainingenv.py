import env
from agents.agent import Agent
from copy import deepcopy
import game_metrics as gm

cards = ["A", "2", "3", "4", "5", "6", "7", "8", "9", "10", "J", "Q", "K"]


def remove_cards(card_list, card, num):
    """Remove num cards from a card list."""
    card_count = card_list[card]

    if card_count == 0 or num > card_count:
        raise ValueError(f"Agent Invalid Action: cards played do not exist in hand")

    card_list[card] -= num

    return card_list


class TrainingBSEnv(env.BSEnv):
    def __init__(self, agent_types: list[Agent], agent_args=[], decks=1):
        super().__init__(agent_types, agent_args, decks)

    def reset(self):
        super().reset()

    def run_game(self):
        self.reset()
        winner = None
        while not self.finished:
            if self.total_turns > 400:
                print("Game Not Stopping")
                self.finished = True
            starting_hands = deepcopy(self.player_hands)
            starting_pile = deepcopy(self.pile)

            # get card
            card, card_amt = self.players[self.turn].get_card(
                cards[self.total_turns % 13], self.player_hands[self.turn]
            )

            # remove cards from hand
            self.player_hands[self.turn] = remove_cards(
                self.player_hands[self.turn], card, card_amt
            )

            # add card to pile
            [self.pile.append(card) for _ in range(card_amt)]

            # check if bs
            is_bs = cards[self.total_turns % 13] != card

            # collect bs bids from other players
            bids = [False] * self.num_players
            for other_player in range(self.turn + 1, self.turn + self.num_players):
                player_index = other_player % self.num_players
                bs_bid = self.players[player_index].get_call_bs(
                    self.turn,
                    cards[self.total_turns % 13],
                    card_amt,
                    self.player_hands[player_index],
                )
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
                    # split evenly among players who bid true
                    loser_indexes = [
                        other_player
                        for other_player in range(self.num_players)
                        if bids[other_player] == True
                    ]

                    pile_size = len(self.pile)
                    for i in range(pile_size):
                        if len(self.pile) == 0:
                            break
                        self.player_hands[loser_indexes[i % len(loser_indexes)]][
                            self.pile.pop()
                        ] += 1

                    for player_index in range(self.num_players):
                        self.players[player_index].give_info(loser_indexes)

            for player_index in range(self.num_players):
                self.players[player_index].give_full_info(is_bs)

            self.action_history.append(
                gm.RoundPlayed(
                    self.turn,
                    self.total_turns,
                    card,
                    card_amt,
                    [i for i in range(self.num_players) if bids[i]],
                    is_bs,
                    starting_hands,
                    deepcopy(self.player_hands),
                    starting_pile,
                    deepcopy(self.pile),
                )
            )
            self.turn += 1
            self.turn %= self.num_players
            self.total_turns += 1

            # check if player hand is empty
            for i in range(self.num_players):
                if sum(self.player_hands[i].values()) == 0:
                    winner = i
                    break

            if winner is not None:
                self.finished = True
                for player in self.players:
                    player.give_winner(winner)

            # sanity check to make sure no cards are being duplicated/deleted
            self.sanity_check()

        return gm.GameMetrics(self.action_history, self.num_players, self.decks, winner)
