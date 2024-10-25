from matplotlib import pyplot as plt
from collections import deque

cards = ["A", "2", "3", "4", "5", "6", "7", "8", "9", "10", "J", "Q", "K"]

class RoundPlayed:
    def __init__(self, player_index, total_turns, card_played, card_amt, bs_calls, was_bs, starting_hands, ending_hands, starting_pile, ending_pile):
        self.player_index = player_index
        self.total_turns = total_turns
        self.card_played = card_played
        self.card_amt = card_amt
        self.bs_calls = bs_calls
        self.was_bs = was_bs
        self.starting_hands = starting_hands
        self.ending_hands = ending_hands
        self.starting_pile = starting_pile
        self.ending_pile = ending_pile
    
    def __str__(self):
        out = ""
        out += ("-----------") + "\n"
        out += (f"Turn: {self.total_turns} Card to Play: {cards[self.total_turns % 13]}") + "\n"
        out += (f"Pile {self.starting_pile}") + "\n"
        out += (f"Player {self.player_index} current hand {self.starting_hands[self.player_index]}") + "\n"
        out += (f"Player {self.player_index} plays {self.card_played} {self.card_amt} time(s).") + "\n"
        out += (f"Player {self.player_index} new hand {self.ending_hands[self.player_index]}") + "\n"
        for player in self.bs_calls:
            out += (f"Player {player} bids BS") + "\n"
        if self.was_bs and len(self.bs_calls) > 0:
            out += (f"Player {self.player_index} takes the pile. Their new hand is {self.ending_hands[self.player_index]}") + "\n"
        elif not self.was_bs and len(self.bs_calls) > 0:
            out += (f"It was not BS. Splitting the pile {self.starting_pile + [self.card_played] * self.card_amt}") + "\n"
            for loser_index in self.bs_calls:
                out += (f"Player {loser_index} old hand: {self.starting_hands[loser_index]}") + "\n"
                out += (f"Player {loser_index} new hand: {self.ending_hands[loser_index]}") + "\n"
        return out
    
class GameMetrics:
    def __init__(self, rounds, num_players, decks, winner):
        self.num_players = num_players
        self.rounds = rounds
        self.decks = decks
        self.winner = winner

    def get_text(self):
        out = ""
        for round in self.rounds:
            out += round.__str__()
            out += "\n\n"
        return out

# game metrics plotting

def plt_bs_accuracy(game_metrics, player_index, n = None):
    if n is None:
        n = len(game_metrics) // 10
    player_rate = []
    calls = deque()
    window_correct = 0
    window_incorrect = 0
    for game in game_metrics:
        for round in game.rounds:
            if (player_index in round.bs_calls) == round.was_bs:
                calls.append(1)
                window_correct += 1
            else:
                calls.append(0)
                window_incorrect += 1
            if len(calls) > n:
                r = calls.popleft()
                if r == 1:
                    window_correct -= 1
                else:
                    window_incorrect -= 1
                player_rate.append(window_correct / (window_correct + window_incorrect))

    plt.figure("BS Accuracy")
    plt.plot(player_rate)

def plt_bs_call_rate(game_metrics, player_index, n = None):
    if n is None:
        n = len(game_metrics) // 10
    calls = deque()
    window_bs = 0
    window_not_bs = 0
    rate = []
    for game in game_metrics:
        for round in game.rounds:
            if player_index in round.bs_calls:
                calls.append(1)
                window_bs += 1
            else:
                calls.append(0)
                window_not_bs += 1
            if len(calls) > n:
                r = calls.popleft()
                if r == 1:
                    window_bs -= 1
                else:
                    window_not_bs -= 1
            rate.append(window_bs / (window_bs + window_not_bs))

    plt.figure("BS Call Rate")
    plt.plot(rate)

def plt_true_bs_ratio(game_metrics, player_index = None,n = None):
    if n is None:
        n = len(game_metrics) // 10
    calls = deque()
    window_bs = 0
    window_not_bs = 0
    rate = []
    for game in game_metrics:
        for round in game.rounds:
            if player_index is not None and round.player_index not in player_index:
                continue
            if round.was_bs:
                calls.append(1)
                window_bs += 1
            else:
                calls.append(0)
                window_not_bs += 1
            if len(calls) > n:
                r = calls.popleft()
                if r == 1:
                    window_bs -= 1
                else:
                    window_not_bs -= 1
            rate.append(window_bs / (window_bs + window_not_bs))

    plt.figure("BS Call Rate " + str(player_index))
    plt.plot(rate)
    

def plt_avg_delta_cards(game_metrics, player_index, n = None):
    if n is None:
        n = len(game_metrics) // 10
    calls = deque()
    window_delta = 0
    rate = []
    for game in game_metrics:
        for round in game.rounds:
            if round.player_index == player_index:
                calls.append(len(round.starting_hands[player_index]) - len(round.ending_hands[player_index]))
                window_delta += len(round.starting_hands[player_index]) - len(round.ending_hands[player_index])
            if len(calls) > n:
                window_delta -= calls.popleft()
                rate.append(window_delta / len(calls))

    plt.figure("Avg. Delta Cards " + str(player_index))
    plt.plot(rate)
    

def plt_win_rate(game_metrics, player_index, n = None):
    if n is None:
        n = len(game_metrics) // 10
    calls = deque()
    window_wins = 0
    rate = []
    for game in game_metrics:
        if game.winner == player_index:
            window_wins += 1
            calls.append(1)
        else:
            calls.append(0)
        if len(calls) > n:
            window_wins -= calls.popleft()
            rate.append(window_wins / len(calls))

    plt.figure("Win Rate " + str(player_index))
    plt.plot(rate)
    
