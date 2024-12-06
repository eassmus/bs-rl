from agents.agent import Agent
from torch import nn
from torch import optim
import torch
from torch import tensor
from torch import float32

import math

cards = ["A", "2", "3", "4", "5", "6", "7", "8", "9", "10", "J", "Q", "K"]

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

sigmoid = nn.Sigmoid()

class _Model(nn.Module):
    def __init__(self, input_size, output_size):
        super(_Model, self).__init__()
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 128)
        self.fc4 = nn.Linear(128, output_size)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.relu(out)
        out = self.fc3(out)
        out = self.relu(out)
        out = self.fc4(out)
        return out


class BSCallLearningAgent(Agent):
    def __init__(self, my_index, num_players, agent_args):
        self.my_index = my_index
        self.num_players = num_players
        self.num_decks = agent_args["num_decks"]
        self.expected_values = None  # generated later when we are given our first hand
        self.in_pile = []
        self.model = _Model(6, 1).to(device)
        self.data = []
        self.hand_sizes = [(52 * self.num_decks) // self.num_players] * self.num_players
        self.last_caller = None
        self.train_every = agent_args["train_every"]
        self.turns = 0
        self.do_fancy = agent_args["do_fancy"]
        self.required_confidence = agent_args["required_confidence"]
        if "load_model_bs" in agent_args and agent_args["load_model_bs"] is not None:
            self.load_model(agent_args["load_model_bs"])
        self.do_training = True
        if "do_training_bs" in agent_args and agent_args["do_training_bs"] is not None:
            self.do_training = agent_args["do_training_bs"]
        if self.do_training:
            self.criterion = nn.BCEWithLogitsLoss()
            self.optimizer = optim.Adam(
                self.model.parameters(), lr=agent_args["learning_rate"]
            )

    def gen_initial_expected_values(self, hand):
        self.expected_values = [
            {card: 0 for card in cards} for _ in range(self.num_players)
        ]
        cards_left = {card: 4 * self.num_decks for card in cards}
        for card in hand:
            cards_left[card] -= hand[card]
            self.expected_values[self.my_index][card] += hand[card]

        for i in range(self.num_players):
            if i != self.my_index:
                self.expected_values[i] = {
                    card: cards_left[card] / (self.num_players - 1) for card in cards
                }

    def train(self):
        #print("Training...")
        if self.do_training is False:
            return
        outputs = self.model(
            tensor(
                [self.data[i][1] for i in range(0, len(self.data), 2)], dtype=float32
            ).to(device)
        ).reshape(-1)
        # print(tensor([self.data[i][1] for i in range(0, len(self.data), 2)],dtype=float32), outputs, tensor([self.data[i][1] for i in range(1, len(self.data), 2)]))
        loss = self.criterion(
            outputs,
            tensor(
                [self.data[i][1] for i in range(1, len(self.data), 2)], dtype=float32
            ).to(device),
        )
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.data = []

    def get_model(self):
        return self.model.state_dict()

    def load_model(self, model):
        if model is not None:
            self.model.load_state_dict(model)

    def update_expected_values(self, hand):
        for card in cards:
            self.expected_values[self.my_index][card] = hand[card]

    def get_card(self, intended_card, hand):
        self.turns += 1
        if self.expected_values is None:
            self.gen_initial_expected_values(hand)
        self.update_expected_values(hand)
        # using get_card strategy from smater_simple_agent
        current_card = cards.index(intended_card)
        future_cards = [
            cards[(current_card + i * self.num_players) % 13] for i in range(1, 14)
        ]
        for card in future_cards[::-1]:
            if hand[card] > 0:
                for _ in range(hand[card]):
                    self.in_pile.append(card)
                return card, hand[card]
        random_chosen = [card for card in hand if hand[card] > 0][0]
        for _ in range(hand[random_chosen]):
            self.in_pile.append(random_chosen)
        return random_chosen, hand[random_chosen]

    def load_model(self, model):
        if model is not None:
            self.model.load_state_dict(model)
    
    def save_model(self):
        return self.model.state_dict()

    def get_call_bs(self, player_index, card, card_amt, hand):
        if len(self.data) > 0 and self.data[-1][0] == "data":
            self.data.pop()
        if self.expected_values is None:
            self.gen_initial_expected_values(hand)
        for _ in range(card_amt):
            self.in_pile.append(card)
        self.update_expected_values(hand)
        d = (
            [self.expected_values[player_index][card]] 
            + [hand[card]]
            + [self.hand_sizes[(player_index - self.my_index) % 4]]
            + [card_amt]
            + [len(self.in_pile)]
            + [self.turns]
        )
        if self.do_training:
            self.data.append(("data", d))
        self.hand_sizes[(player_index - self.my_index) % 4] -= card_amt
        model_result = self.model.forward(tensor([d], dtype=float32).to(device))[0]
        if self.do_fancy:
            val = sigmoid(model_result).item()
            val = 2 * (1 / (1 + math.exp(-self.hand_sizes[(player_index - self.my_index) % 4]))) - 1
            f = self.hand_sizes[0] - self.hand_sizes[(player_index - self.my_index) % 4] - len(self.in_pile)
            f = max(min(f, -5),10)
            val *= (self.required_confidence * 10) / (1.99 * f + 10)
            call = model_result.item() > val
        else:
            call = model_result.item() > self.required_confidence
        # print(call)
        self.last_caller = player_index
        if self.hand_sizes[(player_index - self.my_index) % 4] <= 0:
            call = True
        if hand[card] > 4 - card_amt:
            call = True
        return call

    def give_info(self, player_indexes_picked_up):
        for card in self.in_pile:
            for player_index in player_indexes_picked_up:
                self.expected_values[player_index][card] += 1 / len(
                    player_indexes_picked_up
                )

        num_cards = len(self.in_pile)
        for player in player_indexes_picked_up:
            if num_cards % len(player_indexes_picked_up) > 0:
                self.hand_sizes[(player - self.my_index) % 4] += 1
                num_cards -= 1
            self.hand_sizes[(player - self.my_index) % 4] += len(self.in_pile) // len(player_indexes_picked_up)

        self.in_pile = []

    def give_full_info(self, was_bs):
        if self.do_training is False:
            return
        if len(self.data) > 0 and self.data[-1][0] == "label":
            return
        if len(self.data) == 0:
            return
        # print(was_bs)
        if was_bs:
            self.data.append(("label", 1))
        else:
            self.data.append(("label", 0))
        if (len(self.data) // 2) % self.train_every == 0:
            self.train()

    def reset(self):
        self.expected_values = None
        self.in_pile = []
        self.turns = 0
        self.hand_sizes = [(52 * self.num_decks) // self.num_players] * self.num_players

    def give_winner(self, winner):
        pass
