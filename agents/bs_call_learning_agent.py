from agents.agent import Agent
from torch import nn
from torch import optim
from random import random as rand
from torch import tensor
from torch import float32

cards = ["A", "2", "3", "4", "5", "6", "7", "8", "9", "10", "J", "Q", "K"]


class _Model(nn.Module):
    def __init__(self, input_size, output_size):
        super(_Model, self).__init__()
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.fc1 = nn.Linear(input_size, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, output_size)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.relu(out)
        out = self.fc3(out)
        out = self.sigmoid(out)
        return out


class BSCallLearningAgent(Agent):
    def __init__(self, my_index, num_players, agent_args):
        #print("BS INIT")
        self.my_index = my_index
        self.num_players = num_players
        self.num_decks = agent_args["num_decks"]
        self.expected_values = None  # generated later when we are given our first hand
        self.in_pile = []
        self.model = _Model(7, 1)
        self.data = []
        self.hand_sizes = [(52 * self.num_decks) // self.num_players] * self.num_players
        self.last_caller = None
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.SGD(
            self.model.parameters(), lr=agent_args["learning_rate"]
        )
        self.train_every = agent_args["train_every"]
        self.turns = 0
        self.training = True
        if "training" in agent_args and agent_args["training"] is not None:
            self.training = agent_args["training"]
        self.required_confidence = agent_args["required_confidence"]
        if "load_model" in agent_args and agent_args["load_model"] is not None:
            self.load_model(agent_args["load_model"])

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
        if self.training is False:
            return
        outputs = self.model(
            tensor(
                [self.data[i][1] for i in range(0, len(self.data), 2)], dtype=float32
            )
        ).reshape(-1)
        # print(tensor([self.data[i][1] for i in range(0, len(self.data), 2)],dtype=float32), outputs, tensor([self.data[i][1] for i in range(1, len(self.data), 2)]))
        loss = self.criterion(
            outputs,
            tensor(
                [self.data[i][1] for i in range(1, len(self.data), 2)], dtype=float32
            ),
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

    def get_call_bs(self, player_index, card, card_amt, hand):
        if len(self.data) > 0 and self.data[-1][0] == "data":
            self.data.pop()
        if self.expected_values is None:
            self.gen_initial_expected_values(hand)
        for _ in range(card_amt):
            self.in_pile.append(card)
        self.update_expected_values(hand)
        d = (
            [self.expected_values[player_index][card]] +
            [sum(self.expected_values[(player_index + p) % 4][card] + card_amt for p in range(1, self.num_players))]
            + [hand[card]]
            + [self.hand_sizes[(player_index - self.my_index) % 4]]
            + [card_amt]
            + [len(self.in_pile)]
            + [self.turns]
        )
        if self.training:
            self.data.append(("data", d))
        self.hand_sizes[(player_index - self.my_index) % 4] -= card_amt
        model_result = self.model.forward(tensor([d], dtype=float32))[0]
        call = model_result.item() > self.required_confidence
        # print(call)
        self.last_caller = player_index
        if hand[card] > 4 - card_amt:
            call = True
        return call

    def give_info(self, player_indexes_picked_up):
        for card in self.in_pile:
            for player_index in player_indexes_picked_up:
                self.expected_values[player_index][card] += 1 / len(
                    player_indexes_picked_up
                )

        for player in player_indexes_picked_up:
            self.hand_sizes[(player - self.my_index) % 4] += len(self.in_pile) / 3

        self.in_pile = []

    def give_full_info(self, was_bs):
        if self.training is False:
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
