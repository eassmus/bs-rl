from env import BSEnv

from agents.human_agent import HumanAgent

from agents.dqn_agent import DQNAgent
from agents.ppo_agent_bs_learner import PPOAgentBSLearner
from agents.ppo_agent import PPOAgent
from agents.smarter_simple_agent import SmartSimpleAgent
from agents.simple_agent import SimpleAgent
from agents.smart_simple_agent_bs_learner import SmartSimpleAgentBSLearner
from agents.random_agent import RandomAgent
from agents.smart_expected_value_agent import SmartExpectedValueAgent

import random

import torch

dqn_args = {
    "num_decks": 1,
    "load_model": torch.load("model550000.pt"),
    "ep_decay": 500,
    "training": False,
    "batch_size": 64
}

bs_learner_args = {
    "num_decks": 1,
    "train_every": 3,
    "required_confidence": 0.9,
    "learning_rate": 0.01,
    "do_training_bs": False,
    "load_model_bs": torch.load("model_bs_trained.pt"),
    "do_fancy": False
}

baa = {
    "num_decks": 1
}

ppo_bs_args = {"do_training": False}
ppo_bs_args.update(bs_learner_args)

dqn_args.update(bs_learner_args)

dqn_agent = (DQNAgent, dqn_args)
ppo_bs_agent = (PPOAgentBSLearner, ppo_bs_args)
ppo_agent = (PPOAgent, { "do_training" : False })
smart_simple_bs = (SmartSimpleAgentBSLearner, bs_learner_args)
smart_simple = (SmartSimpleAgent, baa)
simple = (SimpleAgent, baa)
random_agent = (RandomAgent, {"random_chance": 0.1})
smart_expected_value = (SmartExpectedValueAgent, baa)

def play_game(agents):
    env = BSEnv([HumanAgent] + [a[0] for a in agents], [baa] + [a[1] for a in agents],print_callers=True)
    print('\n' * 20)
    env.run_game()

play_game([smart_simple_bs, smart_simple_bs, smart_simple_bs])