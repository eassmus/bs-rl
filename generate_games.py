from trainingenv import TrainingBSEnv
from agents.smart_simple_agent_bs_learner import SmartSimpleAgentBSLearner
from agents.smarter_simple_agent import SmartSimpleAgent
from agents.simple_agent import SimpleAgent
import game_metrics as gm

import matplotlib.pyplot as plt

import pickle

agent_args = {
    "num_decks": 1,
    "train_every" : 20,
    "learning_rate": 0.01,
    "required_confidence": 0.65
}

env = TrainingBSEnv(agent_types=[SmartSimpleAgent, SmartSimpleAgent, SmartSimpleAgent, SmartSimpleAgentBSLearner], agent_args=[agent_args, agent_args, agent_args, agent_args])

total_results = []

def render_graphs():
    plt.close("all")
    gm.plt_avg_delta_cards(total_results, 0)
    gm.plt_avg_delta_cards(total_results, 1)
    gm.plt_avg_delta_cards(total_results, 2)
    gm.plt_avg_delta_cards(total_results, 3)
    gm.plt_win_rate(total_results, 0)
    gm.plt_win_rate(total_results, 1)
    gm.plt_win_rate(total_results, 2)
    gm.plt_win_rate(total_results, 3)
    gm.plt_bs_called_accuracy_not_free(total_results, 0)
    gm.plt_bs_called_accuracy_not_free(total_results, 1)
    gm.plt_bs_called_accuracy_not_free(total_results, 2)
    gm.plt_bs_called_accuracy_not_free(total_results, 3)
    gm.plt_bs_accuracy(total_results, 0)
    gm.plt_bs_accuracy(total_results, 1)
    gm.plt_bs_accuracy(total_results, 2)
    gm.plt_bs_accuracy(total_results, 3)
    gm.plt_bs_call_rate(total_results, 0)
    gm.plt_bs_call_rate(total_results, 1)
    gm.plt_bs_call_rate(total_results, 2)
    gm.plt_bs_call_rate(total_results, 3)
    gm.plt_true_bs_ratio(total_results, player_indexes=[0,1,2,3])
    gm.plt_duration(total_results)
    plt.pause(0.5)
    plt.show()

NUM_EPISODES = 5000

for episode in range(1,1+NUM_EPISODES):
    results = env.run_game()
    env.rotate_dealer()
    total_results.append(results)
    if episode % 100 == 0:
        print(episode)

render_graphs()
plt.show()

with open("total_results_ssabsc.pkl", "wb") as f:
    pickle.dump(total_results, f)
