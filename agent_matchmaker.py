"""Rank agents based on simple multiplayer elo system."""
import random

from tqdm import tqdm

from agents.random_agent import RandomAgent
from agents.simple_agent import SimpleAgent
from agents.smarter_simple_agent import SmartSimpleAgent
from agents.aggressive_agent import AggressiveAgent
from agents.ppo_agent import PPOAgent

from env import BSEnv

# elo system based on the formula from Reiner Knizia’s Samurai game
def expected_score(ratings, player_index, B=1/400):
    Za = 10 ** (B * ratings[player_index])
    sum_Z = sum(10 ** (B * ratings[i]) for i in range(len(ratings)))
    return Za / sum_Z

def update_ratings(ratings, outcomes, K=32):
    new_ratings = ratings.copy()
    num_players = len(ratings)
    expected_scores = [expected_score(ratings, i) for i in range(num_players)]
    for i in range(num_players):
        new_ratings[i] += K * (outcomes[i] - expected_scores[i])
    
    return new_ratings

class AgentPlayer:
    def __init__(self, agent_type, agent_args = [], initial_rating=1200):
        self.agent_type = agent_type  
        self.rating = initial_rating 
        self.agent_args = agent_args

    def agent_indentifier(self):
        return self.agent_type.__name__ + "  :  " + str(self.agent_args)

    def __repr__(self):
        return f"<AgentPlayer(type={self.agent_type.__name__}, rating={self.rating})>"

class MatchMaker:
    def __init__(self, agent_types, num_agents_per_type=20, agent_args=None, initial_rating=1200):
        self.agent_types = agent_types
        self.agents = [] 
        self.agent_args = agent_args
        
        # create a bunch of agents of each agent type, each with their own rating (not shared)
        i = 0
        for agent_type in agent_types:
            for _ in range(num_agents_per_type):
                agent_instance = AgentPlayer(agent_type, initial_rating=initial_rating, agent_args=agent_args[i])
                self.agents.append(agent_instance)
            i += 1

        self.wins = {a.agent_indentifier(): 0 for a in self.agents} 


    def form_groups(self):
        """Form groups of 4 randomly between agents."""

        #TODO: form groups based on elo, ie. pair different agent types that are similar in elo
        random.shuffle(self.agents)
        groups = []
        
        for i in range(0, len(self.agents), 4):
            group = self.agents[i:i + 4]
            if len(group) == 4:
                groups.append(group)
        
        return groups

    def run_matches(self, groups, matches=3):
        """Run matches within each group and updates ratings for each agent."""
        for group in groups:
            wins = [0] * len(group)
            
            # run some matches in the same group
            for _ in range(matches):
                env = BSEnv(agent_types=[agent.agent_type for agent in group], agent_args=[agent.agent_args for agent in group])
                env.reset()
                game_results = env.run_game()
                winner_index = game_results.winner
                wins[winner_index] += 1 

            # update wins per agent type
            for j, agent in enumerate(group):
                self.wins[agent.agent_indentifier()] += wins[j]
            
            # update ratings based on expected value
            normalized_outcomes = [win / sum(wins) for win in wins]
            current_ratings = [agent.rating for agent in group]
            new_ratings = update_ratings(current_ratings, normalized_outcomes)

            for j, agent in enumerate(group):
                agent.rating = new_ratings[j]

    def report_results(self):
        """Report the results with average ratings and wins per agent type."""
        type_ratings = {a.agent_indentifier(): [] for a in self.agents}
        
        for agent in self.agents:
            type_ratings[agent.agent_indentifier()].append(agent.rating)
        
        average_ratings = {agent_type: sum(ratings) / len(ratings)
                           for agent_type, ratings in type_ratings.items()}
        
        elo_and_wins = [
            (a, average_ratings[a], self.wins[a])
            for a in set(a.agent_indentifier() for a in self.agents)
        ]

        sorted_elo_and_wins = sorted(elo_and_wins, key=lambda x: x[1], reverse=True)
        
        return sorted_elo_and_wins

    def simulate(self, num_iterations=20, matches_per_group=3):
        for _ in tqdm(range(num_iterations)):
            groups = self.form_groups()
            self.run_matches(groups, matches=matches_per_group)

def matchmake(agent_types, num_agents_per_type = 9, num_iterations = 1000, matches_per_group = 3, agent_args = None, seed = None):
    if seed is not None:
        random.seed(seed)
    matchmaker = MatchMaker(agent_types,num_agents_per_type = num_agents_per_type, agent_args=agent_args)

    matchmaker.simulate(num_iterations,matches_per_group)
    results = matchmaker.report_results()

    print(f"Elo Ratings and Total Wins (Sorted by Elo):")
    max_name_length = max(len(agent_name) for agent_name, _, _ in results)
    for agent_name, elo, win in results:
        print(f"{agent_name} {' ' * (max_name_length - len(agent_name))} Elo: {elo:.2f}, Wins: {win}")


# example
if __name__ == "__main__":
    matchmake([SimpleAgent, SmartSimpleAgent, PPOAgent, RandomAgent],agent_args=[{}, {}, {}, {}])