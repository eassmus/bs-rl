from agents.random_agent import RandomAgent
from agents.simple_agent import SimpleAgent
from agents.smarter_simple_agent import SmartSimpleAngent
from agents.aggressive_agent import AggressiveAgent
from env import BSEnv

wins = [0] * 4    
env = BSEnv(agent_types = [SimpleAgent, SmartSimpleAngent])
for _ in range(1000):
    env.reset()
    game_results = env.run_game()
    wins[game_results.winner] += 1

print(wins)