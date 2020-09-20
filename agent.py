from tictactoe_env import TicTacToe
import numpy as np
from collections import defaultdict
from matplotlib import pyplot as plt
import pickle

POTENTIAL_POS = np.array([i for i in range(1, 10)])

class Agent:

    def __init__(self, alpha, gamma, epsilon):
        self.Q = defaultdict(float)
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon

    def update(self, state, action, reward, next_state, done):
        hash_state = self._hashState(state)

        if done:
            advantage = reward - (self.Q[(hash_state, action)])
        else:
            max_q_next, best_action_next = self._maxQValue(next_state)
            advantage = ( reward + self.gamma * max_q_next - (self.Q[(hash_state, action)]) )
        self.Q[(hash_state, action)] = self.Q[(hash_state, action)] + self.alpha * advantage

    def _hashState(self, state):
        return ''.join( [ str(int(i)) for i in np.ravel(state) ] )

    def _validPositions(self, state):
        return np.ravel(np.nonzero(POTENTIAL_POS * np.ravel(state == 0)) + np.array(1))

    # Given a state, return the action and the associated max q-value
    def _maxQValue(self, state):
        max_q = float('-inf')
        hash_state = self._hashState(state)
        positions = self._validPositions(state)
        
        best_action = None
        
        for pos in positions:
            if self.Q[(hash_state, pos)] > max_q:
                max_q = self.Q[(hash_state, pos)]
                best_action = pos
        if best_action is None:
            print(hash_state)
            best_action = positions[0]
        return max_q, best_action

    def act(self, state):
        pos = self._validPositions(state)
        if np.random.rand() < self.epsilon:
            return np.random.choice(pos)
        else:
            _, best_action = self._maxQValue(state)
            return best_action

# Plots the data with a horizontal line as the max value in the data
def plot(i, data, final_plot_i = 1000, plot_frequency = 10, player=1):
    if i > 0 and i % plot_frequency == 0:
        print(f'EP[{i}]: {data[-1]}')
    
    if i == final_plot_i:
        plt.plot(data)
        plt.title('Player {} | {} EP'.format(player, i))
        plt.xlabel('EP')
        plt.ylabel('Reward')
        plt.show()

def evaluate_greedy_policy(agent, env, niter=100):
    agent.epsilon = 0
    env.win_count['X'] = 0
    env.win_count['O'] = 0
    env.draw_count = 0
    for n in range(niter):
        state = env.reset()
        rewards = []
        done = False
        while not done:
            action = agent.act(state)
            state, reward, done = env.step(action)
            rewards.append(reward)

    return env.win_count['X'] / niter, env.draw_count / niter, env.win_count['O'] / niter

def evaluate_self_play(agent1, agent2, env, niter=100, greedy=True):
    if greedy:
        agent1.epsilon = 0
        agent2.epsilon = 0
    rewards_list = []
    rewards_list2 = []
    env.draw_count = 0
    env.win_count = {
        'X': 0,
        'O': 0
    }
    for n in range(niter):
        state = env.reset()
        reward1 = 0
        reward2 = 0
        done = False
        while not done:
            action = agent1.act(state)
            state, reward, done = env.step_self_play(action, action_player=1, step_player=2)
            reward1 += reward
            if done:
                break

            action = agent2.act(state)
            state, reward, done = env.step_self_play(action, action_player=2, step_player=1)
            reward2 += reward

        rewards_list.append(reward1)
        rewards_list2.append(reward2)

    win1, win2, tie = env.win_count['X'], env.win_count['O'], env.draw_count
    total = np.sum([env.win_count['X'], env.win_count['O'], env.draw_count])
    return win1 / total, win2 / total, tie / total

def save_model(agent, filename = 'qtable'):
    with open(filename, 'wb') as f:
        pickle.dump(agent.Q, f, protocol=pickle.HIGHEST_PROTOCOL)

def load_model(agent, filename='qtable'):
    with open(filename, 'rb') as f:
        agent.Q = pickle.load(f)
    
def plot_counts(p1, p2, draw, freq):
    n = np.array(range(len(p1))) * freq # adjust to show episode count
    plt.plot((n), p1, label='Player 1 Win', color='blue', linewidth=1, linestyle=':')
    plt.plot((n), p2, label='Player 2 Win', color='red', linewidth=1, linestyle=':')
    plt.plot((n), draw, label='Draw', color='green', linewidth=1)
    plt.title('Battle Statistics')
    plt.xlabel('Episodes')
    plt.ylabel('Percentage (%)')
    plt.legend()
    plt.show()

def plot_agent_counts(p1):
    n = list(range(len(p1)))
    plt.plot((n), p1, label='Agent Win', color='blue', linewidth=1, linestyle=':')
    plt.title('Battle Statistics')
    plt.xlabel('EP')
    plt.ylabel('Percentage (%)')
    plt.legend()
    plt.show()


def train_selfplay():
    """
    This method trains two agents against each-other (self-play)
    """
    EPISODES = 5000
    STATS_EVERY_N_EP = int(EPISODES / 20)
    PLOT_SMOOTH = 50
    EVAL_SELF_PLAY_ITERATIONS = 1
    EVAL_EVERY_N_EP = 5
    env = TicTacToe()
    eval_env = TicTacToe()
    INIT_EPS = 1
    TERM_EPS = 0
    agent = Agent(alpha=1e-1, gamma=0.9, epsilon=INIT_EPS)
    load_model(agent, 'minmax_trained_{}ep'.format(1000)) # Load existing model if necessary

    agent2 = Agent(alpha=1e-1, gamma=0.9, epsilon=INIT_EPS)
    # load_model(agent2, 'minmax_trained_{}ep'.format(1000)) # Load existing model if necessary

    # Statistics
    episode_rewards = []
    episode_rewards2 = []
    player1_win_trace = []
    player2_win_trace = []
    draw_trace = []

    for i in range(EPISODES + 1):
        state = env.reset()
        rewards = 0
        rewards2 = 0        
        done = False
        agent.epsilon -= (INIT_EPS - TERM_EPS) / EPISODES
        agent2.epsilon -= (INIT_EPS - TERM_EPS) / EPISODES
        
        action2 = -1 # dummy for updates
        reward2 = -1 # dummy
        while not done:
            # For self-play, the next_state of player 1 is the state of player 2
            action = agent.act(state)
            state2, reward, done = env.step_self_play(action, action_player=1, step_player=2) # player1 takes response action, env steps, returns reward for player 2
            if action2 != -1:
                agent2.update(state_temp, action2, reward, state2, done) # updates are reversed, because we need to see the opponent's outcome
            rewards2 += reward
            if done:
                break
            
            state_temp = state2
            action2 = agent2.act(state2)
            state2, reward2, done = env.step_self_play(action2, action_player=2, step_player=1)
            agent.update(state, action, reward2, state2, done)
            state = state2
            rewards += reward2

        episode_rewards.append(rewards)
        episode_rewards2.append(rewards2)
        
        if i % STATS_EVERY_N_EP == 0 and i > 0:
            print('EP: {}'.format(i))
            print('Player 1 | latest {} episodes Mean Reward: {}'.format(STATS_EVERY_N_EP, np.mean(episode_rewards[-STATS_EVERY_N_EP:])))
            print('Player 2 | latest {} episodes Mean Reward: {}'.format(STATS_EVERY_N_EP, np.mean(episode_rewards2[-STATS_EVERY_N_EP:])))

        if i % EVAL_EVERY_N_EP == 0 and i > 0:
            win1_fra, win2_frac, tie_frac = evaluate_self_play(agent, agent2, eval_env, EVAL_SELF_PLAY_ITERATIONS, greedy=False)
            player1_win_trace.append(win1_fra * 100)
            player2_win_trace.append(win2_frac * 100)
            draw_trace.append(tie_frac * 100)

    mean_episode_rewards = np.convolve(episode_rewards, np.ones((PLOT_SMOOTH,)) / PLOT_SMOOTH, mode='valid').tolist()
    mean_episode_rewards2 = np.convolve(episode_rewards2, np.ones((PLOT_SMOOTH,)) / PLOT_SMOOTH, mode='valid').tolist()
    player1_win_trace = np.convolve(player1_win_trace, np.ones((PLOT_SMOOTH,)) / PLOT_SMOOTH, mode='valid').tolist()
    player2_win_trace = np.convolve(player2_win_trace, np.ones((PLOT_SMOOTH,)) / PLOT_SMOOTH, mode='valid').tolist()
    draw_trace = np.convolve(draw_trace, np.ones((PLOT_SMOOTH,)) / PLOT_SMOOTH, mode='valid').tolist()

    plot(i, mean_episode_rewards, EPISODES)
    plot(i, mean_episode_rewards2, EPISODES, player=2)
    plot_counts(player1_win_trace, player2_win_trace, draw_trace, EVAL_EVERY_N_EP)
    save_model(agent, 'selfplayX_{}ep'.format(EPISODES))
    save_model(agent2, 'selfplayO_{}ep'.format(EPISODES))

    print('\n\nPost-Training Self-play Evaluation')
    win1_fra, win2_frac, tie_frac = evaluate_self_play(agent, agent2, env, 100)
    print("Player1: {}% | Player2: {}% | Tie: {}%".format(win1_fra * 100, win2_frac * 100, tie_frac * 100))


def train_original():
    """
    This method trains the agent against the original min-max opponent
    """
    EPISODES = 1000
    STATS_EVERY_N_EP = 100
    PLOT_SMOOTH = 20
    env = TicTacToe()
    max_eps = 1
    min_eps = 0
    agent = Agent(alpha=1e-1, gamma=0.9, epsilon=max_eps)
    
    # Statistics
    episode_rewards = []
    win_trace = []
    draw_trace = []

    for i in range(EPISODES + 1):
        state = env.reset()
        rewards = 0
        done = False
        agent.epsilon -= (max_eps - min_eps) / EPISODES
        
        while not done:
            action = agent.act(state)
            next_state, reward, done = env.step(action)
            agent.update(state, action, reward, next_state, done)
            rewards += reward
            state = next_state

        episode_rewards.append(rewards)

        if i % STATS_EVERY_N_EP == 0 and i > 0:
            print('EP{} | Agent Reward: {}'.format(i, episode_rewards[-1]))

    mean_episode_rewards = np.convolve(episode_rewards, np.ones((PLOT_SMOOTH,)) / PLOT_SMOOTH, mode='valid').tolist()
    plot(i, mean_episode_rewards, EPISODES)
    save_model(agent, 'minmax_trained_{}ep'.format(EPISODES))

    # TEST PHASE
    print('\n\nEvaluating learnt policy')
    win_frac, tie_frac, lose_frac = evaluate_greedy_policy(agent, env, 1)
    print("Win: {}% | Tie: {}% | Lose: {}%".format(win_frac * 100, tie_frac * 100, lose_frac * 100))

if __name__ == '__main__':
    train_original()
    train_selfplay()
    


    