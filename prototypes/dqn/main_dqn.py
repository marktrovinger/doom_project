import os
import numpy as np
from vizdoom import *
from dqn_agent import DQNAgent
import skimage.color, skimage.transform
import itertools as it
#from util import make_env, plot_learning_curve

config_file_path = '../../scenarios/simpler_basic.cfg'
resolution = (30, 45)

def initialize_vizdoom(config_file_path):
    print("Initializing doom...")
    game = DoomGame()
    game.load_config(config_file_path)
    game.set_window_visible(False)
    game.set_mode(Mode.PLAYER)
    game.set_screen_format(ScreenFormat.GRAY8)
    game.set_screen_resolution(ScreenResolution.RES_640X480)
    game.init()
    print("Doom initialized.")
    return game

def preprocess(img):
    img = skimage.transform.resize(img, resolution)
    img = img.astype(np.float32)
    return img

if __name__=='__main__':
    
    game = initialize_vizdoom(config_file_path)
    
    n = game.get_available_buttons_size()
    actions = [list(a) for a in it.product([0, 1], repeat=n)]
    best_score = -np.inf
    load_checkpoint = False
    n_games = 500
    agent = DQNAgent(gamma=0.99, epsilon=1.0, lr=0.001, input_dims=resolution,
                    n_actions=len(actions), mem_size=50000, eps_min=0.1,
                    batch_size=32, replace=1000, eps_dec=1e-5, chkpoint_dir='models/',
                    algo='DQNAgent',env_name='Doom-Basic')
    
    if load_checkpoint:
        agent.load_models()

    fname = agent.algo + '_' + agent.env_name + '_lr' + str(agent.lr) + '_' + str(n_games) + 'games'
    figure_file = 'plots/' + fname + '.png'

    n_steps = 0
    scores, eps_history, steps_array = [], [], []

    for i in range(n_games):
        done = False
        score = 0
        #observation = env.reset()
        game.new_episode()

        while not game.is_episode_finished():
            observation = preprocess(game.get_state().screen_buffer)
            action = agent.choose_action(observation)
            observation_, reward, done, info = env.step(action)
            score += reward

            if not load_checkpoint:
                agent.store_transition(observation, action, reward, observation_, int(done))
                agent.learn()
            observation = observation_
            n_steps += 1
        scores.append(score)
        steps_array.append(n_steps)

        avg_score = np.mean(scores[-100:])
        print('episode ', i, 'score: ', score, 'average score %.1f best score %.1f epsilon %.2f' % (avg_score, best_score, agent.epsilon), 
            'steps ', n_steps)

        if avg_score > best_score:
            if not load_checkpoint:
                agent.save_models()
            best_score = avg_score

        eps_history.append(agent.epsilon)

    #plot_learning_curve(steps_array, scores, eps_history, figure_file)