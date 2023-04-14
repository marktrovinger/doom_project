# Copy of trainer_atari to test the eval() of vizdoom - NB training is not done here, just a shortcut to eval

"""
The MIT License (MIT) Copyright (c) 2020 Andrej Karpathy

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""

"""
Simple training loop; Boilerplate that could apply to any arbitrary neural network,
so nothing in this file really has anything to do with GPT specifically.
"""

import math
import logging

from tqdm import tqdm
import numpy as np

import torch
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data.dataloader import DataLoader

logger = logging.getLogger(__name__)

from atari.mingpt.utils import sample
import atari_py
from collections import deque
import random
import cv2
import torch
from PIL import Image

class TrainerConfig:
    # optimization parameters
    max_epochs = 1
    batch_size = 64
    learning_rate = 3e-4
    betas = (0.9, 0.95)
    grad_norm_clip = 1.0
    weight_decay = 0.1 # only applied on matmul weights
    # learning rate decay params: linear warmup followed by cosine decay to 10% of original
    lr_decay = False
    warmup_tokens = 375e6 # these two numbers come from the GPT-3 paper, but may not be good defaults elsewhere
    final_tokens = 260e9 # (at what point we reach 10% of original LR)
    # checkpoint settings
    ckpt_path = '/Users/perusha/git_repos/decision-transformer/runs/'
    num_workers = 0 # for DataLoader

    def __init__(self, **kwargs):
        for k,v in kwargs.items():
            setattr(self, k, v)

class Trainer:

    def __init__(self, model, train_dataset, test_dataset, config):
        self.model = model
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.config = config

        # take over whatever gpus are on the system
        self.device = 'cpu'
        if torch.cuda.is_available():
            self.device = torch.cuda.current_device()
            self.model = torch.nn.DataParallel(self.model).to(self.device)

    def save_checkpoint_original(self):
        # DataParallel wrappers keep raw model object in .module attribute
        raw_model = self.model.module if hasattr(self.model, "module") else self.model
        logger.info("saving %s", self.config.ckpt_path)
        torch.save(raw_model.state_dict(), f"{self.config.ckpt_path}dt_model")

    def save_checkpoint(self):
        model = self.model.module if hasattr(self.model, "module") else self.model
        torch.save({
            'tok_emb': model.tok_emb.state_dict(),
            'pos_emb': model.pos_emb,
            'global_pos_emb': model.global_pos_emb,
            'drop': model.drop.state_dict(),
            'blocks': model.blocks.state_dict(),
            'head': model.head.state_dict(),
            'state_encoder': model.state_encoder.state_dict(),
            'ret_emb': model.ret_emb.state_dict(),
            'action_embeddings': model.action_embeddings.state_dict(),
        }, f"{self.config.ckpt_path}dt_model")
        logger.info("saving %s", self.config.ckpt_path)

    def load_checkpoint(self):
        modules = torch.load(f"{self.config.ckpt_path}dt_model")
        self.model.tok_emb.load_state_dict(modules['tok_emb'])
        self.model.pos_emb=modules['pos_emb']
        self.model.global_pos_emb = modules['global_pos_emb']
        self.model.drop.load_state_dict(modules['drop'])
        self.model.blocks.load_state_dict(modules['blocks'])
        self.model.head.load_state_dict(modules['head'])
        self.model.state_encoder.load_state_dict(modules['state_encoder'])
        self.model.ret_emb.load_state_dict(modules['ret_emb'])
        self.model.action_embeddings.load_state_dict(modules['action_embeddings'])

        return self.model

    def train(self):
        model= self.load_checkpoint()
        config = self.config
        raw_model = model.module if hasattr(self.model, "module") else model
        optimizer = raw_model.configure_optimizers(config)

        def run_epoch(split, epoch_num=0):
            is_train = split == 'train'
            model.train(is_train)
            data = self.train_dataset if is_train else self.test_dataset
            loader = DataLoader(data, shuffle=True, pin_memory=True,
                                batch_size=config.batch_size,
                                num_workers=config.num_workers)

            losses = []
            pbar = tqdm(enumerate(loader), total=len(loader)) if is_train else enumerate(loader)
            for it, (x, y, r, t) in pbar:

                # place data on the correct device
                x = x.to(self.device)
                y = y.to(self.device)
                r = r.to(self.device)
                t = t.to(self.device)

                # forward the model
                with torch.set_grad_enabled(is_train):
                    # logits, loss = model(x, y, r)
                    logits, loss = model(x, y, y, r, t)
                    loss = loss.mean() # collapse all losses if they are scattered on multiple gpus
                    losses.append(loss.item())

                if is_train:

                    # backprop and update the parameters
                    model.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_norm_clip)
                    optimizer.step()

                    # decay the learning rate based on our progress
                    if config.lr_decay:
                        self.tokens += (y >= 0).sum() # number of tokens processed this step (i.e. label is not -100)
                        if self.tokens < config.warmup_tokens:
                            # linear warmup
                            lr_mult = float(self.tokens) / float(max(1, config.warmup_tokens))
                        else:
                            # cosine learning rate decay
                            progress = float(self.tokens - config.warmup_tokens) / float(max(1, config.final_tokens - config.warmup_tokens))
                            lr_mult = max(0.1, 0.5 * (1.0 + math.cos(math.pi * progress)))
                        lr = config.learning_rate * lr_mult
                        for param_group in optimizer.param_groups:
                            param_group['lr'] = lr
                    else:
                        lr = config.learning_rate

                    # report progress
                    pbar.set_description(f"epoch {epoch+1} iter {it}: train loss {loss.item():.5f}. lr {lr:e}")

            if not is_train:
                test_loss = float(np.mean(losses))
                logger.info("test loss: %f", test_loss)
                return test_loss

        # best_loss = float('inf')
        best_return = -float('inf')
        self.tokens = 0 # counter used for learning rate decay

        for epoch in range(config.max_epochs):

            # run_epoch('train', epoch_num=epoch)
            # if self.test_dataset is not None:
            #     test_loss = run_epoch('test')

            # # supports early stopping based on the test loss, or just save always if no test set is provided
            # good_model = self.test_dataset is None or test_loss < best_loss
            # if self.config.ckpt_path is not None and good_model:
            #     best_loss = test_loss
            #     self.save_checkpoint()
            if self.config.ckpt_path is not None:
                self.save_checkpoint()

            # -- pass in target returns
            if self.config.model_type == 'naive':
                eval_return = self.get_returns(0)
            elif self.config.model_type == 'reward_conditioned':
                if self.config.game == 'Breakout':
                    eval_return = self.get_returns(90)
                elif self.config.game == 'Seaquest':
                    eval_return = self.get_returns(1150)
                elif self.config.game == 'Qbert':
                    eval_return = self.get_returns(14000)
                elif self.config.game == 'Pong':
                    eval_return = self.get_returns(20)
                elif self.config.game == 'VizdoomHealthGatheringSupreme':
                    eval_return = self.get_returns(400)
                else:
                    raise NotImplementedError()
            else:
                raise NotImplementedError()

    def get_returns(self, ret):
        self.model.train(False)
        # args=Args(self.config.game.lower(), self.config.seed)
        # PMMOD - remove lower()
        args=Args(self.config.game, self.config.seed)
        env = Env_doom(args)
        env.eval()

        T_rewards, T_Qs = [], []
        done = True
        for i in range(10):
            state = env.reset()
            state = state.type(torch.float32).to(self.device).unsqueeze(0).unsqueeze(0)
            rtgs = [ret]
            # first state is from env, first rtg is target return, and first timestep is 0
            # sampled_action = sample(self.model.module, state, 1, temperature=1.0, sample=True, actions=None,
            #                         rtgs=torch.tensor(rtgs, dtype=torch.long).to(self.device).unsqueeze(0).unsqueeze(-1),
            #                         timesteps=torch.zeros((1, 1, 1), dtype=torch.int64).to(self.device))
            # self.model.module - errors as module does not exist...
            sampled_action = sample(self.model, state, 1, temperature=1.0, sample=True, actions=None,
                                    rtgs=torch.tensor(rtgs, dtype=torch.long).to(self.device).unsqueeze(0).unsqueeze(-1),
                                    timesteps=torch.zeros((1, 1, 1), dtype=torch.int64).to(self.device))
            j = 0
            all_states = state
            actions = []
            while True:
                if done:
                    state, reward_sum, done = env.reset(), 0, False
                action = sampled_action.cpu().numpy()[0,-1]
                actions += [sampled_action]
                state, reward, done = env.step(action)
                reward_sum += reward
                j += 1

                if done:
                    T_rewards.append(reward_sum)
                    break

                state = state.unsqueeze(0).unsqueeze(0).to(self.device)

                all_states = torch.cat([all_states, state], dim=0)

                rtgs += [rtgs[-1] - reward]
                # all_states has all previous states and rtgs has all previous rtgs (will be cut to block_size in utils.sample)
                # timestep is just current timestep
                sampled_action = sample(self.model, all_states.unsqueeze(0), 1, temperature=1.0, sample=True,
                                        actions=torch.tensor(actions, dtype=torch.long).to(self.device).unsqueeze(1).unsqueeze(0),
                                        rtgs=torch.tensor(rtgs, dtype=torch.long).to(self.device).unsqueeze(0).unsqueeze(-1),
                                        timesteps=(min(j, self.config.max_timestep) * torch.ones((1, 1, 1), dtype=torch.int64).to(self.device)))
        env.close()
        eval_return = sum(T_rewards)/10.
        print("target return: %d, eval return: %d" % (ret, eval_return))
        self.model.train(True)
        return eval_return

# Added for evaluating Vizdoom - Start:
import gym
from vizdoom import GameVariable
from gym.envs.registration import register
try:
    register(
        id="VizdoomHealthGatheringSupreme-v0",
        entry_point="vizdoom.gym_wrapper.gym_env_defns:VizdoomScenarioEnv",
        kwargs={"scenario_file": "health_gathering_supreme.cfg"}
    )
except:
    print(f"Registration failed for Vizdoom envs")
    pass

class Env_doom():
    def __init__(self, args):
        self.doom = gym.make("VizdoomHealthGatheringSupreme-v0")
        self.device = args.device
        self.random_seed = args.seed
        self.max_num_frames_per_episode= args.max_episode_length
        self.repeat_action_probability=0  # Disable sticky actions
        self.frame_skip= 0
        self.color_averaging=False
        actions = [0,1,2,3]
        self.actions = dict([i, e] for i, e in zip(range(len(actions)), actions))
        self.lives = 0  # Life counter (used in DeepMind training)
        self.life_termination = False  # Used to check if resetting only from loss of life
        self.window = args.history_length  # Number of frames to concatenate
        self.state_buffer = deque([], maxlen=args.history_length)
        self.training = True  # Consistent with model training mode

    def _get_state(self, obs):
        state = cv2.resize(self.getScreenGrayscale(obs), (84, 84), interpolation=cv2.INTER_LINEAR)
        return torch.tensor(state, dtype=torch.float32, device=self.device).div_(255)

    def _reset_buffer(self):
        for _ in range(self.window):
            self.state_buffer.append(torch.zeros(84, 84, device=self.device))

    def getScreenGrayscale(self, output):
        #Borrowed from atari_lib (dopamine - discrete) - Returns the current observation in grayscale.
        # Copied from https://github.com/KatyNTsachi/Hierarchical-RL/commit/40e995d9ab8cdab396415ea77c9041a53e3acbb5#diff-68e8e61267696ad4397942e952fefc8c22627a0d7203fcf8cf08534a086d6220
        if np.shape(np.shape(output))[0]==3:
            output=cv2.cvtColor(output, cv2.COLOR_BGR2GRAY)
        return output


    def reset(self):
        if self.life_termination:
            self.life_termination = False  # Reset flag
            # self.act(0)  # Use a no-op after loss of life
        else:
            # Reset internals
            self._reset_buffer() # ??
            obs = self.doom.reset()[0]['screen']
            # # Perform up to 30 random no-ops before starting - TBD should we do this for doom??
            # for _ in range(random.randrange(30)):
            #     self.ale.act(0)  # Assumes raw action 0 is always no-op
            #     if self.ale.game_over():
            #         self.ale.reset_game()
            #
        # Process and return "initial" state
        observation = self._get_state(obs)
        self.state_buffer.append(observation)
        self.lives = self.get_lives()
        return torch.stack(list(self.state_buffer), 0)


    def step(self, action):
        # Repeat action 4 times, max pool over last 2 frames
        frame_buffer = torch.zeros(2, 84, 84, device=self.device)
        reward, done = 0, False
        for t in range(4):
            obs, rew, game_over,game_trunc, info = self.doom.step(action)
            reward += rew
            if t == 2:
                frame_buffer[0] = self._get_state(obs['screen'])
            elif t == 3:
                frame_buffer[1] = self._get_state(obs['screen'])
            done = game_over
            if done:
                break
        observation = frame_buffer.max(0)[0]
        self.state_buffer.append(observation)
        # TBD:  Detect loss of life as terminal in training mode
        if self.training:
            lives = self.get_lives()
            if lives < self.lives and lives > 0:  # Lives > 0 for Q*bert
                self.life_termination = not done  # Only set flag when not truly done
                done = True
            self.lives = lives
        # Return state, reward, done
        return torch.stack(list(self.state_buffer), 0), reward, done

    # Uses loss of life as terminal signal
    def train(self):
        self.training = True

    # Uses standard terminal signal
    def eval(self):
        self.training = False

    def action_space(self):
        return len(self.actions)

    def render(self):
        cv2.imshow('screen', self.doom.render())
        cv2.waitKey(1)

    def close(self):
        cv2.destroyAllWindows()

    def get_lives(self):
        game_vars = self.doom.game.get_available_game_variables()
        lives = [i.value for i in game_vars if i == GameVariable.HEALTH]
        return lives[0]
# Vizdoom - End

class Env():
    def __init__(self, args):
        self.device = args.device
        self.ale = atari_py.ALEInterface()
        self.ale.setInt('random_seed', args.seed)
        self.ale.setInt('max_num_frames_per_episode', args.max_episode_length)
        self.ale.setFloat('repeat_action_probability', 0)  # Disable sticky actions
        self.ale.setInt('frame_skip', 0)
        self.ale.setBool('color_averaging', False)
        self.ale.loadROM(atari_py.get_game_path((args.game).lower()))  # ROM loading must be done after setting options
        actions = self.ale.getMinimalActionSet()
        self.actions = dict([i, e] for i, e in zip(range(len(actions)), actions))
        self.lives = 0  # Life counter (used in DeepMind training)
        self.life_termination = False  # Used to check if resetting only from loss of life
        self.window = args.history_length  # Number of frames to concatenate
        self.state_buffer = deque([], maxlen=args.history_length)
        self.training = True  # Consistent with model training mode

    def _get_state(self):
        state = cv2.resize(self.ale.getScreenGrayscale(), (84, 84), interpolation=cv2.INTER_LINEAR)
        return torch.tensor(state, dtype=torch.float32, device=self.device).div_(255)

    def _reset_buffer(self):
        for _ in range(self.window):
            self.state_buffer.append(torch.zeros(84, 84, device=self.device))

    def reset(self):
        if self.life_termination:
            self.life_termination = False  # Reset flag
            self.ale.act(0)  # Use a no-op after loss of life
        else:
            # Reset internals
            self._reset_buffer()
            self.ale.reset_game()
            # Perform up to 30 random no-ops before starting
            for _ in range(random.randrange(30)):
                self.ale.act(0)  # Assumes raw action 0 is always no-op
                if self.ale.game_over():
                    self.ale.reset_game()
        # Process and return "initial" state
        observation = self._get_state()
        self.state_buffer.append(observation)
        self.lives = self.ale.lives()
        return torch.stack(list(self.state_buffer), 0)

    def step(self, action):
        # Repeat action 4 times, max pool over last 2 frames
        frame_buffer = torch.zeros(2, 84, 84, device=self.device)
        reward, done = 0, False
        for t in range(4):
            reward += self.ale.act(self.actions.get(action))
            if t == 2:
                frame_buffer[0] = self._get_state()
            elif t == 3:
                frame_buffer[1] = self._get_state()
            done = self.ale.game_over()
            if done:
                break
        observation = frame_buffer.max(0)[0]
        self.state_buffer.append(observation)
        # Detect loss of life as terminal in training mode
        if self.training:
            lives = self.ale.lives()
            if lives < self.lives and lives > 0:  # Lives > 0 for Q*bert
                self.life_termination = not done  # Only set flag when not truly done
                done = True
            self.lives = lives
        # Return state, reward, done
        return torch.stack(list(self.state_buffer), 0), reward, done

    # Uses loss of life as terminal signal
    def train(self):
        self.training = True

    # Uses standard terminal signal
    def eval(self):
        self.training = False

    def action_space(self):
        return len(self.actions)

    def render(self):
        cv2.imshow('screen', self.ale.getScreenRGB()[:, :, ::-1])
        cv2.waitKey(1)

    def close(self):
        cv2.destroyAllWindows()

class Args:
    def __init__(self, game, seed):
        self.device = torch.device('cuda')
        # self.device = torch.device('cuda')
        self.seed = seed
        self.max_episode_length = 108e3
        self.game = game
        self.history_length = 4