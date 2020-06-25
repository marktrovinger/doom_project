# A basic random choice agent using VizDoom, basically the tutorial

from vizdoom import DoomGame
from vizdoom import Button
from vizdoom import GameVariable
from vizdoom import ScreenFormat
from vizdoom import ScreenResolution

from random import choice
from time import sleep

scenarios_path = '../scenarios/basic.wad'

# set path and create game object
game = DoomGame()
game.set_doom_scenario_path(scenarios_path)
game.set_doom_map('map01')

# game rendering options
game.set_screen_resolution(ScreenResolution.RES_640X480)
game.set_screen_format(ScreenFormat.RGB24)
game.set_render_hud(False)
game.set_render_weapon(True)
game.set_render_crosshair(False)
game.set_render_decals(False)
game.set_render_particles(False)

# add controls so we can move around
game.add_available_button(Button.MOVE_LEFT)
game.add_available_button(Button.MOVE_RIGHT)
game.add_available_button(Button.ATTACK)

# what pickups do we want in the state? pistol ammo in this case
game.add_available_game_variable(GameVariable.AMMO2)

# episode variables and window
game.set_episode_timeout(200)
game.set_episode_start_time(10)
game.set_window_visible(True)

# set living reward
game.set_living_reward(-1)

# start the game
game.init()

episodes = 10
actions = [[True, False, False], [False, True, False], [False, False, True]]

sleep_time = 1.0 / game.get_ticrate()  # = 0.028

# game loop

for i in range(episodes):
    print("Episode #" + str(i + 1))

    game.new_episode()

    while not game.is_episode_finished():

        state = game.get_state()
        
        n = state.number
        vars = state.game_variables
        screen_buf = state.screen_buffer
        depth_buf = state.depth_buffer
        labels_buf = state.labels_buffer
        automap_buf = state.automap_buffer
        labels = state.labels
        #objects = state.objects
        #sectors = state.sectors

        r = game.make_action(choice(actions))

        print("State #" + str(n))
        print("Game variables:", vars)
        print("Reward:", r)
        print("=====================")

        if sleep_time > 0:
            sleep(sleep_time)

        # Check how the episode went.
    print("Episode finished.")
    print("Total reward:", game.get_total_reward())
    print("************************")
    
game.close()