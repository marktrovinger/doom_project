from pyoblige import *

seed = 123

# configuration for a map that contains secrets, but no enemies
# this configuration will be used primarily to develop the system
# necessary for detecting hidden doors
map_config_no_enemies = {"game": "doom",
                        "engine": "vizdoom",
                        "}
generator = DoomLevelGenerator(seed)

