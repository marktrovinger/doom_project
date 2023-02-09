import numpy as np
from file_conversion_utils import strip_file_extension, gzip_files, read_folder
from logged_replay_buffer import OutOfGraphLoggedReplayBuffer

load_data_dir_prefix = '/Users/perusha/tensorboard/DT_dataset/atari_9Feb/'
data_store = '/Users/perusha/tensorboard/DT_dataset/2/'
STORE_FILENAME_PREFIX = '$store$'
buffer_id = 0

lrb = OutOfGraphLoggedReplayBuffer(f"{data_store}", observation_shape=(84,84), stack_size=4, replay_capacity=10000,
                                   batch_size=32)

for buffer_id in range(9):
    obss = np.load(f"{load_data_dir_prefix}/{STORE_FILENAME_PREFIX}_observation_ckpt.{buffer_id}")
    rewards = np.load(f"{load_data_dir_prefix}/{STORE_FILENAME_PREFIX}_reward_ckpt.{buffer_id}")
    terminals = np.load(f"{load_data_dir_prefix}/{STORE_FILENAME_PREFIX}_terminal_ckpt.{buffer_id}")
    actions = np.load(f"{load_data_dir_prefix}/{STORE_FILENAME_PREFIX}_action_ckpt.{buffer_id}")

    num_recs = len(actions)
    for i in range(num_recs):
        lrb.add(obss[i], actions[i], rewards[i], terminals[i])

lrb.log_final_buffer()

# strip_file_extension(data_store)
# gzip_files(data_store)