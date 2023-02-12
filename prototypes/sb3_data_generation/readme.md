To create data for Atari using cleanrl SB3 replay buffer and transfer to DT: <br>

1. Run cleanrl_dqn_atari modified for generating dqn atari data and writing the replay buffer out in DT format at intervals. Set the custom args as below: 
   1. folder in cleanrl_dqn_atari.py and other params and run file. Should generate .npy files in folder. <br>
      ```
      parser.add_argument("--dt-folder", type=str, default="/Users/perusha/tensorboard/DT_dataset/atari_9Feb/",
      help="location of buffer files for dt")
      ```
   2. write to buffer file every 10000 steps
      ```
      parser.add_argument("--dt-buffer-size", type=int, default=10000,
      help="the size of the files saved as decision transformer checkpoints")
      ```

2. Run ```strip_file_extension(folder)``` in file_conversion_utils.py to strip out the .npy extension. WARNING: don't run this more than once! 
3. Run ```gzip_files(folder)``` in file_conversion_utils.py to gzip the files in folder. These files are ready for uploading into DT now. 
4. Run customised ```create_dataset_atari``` to test the upload. Provide folder names before running - see end of file. <br> <br>


A second way to generate files from SB3 pickles is using the logged_replay_buffer. <br>
Run logged_wrapper_test and pass it the transitions. It will generate the buffer files automatically!
