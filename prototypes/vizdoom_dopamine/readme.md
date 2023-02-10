Running Vizdoom in Dopamine's framework: 

1. Install Dopamine
2. Install Vizdoom:
Url: https://github.com/Farama-Foundation/ViZDoom
Conda install:
   conda install -c conda-forge boost cmake gtk2 sdl2
   git clone https://github.com/mwydmuch/ViZDoom.git --recurse-submodules
   cd ViZDoom
   python setup.py build && python setup.py install

Plus gym wrappers:
pip install vizdoom[gym]

Test installation: 
run vizdoom_wrapper_test

3. Copy/override atari_lib file in /Users/perusha/git_repos/dopamine/dopamine/discrete_domains with file provided 
4. Run vizdoom_dopamine_test

Note that only two gym envs are registered in this test code (manually registered)
   register(
   id="VizdoomBasic-v0",
   entry_point="vizdoom.gym_wrapper.gym_env_defns:VizdoomScenarioEnv",
   kwargs={"scenario_file": "basic.cfg"}
   )

   register(
   id="VizdoomHealthGatheringSupreme-v0",
   entry_point="vizdoom.gym_wrapper.gym_env_defns:VizdoomScenarioEnv",
   kwargs={"scenario_file": "health_gathering_supreme.cfg"}
   )

Note: don't install SB3 - this will downgrade gym and completely mess everything up :( 