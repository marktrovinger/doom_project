Running Vizdoom in Dopamine's framework: 

1. Install Dopamine
2. Install Vizdoom: <>br
Url: https://github.com/Farama-Foundation/ViZDoom <br>
Conda install: <br>
   >conda install -c conda-forge boost cmake gtk2 sdl2 <br>
   >git clone https://github.com/mwydmuch/ViZDoom.git --recurse-submodules <br>
   >cd ViZDoom <br>
   >python setup.py build && python setup.py install <br>

Plus gym wrappers:
```pip install vizdoom[gym] ```

Test installation: 
```run vizdoom_wrapper_test ```

3. Copy/override atari_lib file in dopamine/dopamine/discrete_domains with file provided <br>
This is a quick and dirty work-around; ideally we should create the vizdoom_lib file and figure out how to get dopamine to see it without the ambiguity error it comes up with. <br>

4. Run vizdoom_dopamine_test <br>
Even though the vizdoom wrapper runs in this conda env, the code was not picking up the vizdoom envs as registered so manually registering for now. <br>
Note that only two gym envs are registered in this test code (manually registered) <br>
   ```
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
   ```

Note: don't install SB3 - this will downgrade gym and completely mess everything up :( <br>
Not sure how to get around this yet...
