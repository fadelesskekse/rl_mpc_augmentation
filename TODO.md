
1. track_lin_vel_xy Reward:
    - Currently uses simulation yaw frame velocity, need to create way to estimate this.

    - Implement estiamtor

3. JointPositionAction:
    - Do I need to use 'preserve_order'?
    - In what regard does it preserve order? 

    - Come back to once I implement MPC

4. Observations: 
    - Should CriticCfg get un-noised observations or even
    additional observations than the PolicyCfg?


5. Rewards:
    - Confirm rewards are all within similar scale

6. Observation __post_init__:
    - Configure history_length perhaps individually depending on observation? 
        - Justifcation: Might need different history length for depth images.

    - How does 'enable_corruption' actually work?

7. Output of policy Idea:
    - An output of policy will already be gait-cyle
    - We will have a function that uses gait-cycle to produce phases used in swing leg and MPC gait-table
    - What if we output a phase augmentation that learns to modulate phase such that if the swing leg collides in z-direction it cancels out phase so swing leg becomes stance? 

8. Do I want gait phase as an observation?
    - I have read a few papers that have done this.
    - If so, I dont want the phase calculation to be done in a mdp function. 
        - Right now I have feet_gait calculations as an mdp function

9. base_lin_vel observation:
    - Configure noise

    - Need to use estimated value of this, not the sim data version.

    - Unclear if I need to use estimated or sim data version in the critic_cfg

    - Unclear if I should only use x_y or also z_velocity

10. base_z_pos observation:
    - Configure noise 

    - Need to use estimated value of this, not the sim data version. 

    - Unclear if I need to use estimated or sim data version in the critic_cfg

11. Need to revisit domain randomization and normalization.
    - Add randomized motor gains
    - Add randomized CoM


12. Train blind end to end -rl

13. Create a new train_cfg that is passed to my custom OnPolicyRunner

14. resolve_rnd_config in OnPolicyRunner constructalgorithm().
    - Is this called during operation?


17. Implement exeception errors for:
    - Check dimension of each extreme_parkour related obs group, and check if it matches the hard coded dimension that I copied from extreme parkour

20. extreme parkour policy cfg has "continue_from_last_std" is this a thing for the new one? 

21. How does history_len I defined in env.cfg relate to any other history buffers I am currently using for my obs groups. That might need to be specified differently. 

22. onPolictyRunnerCustom successfully creates ActorCriticRMA, but there is still work that needs to be done inside of ActorCriticRMA. 

    - I think estimator creation is fine for now.

23. The dimension sizes that get passed to Actor are only for extreme parkour values, this needs to be updated a lot
    - The num_actions passed to Actor is from env.num_actions, not a hard-coded/ported value from extreme parkour. The num_actions passed to Actor in extreme parkour is hard coded.
        - Need to make sure this works properly. 

24. num_critic_obs needs to be carefully setup to include all observations specified in isaaclab obs groups and confirmed that the obs groups that correspond to extreme_parkour observations match the number of hard_coded values I copied over. 

 Next step is creating a custom PPO

####################################################
19. Early termination if I exceed half of the course?: SOLVED

18. Remove min height early termination? SOLVED
    - Removed, it would mess up the downwards stair climb.

15. Figure out how to do terrain: SOLVED

14. reset_base event: SOLVED
    - Probably need to adjust the yaw and x,y positions based on terrain generation

2.  - Confirm scales of Observations are all similar: SOLVED


14. For every terrain type (row), plot average difficulty as a metric. : SOLVED
    - Only turn on during inference, it slows down training a ton

13. In mpc_actions, I use preserve_order = True on self._joint_ids. :SOLVED
    - Don't use it, its not needed. 


