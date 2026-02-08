
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


30. Scale the individual new extreme parkour terms as needed
    - E.G. stiffness gains are way high. 

31. IF I WANT TO USE EMPIRICAL NORMAILZATION, I WILL NEED TO EDIT HOW I CHANGE NUM_ACTOR_OBS IN ACTORCRITICCUSTOM. FOR NOW IT DOES NOT FUNCTION

32. Next Step is making the custom PPO algorithm. 

33. Confirm scan dot output is what I want. Relative distance from the camera. (naturally in body frame)

34. Need to update PPOCustom.Act for onPolicyrunner rollout (Next Step)

- look at extreme parkour onPolicyRunner and update accordingly

35. Confirm that propriocept used in estimator is the latest obs's



37. Confirm actor_backbone_prop_pos is correct along with vel
        - After this, I need to update the update() functions to include loss for estimator and hist encoders etc. 



38. Need to Get a metric down for comparison to see what helps and what doesn't
38. I don't know if the hist_ proprio needs to be fed from oldest to newest or vice versa into the history encoder. 

39. Setup way to view estimated velocity vs real one.

40. need to fix model export

41. up the gait deviation weight 

    - Make velocity a command
    - Reward tracking that velocity
    - Add a reward that rewards a higher velocity?

42. Added higher scale to vel tracking
    lowered ankle dev. mininimization
    added a third 128 node layer to the priv encoder

43. Recall what size these batches are, what does mean() and item() do?

44. Does estimator get saved between resume calls? 


46. I can either:
    - Tonight: Train with massively varied gains and frictions
    - Get sim2sim setup
    - Get scan dots working
    
47. Investigate why pos_w and quat_w are coming out NaN in the rayCaster file. 

48. Investigate how this could be related to my program destabilizing. 
    - Maybe when using raycaster with base it overrides the sim position with this Nan, but with other settings it doesn't
    - What is the functionality supposed to be?
    - Maybe Nans are coming out all the time but are being suppressed?

49. Scan dots aren't looking close enough to our feet. Sometimes, specifically with going down stairs, even having a recurrent network wouldn't help because it doesn't ever look close enough.

    - I need to get it to look more towards its feet. 
    - So in general the depth camera will have its legs in the image, this could potentially be an issue. 
        - I think I will need to look farther out with a recurrent such that I get a good view of the stairs before my legs block the view.
        - Maybe try to encourage the robot to look slightly more down, but not a whole lot. 

    -compare max velocity with and without scan dots. 
    - try another one with the stepping stones
    - try to increase domain randomization

50. Move scan dots closer to feet, try to crank up max velocity and see what the maximum safe velocity it can traverse things is. 
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


35. ensure ActorCriticRMA.evaluate works properly
    - called in PPOCustom.act, is passed the entire obs from rollout. 
    - looks good to me, its just passed the entire critic obs stuff, which aligns with extreme parkour passing all of the raw stuff to it. 

