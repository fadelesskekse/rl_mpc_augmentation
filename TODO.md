
1. track_lin_vel_xy Reward:
    - Currently uses simulation yaw frame velocity, need to create way to estimate this.

2. track_ang_vel_z Reward:
    - Currently uses body frame ang vel from sim, need to make sure it takes in a noise augmented signal to approximate IMU reading.

3. JointPositionAction:
    - Do I need to use 'preserve_order'?
    - In what regard does it preserve order? 

    - Come back to once I implement MPC

4. Observations: 
    - Should CriticCfg get un-noised observations or even
    additional observations than the PolicyCfg?

    - Confirm scales of Observations are all similar: SOLVED


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

11. feet_clearance Reward:
    - Currently implemented to aid in foot step without vision
        -Try removing this once vision is added in end-to-end RL

12. feet_gait Reward:
    - Used for end-to-end RL to promote gait
        -Not used in MPC

13. Event funcs that are actually classes:
    - Confirm that the funcs are actually executing

14. reset_base event:
    - Probably need to adjust the yaw and x,y positions based on terrain generation

15. Figure out how to do terrain.

16. Need to revisit domain randomization and normalization.

17. Train blind end to end -rl

18. Remove min height early termination? SOLVED
    - Removed, it would mess up the downwards stair climb.

19. Early termination if I exceed half of the course?

