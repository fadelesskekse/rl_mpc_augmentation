# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
import torch.nn as nn
from torch.distributions import Normal

from rsl_rl.networks import MLP, EmpiricalNormalization

class StateHistoryEncoder(nn.Module):
    def __init__(self, activation_fn, input_size, tsteps, output_size, tanh_encoder_output=False):
                       #activation, num_prop, num_hist, priv_encoder_output_dim)

        # self.device = device
        super(StateHistoryEncoder, self).__init__()
        self.activation_fn = activation_fn
        self.tsteps = tsteps

        channel_size = 10
        # last_activation = nn.ELU()

        self.encoder = nn.Sequential(
                nn.Linear(input_size, 3 * channel_size), self.activation_fn,
                )

        if tsteps == 50:
            self.conv_layers = nn.Sequential(
                    nn.Conv1d(in_channels = 3 * channel_size, out_channels = 2 * channel_size, kernel_size = 8, stride = 4), self.activation_fn,
                    nn.Conv1d(in_channels = 2 * channel_size, out_channels = channel_size, kernel_size = 5, stride = 1), self.activation_fn,
                    nn.Conv1d(in_channels = channel_size, out_channels = channel_size, kernel_size = 5, stride = 1), self.activation_fn, nn.Flatten())
        elif tsteps == 10:
            self.conv_layers = nn.Sequential(
                nn.Conv1d(in_channels = 3 * channel_size, out_channels = 2 * channel_size, kernel_size = 4, stride = 2), self.activation_fn,
                nn.Conv1d(in_channels = 2 * channel_size, out_channels = channel_size, kernel_size = 2, stride = 1), self.activation_fn,
                nn.Flatten())
        elif tsteps == 20:
            self.conv_layers = nn.Sequential(
                nn.Conv1d(in_channels = 3 * channel_size, out_channels = 2 * channel_size, kernel_size = 6, stride = 2), self.activation_fn,
                nn.Conv1d(in_channels = 2 * channel_size, out_channels = channel_size, kernel_size = 4, stride = 2), self.activation_fn,
                nn.Flatten())
        else:
            raise(ValueError("tsteps must be 10, 20 or 50"))

        self.linear_output = nn.Sequential(
                nn.Linear(channel_size * 3, output_size), self.activation_fn
                )

    def forward(self, obs):
        # nd * T * n_proprio
        nd = obs.shape[0]
        T = self.tsteps
        # print("obs device", obs.device)
        # print("encoder device", next(self.encoder.parameters()).device)

       # print(f"obs shape in hist encod forward: {obs.shape}")
        projection = self.encoder(obs.reshape([nd * T, -1]))
        #print(f"proj shape in hist encod forward: {projection.shape}")
        output = self.conv_layers(projection.reshape([nd, T, -1]).permute((0, 2, 1)))
        output = self.linear_output(output)

       # print(f"hist encoder output shape: {output.shape}")
        return output
    
class Actor(nn.Module):
    def __init__(self,
                 num_actor_backbone_prop_hist,
                 num_obs,
                 num_prop, 
                 num_scan, 
                 num_actions, 
                 scan_encoder_dims,
                 actor_hidden_dims, 
                 priv_encoder_dims, 
                 num_priv_latent, 
                 num_priv_explicit, 
                 num_hist, activation, 
                 scan_cnn,
                 row_scan,
                 col_scan,
                 scan_cnn_output_dim,
                 tanh_encoder_output=False,
                 ) -> None:
        
       

        super().__init__()
        # prop -> scan -> priv_explicit -> priv_latent -> hist
        # actor input: prop -> scan -> priv_explicit -> latent
        self.num_actor_backbone_prop_hist = num_actor_backbone_prop_hist
        self.num_obs = num_obs
        self.num_prop = num_prop
        self.num_scan = num_scan
        self.num_hist = num_hist
        self.num_actions = num_actions
        self.num_priv_latent = num_priv_latent
        self.num_priv_explicit = num_priv_explicit
        self.if_scan_encode = scan_encoder_dims is not None and num_scan > 0
        self.scan_encode_cnn = self.if_scan_encode and scan_cnn
        self.row_scan = row_scan
        self.col_scan = col_scan
        self.scan_cnn_output_dim = scan_cnn_output_dim

        assert self.row_scan * self.col_scan == self.num_scan, \
            f"row_scan*col_scan={self.row_scan*self.col_scan} != num_scan={self.num_scan}"

       # print(f"scan_encoder_dins: {scan_encoder_dims} and scan_cnn: {self.scan_encode_cnn}")

       # print(f"scan cnn output dim: {self.scan_cnn_output_dim}")

        if len(priv_encoder_dims) > 0:
                    priv_encoder_layers = []
                    priv_encoder_layers.append(nn.Linear(num_priv_latent, priv_encoder_dims[0]))
                    priv_encoder_layers.append(activation)
                    for l in range(len(priv_encoder_dims) - 1):
                        priv_encoder_layers.append(nn.Linear(priv_encoder_dims[l], priv_encoder_dims[l + 1]))
                        priv_encoder_layers.append(activation)
                    self.priv_encoder = nn.Sequential(*priv_encoder_layers)
                    priv_encoder_output_dim = priv_encoder_dims[-1]

                    print(f"priv_encoder Output dim: {priv_encoder_output_dim}")

                    self.num_obs += priv_encoder_output_dim #dont add this twice in the historyencoder. 
        else:
            self.priv_encoder = nn.Identity()
            priv_encoder_output_dim = num_priv_latent

            self.num_obs += priv_encoder_output_dim #dont add this twice in the historyencoder. 

        self.history_encoder = StateHistoryEncoder(activation, num_prop, num_hist, priv_encoder_output_dim)
        
        if self.if_scan_encode:
            scan_encoder = []

            if not self.scan_encode_cnn:

                
                scan_encoder.append(nn.Linear(self.num_scan, scan_encoder_dims[0]))  #scan_encoder_dims = [128, 64, 32],
                scan_encoder.append(activation)
                for l in range(len(scan_encoder_dims) - 1):
                    if l == len(scan_encoder_dims) - 2:
                        scan_encoder.append(nn.Linear(scan_encoder_dims[l], scan_encoder_dims[l+1]))
                        scan_encoder.append(nn.Tanh())
                    else:
                        scan_encoder.append(nn.Linear(scan_encoder_dims[l], scan_encoder_dims[l + 1]))
                        scan_encoder.append(activation)
                self.scan_encoder = nn.Sequential(*scan_encoder)
                self.scan_encoder_output_dim = scan_encoder_dims[-1]

                self.num_obs += self.scan_encoder_output_dim

            else:

                # oc1 = 32
                # k1 = 5
                
                # k2 = 2
                # s2 = 2

                # oc3 = 64
                # k3 = 3

                # H1 = (row_scan - k1)+1
                # W1 = (col_scan - k1)+1

                # H2 = (H1  - k2)//s2 + 1
                # W2 = (W1  - k2)//s2 + 1

                # H3 = (H2 - k3) + 1
                # W3 = (W2 - k3) + 1

                # flatten_dim = oc3 * H3 * W3

                # #[1,row_scan,col_scan]
                # self.scan_encoder = nn.Sequential(
                #     # [1, 58, 87]
                #     #[1, row_scan,col_scan]
                #     nn.Conv2d(in_channels=1, out_channels=oc1, kernel_size=k1),
                #     # [32, 54, 83]
                #     #[32,prev_row = (row_scan - k1)+1, prev_col = (col_scan - k1)+1]
                #     nn.MaxPool2d(kernel_size=k2, stride=s2),
                #    # [32, 27, 41]
                #    # [32, prev_row = (prev_row  - k2)/s2 + 1, prev_col = (prev_col  - k2)/s2 + 1 ]
                #     activation,
                #     nn.Conv2d(in_channels=oc1, out_channels=oc3, kernel_size=k3),
                #     activation,
                #     nn.Flatten(),
                #     # [64, 25, 39]
                #     #[64,prev_row = (prev_row - k3) + 1, prev_col = (prev_col - k3) + 1]
                #     nn.Linear(flatten_dim, 128),
                #     activation,
                #     nn.Linear(128, self.scan_cnn_output_dim )
                # )

                oc1 = 16
                k1 = 3
                p1 = 1
                
                oc2 = 16
                k2 = 3
                p2 = 1

                k3_h = 1
                k3_w = 2
                
                oc4 = 32
                k4 = 3
                p4 = 1

                H0 = row_scan
                W0 = col_scan

                H1 = (H0 +2*p1- k1)+1
                W1 = (W0 +2*p1- k1)+1

                H2 = (H1 +2*p2- k2) + 1
                W2 = (W1 +2*p2- k2) + 1

                H3 = (H2 - k3_h)//k3_h + 1
                W3 = (W2 - k3_w)//k3_w + 1

                H4 = (H3 +2*p4 - k4) + 1
                W4 = (W3 +2*p4 - k4) + 1


                flatten_dim = oc4 * H4 * W4

                #[1,row_scan,col_scan]
                self.scan_encoder = nn.Sequential(
                    # [1, 58, 87]
                    #[1, row_scan,col_scan]
                    nn.Conv2d(in_channels=1, out_channels=oc1, kernel_size=k1,padding=p1),
                    # [32, 54, 83]
                    activation,
                    nn.Conv2d(in_channels=oc1, out_channels=oc2, kernel_size=k2,padding=p2),
                    activation,
                    #[32,prev_row = (row_scan - k1)+1, prev_col = (col_scan - k1)+1]
                    nn.MaxPool2d(kernel_size=(k3_h,k3_w)),
                   # [32, 27, 41]
                   # [32, prev_row = (prev_row  - k2)/s2 + 1, prev_col = (prev_col  - k2)/s2 + 1 ]
                    #activation,
                    nn.Conv2d(in_channels=oc2, out_channels=oc4, kernel_size=k4,padding=p4),
                    activation,
                    nn.Flatten(),
                    # [64, 25, 39]
                    #[64,prev_row = (prev_row - k3) + 1, prev_col = (prev_col - k3) + 1]
                    nn.Linear(flatten_dim, 128),
                    activation,
                    nn.Linear(128, self.scan_cnn_output_dim )
                )

                self.scan_encoder_output_dim = self.scan_cnn_output_dim 
                self.num_obs += self.scan_cnn_output_dim 
                


        else:
            self.scan_encoder = nn.Identity()
            self.scan_encoder_output_dim = self.num_scan
            self.num_obs += self.scan_encoder_output_dim
            print(f"scan_encoder_output_dim should be 0 for testing: {self.scan_encoder_output_dim}")
        
        actor_layers = []
        # actor_layers.append(nn.Linear(num_prop+ #actor_backbone only takes in a
        #                               self.scan_encoder_output_dim+
        #                               num_priv_explicit+
        #                               priv_encoder_output_dim, 
        #                               actor_hidden_dims[0]))

        #Input num_obs should only have the values associated with any extra observations not related to extreme_parkour. Before passing it in, we subtract
        #away the raw scan encode dim, the priv_latent dim, and any extra proprioception history not used in the backbone. Afetr passing in, we add the 
        #scan encode output and the priv/hist encoder output dim, thus the final num_obs as all proprioception history relevant to the backbone, 
        #scan encodeoutput, priv/hist encode output, vel_estimated/raw data, and any other extra observations not related to extreme parkour. 
        actor_layers.append(nn.Linear(self.num_obs,actor_hidden_dims[0]))
        
        actor_layers.append(activation)
        for l in range(len(actor_hidden_dims)):
            if l == len(actor_hidden_dims) - 1:
                actor_layers.append(nn.Linear(actor_hidden_dims[l], num_actions))
            else:
                actor_layers.append(nn.Linear(actor_hidden_dims[l], actor_hidden_dims[l + 1]))
                actor_layers.append(activation)
        if tanh_encoder_output:
            actor_layers.append(nn.Tanh())
        self.actor_backbone = nn.Sequential(*actor_layers)

    def forward(self, obs, hist_encoding: bool = False, eval: bool=False, scandots_latent=None):

      #  print(f"eval : {eval}")
        if not eval:
           # print(f"obs shape passed to actor forward: {obs.shape}")
            if self.if_scan_encode: #Do we want to encode the scan dots? 
                #print(f"obs in forward actor pass: {obs}")
                #obs_scan = obs[:, self.num_prop:self.num_prop + self.num_scan]
                obs_scan = obs[:,:self.num_scan]

                #print(f"obs_scan: {obs_scan}")

                #print(f"obs shape: {obs.shape}")

               # print(f"obs_scan shape: {obs_scan.shape}")

                if obs_scan.shape[1] != self.num_scan:
                    raise ValueError(f"obs_scan should be the same size as num_scan, it is size {obs_scan.shape[1]}")
                

                if scandots_latent is None: #If we dont pass a scandot_latent in the Actors forward pass, we generate a new latent. 

                    if not self.scan_encode_cnn:
                        scan_latent = self.scan_encoder(obs_scan) 
                    else:
                        B = obs_scan.shape[0]
                        obs_scan = obs_scan.view(B, 1, self.row_scan, self.col_scan)
                       # print(f"obs_scan: {obs_scan}")
                        # print(f"obs shape for conv scan {obs_scan.shape}")
                        scan_latent = self.scan_encoder(obs_scan)

                else:
                    scan_latent = scandots_latent
               # obs_prop_scan = torch.cat([obs[:, :self.num_prop], scan_latent], dim=1) #combines the scan latent and propropception observations into
                                                                                        #one observation
                obs_scan_actor_backbone_input = scan_latent

               # print(f"scan output dim: {self.scan_encoder_output_dim}")
              #  print(f"obs_scan_actor_backbone_input: {obs_scan_actor_backbone_input.shape}")

               
                if obs_scan_actor_backbone_input.shape[0] != obs.shape[0]:
                    raise ValueError(f"Error in scan encoder output shape. 0th element shoudl be of size num_envs, it is {obs_scan_actor_backbone_input.shape[0]}")
                 

                if obs_scan_actor_backbone_input.shape[1] != self.scan_encoder_output_dim:
                    raise ValueError(f"Error in size for 'obs_scan_actor_backbone_input'. Is {obs_scan_actor_backbone_input.shape[1]}, should be {self.scan_encoder_output_dim}")
                

            else:
                #obs_prop_scan = obs[:, :self.num_prop + self.num_scan] #if we aren't encoding the scan, combine the raw scan obs with prop obs
                obs_scan_actor_backbone_input = obs[:,:self.num_scan] #If we aren't encoding, just pass the raw obs
              #  print(f"obs_scan_actor_backbone_input should be of size num_envs,0, it is: {obs_scan_actor_backbone_input.shape}")
            #obs_priv_explicit = obs[:, self.num_prop + self.num_scan:self.num_prop + self.num_scan + self.num_priv_explicit] #grabs the priviledged states
                                                                                                                            # which are from the estimator

            obs_priv_explicit = obs[:,self.num_scan: self.num_scan + self.num_priv_explicit]

           # print(f"obs priv explicit in actor forward: {obs_priv_explicit}")

            if obs_priv_explicit.shape[1] != self.num_priv_explicit:
                raise ValueError(f"obs_priv_explicit should be of size {self.num_priv_explicit}, but it of size {obs_priv_explicit.shape[1]}")
            
           # print(f"hist_encoding: {hist_encoding}")
           # hist = obs[:,self.num_scan + self.num_priv_explicit + self.num_priv_latent: self.num_scan + self.num_priv_explicit + self.num_priv_latent + self.num_hist*self.num_prop//2]
            #print(f"history proprio: {hist}")
            if hist_encoding:
                latent = self.infer_hist_latent(obs)
               # print("I am using hist encoder")

                

            else:

                latent = self.infer_priv_latent(obs)
               # print("I am using priv encoder")
                 # Adaptation module update
               # with torch.inference_mode():
                   # print(f"priv latenet: {self.infer_priv_latent(obs)}")
                


            

            extreme_parkour_obs_base = self.num_scan + self.num_priv_explicit + self.num_priv_latent + self.num_hist*self.num_prop

            non_exr_pkr_obs = obs[:,extreme_parkour_obs_base:]

            

           # print(f"non extreme obs: {non_exr_pkr_obs}")

            base = self.num_scan + self.num_priv_explicit + self.num_priv_latent
            half_prop = self.num_prop // 2
            hist_offset = (self.num_hist - self.num_actor_backbone_prop_hist) * half_prop

            
            actor_backbone_prop_pos = obs[:,base + hist_offset: base + hist_offset + half_prop*self.num_actor_backbone_prop_hist]

            actor_backbone_prop_vel = obs[:,base + half_prop*self.num_hist + hist_offset: base + half_prop*self.num_hist + hist_offset +half_prop*self.num_actor_backbone_prop_hist]

          #  print(f"size of actor_pos_prop_back: {actor_backbone_prop_pos.shape[1]}")
           # print(f"size of actor_vel_prop_back: {actor_backbone_prop_vel.shape[1]}")

            if actor_backbone_prop_pos.shape[1] != self.num_actor_backbone_prop_hist*half_prop or actor_backbone_prop_vel.shape[1] != self.num_actor_backbone_prop_hist*half_prop:
                
                print(f"size of actor_pos_prop_back: {actor_backbone_prop_pos.shape[1]}")
                print(f"size of actor_vel_prop_back: {actor_backbone_prop_vel.shape[1]}")
                raise ValueError(f"Backbone pos or vel proprio history shape is off, it should be of size {self.num_actor_backbone_prop_hist*half_prop}")

            
            #backbone_input = torch.cat([obs_prop_scan, obs_priv_explicit, latent], dim=1)
            backbone_input = torch.cat([obs_scan_actor_backbone_input, obs_priv_explicit, latent,actor_backbone_prop_pos,actor_backbone_prop_vel,non_exr_pkr_obs], dim=1)
           
            #print(f"shape of backbone input: {backbone_input.shape}")

            if backbone_input.shape[1] != self.num_obs:
                raise ValueError(f"backbone input should have shape of {self.num_obs} but it is of size {backbone_input}")


            backbone_output = self.actor_backbone(backbone_input)
            return backbone_output
        
        else:
            # print(f"obs shape passed to actor forward: {obs.shape}")
            if self.if_scan_encode: #Do we want to encode the scan dots? 
                #print(f"obs in forward actor pass: {obs}")
                #obs_scan = obs[:, self.num_prop:self.num_prop + self.num_scan]
                obs_scan = obs[:,:self.num_scan]

                #print(f"obs_scan: {obs_scan}")

                #print(f"obs shape: {obs.shape}")

               # print(f"obs_scan shape: {obs_scan.shape}")

                if obs_scan.shape[1] != self.num_scan:
                    raise ValueError(f"obs_scan should be the same size as num_scan, it is size {obs_scan.shape[1]}")
                
                if scandots_latent is None: #If we dont pass a scandot_latent in the Actors forward pass, we generate a new latent. 
                    scan_latent = self.scan_encoder(obs_scan)   
                else:
                    scan_latent = scandots_latent
               # obs_prop_scan = torch.cat([obs[:, :self.num_prop], scan_latent], dim=1) #combines the scan latent and propropception observations into
                                                                                        #one observation
                obs_scan_actor_backbone_input = scan_latent

               # print(f"scan output dim: {self.scan_encoder_output_dim}")
              #  print(f"obs_scan_actor_backbone_input: {obs_scan_actor_backbone_input.shape}")

               
                if obs_scan_actor_backbone_input.shape[0] != obs.shape[0]:
                    raise ValueError(f"Error in scan encoder output shape. 0th element shoudl be of size num_envs, it is {obs_scan_actor_backbone_input.shape[0]}")
                 

                if obs_scan_actor_backbone_input.shape[1] != self.scan_encoder_output_dim:
                    raise ValueError(f"Error in size for 'obs_scan_actor_backbone_input'. Is {obs_scan_actor_backbone_input.shape[1]}, should be {self.scan_encoder_output_dim}")
                

            else:
                #obs_prop_scan = obs[:, :self.num_prop + self.num_scan] #if we aren't encoding the scan, combine the raw scan obs with prop obs
                obs_scan_actor_backbone_input = obs[:,:self.num_scan] #If we aren't encoding, just pass the raw obs
              #  print(f"obs_scan_actor_backbone_input should be of size num_envs,0, it is: {obs_scan_actor_backbone_input.shape}")
            #obs_priv_explicit = obs[:, self.num_prop + self.num_scan:self.num_prop + self.num_scan + self.num_priv_explicit] #grabs the priviledged states
                                                                                                                            # which are from the estimator

            obs_priv_explicit = obs[:,self.num_scan: self.num_scan + self.num_priv_explicit]

           # print(f"obs priv explicit in actor forward: {obs_priv_explicit}")

            if obs_priv_explicit.shape[1] != self.num_priv_explicit:
                raise ValueError(f"obs_priv_explicit should be of size {self.num_priv_explicit}, but it of size {obs_priv_explicit.shape[1]}")
            
           # print(f"hist_encoding: {hist_encoding}")
            #hist = obs[:,self.num_scan + self.num_priv_explicit + self.num_priv_latent: self.num_scan + self.num_priv_explicit + self.num_priv_latent + self.num_hist*self.num_prop]
            #print(f"history proprio: {hist}")
            if hist_encoding:
                latent = self.infer_hist_latent(obs)
            else:
               # print("We are using priv_latent")
              #  print(f"shape of obs passed {obs.shape}")
                latent = self.infer_priv_latent(obs)

            #print(f"size of latent: {latent.size}")

            extreme_parkour_obs_base = self.num_scan + self.num_priv_explicit + self.num_priv_latent + self.num_hist*self.num_prop

            non_exr_pkr_obs = obs[:,extreme_parkour_obs_base:]

           # print(f"non extreme obs: {non_exr_pkr_obs}")

            base = self.num_scan + self.num_priv_explicit + self.num_priv_latent
            half_prop = self.num_prop // 2
            hist_offset = (self.num_hist - self.num_actor_backbone_prop_hist) * half_prop

            
            actor_backbone_prop_pos = obs[:,base + hist_offset: base + hist_offset + half_prop*self.num_actor_backbone_prop_hist]

            actor_backbone_prop_vel = obs[:,base + half_prop*self.num_hist + hist_offset: base + half_prop*self.num_hist + hist_offset +half_prop*self.num_actor_backbone_prop_hist]

          #  print(f"size of actor_pos_prop_back: {actor_backbone_prop_pos.shape[1]}")
           # print(f"size of actor_vel_prop_back: {actor_backbone_prop_vel.shape[1]}")

            if actor_backbone_prop_pos.shape[1] != self.num_actor_backbone_prop_hist*half_prop or actor_backbone_prop_vel.shape[1] != self.num_actor_backbone_prop_hist*half_prop:
                
                print(f"size of actor_pos_prop_back: {actor_backbone_prop_pos.shape[1]}")
                print(f"size of actor_vel_prop_back: {actor_backbone_prop_vel.shape[1]}")
                raise ValueError(f"Backbone pos or vel proprio history shape is off, it should be of size {self.num_actor_backbone_prop_hist*half_prop}")

            
            #backbone_input = torch.cat([obs_prop_scan, obs_priv_explicit, latent], dim=1)
            backbone_input = torch.cat([obs_scan_actor_backbone_input, obs_priv_explicit, latent,actor_backbone_prop_pos,actor_backbone_prop_vel,non_exr_pkr_obs], dim=1)
           
            #print(f"shape of backbone input: {backbone_input.shape}")

            if backbone_input.shape[1] != self.num_obs:
                raise ValueError(f"backbone input should have shape of {self.num_obs} but it is of size {backbone_input}")


            backbone_output = self.actor_backbone(backbone_input)
            return backbone_output
            #raise ValueError("eval for whatever reason is True, I need to relook at code")
        # else:
        #     if self.if_scan_encode:
        #         obs_scan = obs[:, self.num_prop:self.num_prop + self.num_scan]
        #         if scandots_latent is None:

        #             scan_latent = self.scan_encoder(obs_scan)
        #             print("I made it past scan encoder")   
        #         else:
        #             scan_latent = scandots_latent
        #         obs_prop_scan = torch.cat([obs[:, :self.num_prop], scan_latent], dim=1)
        #     else:
        #         obs_prop_scan = obs[:, :self.num_prop + self.num_scan]
        #     obs_priv_explicit = obs[:, self.num_prop + self.num_scan:self.num_prop + self.num_scan + self.num_priv_explicit]
        #     if hist_encoding:
        #         latent = self.infer_hist_latent(obs)
        #     else:
        #         latent = self.infer_priv_latent(obs)
        #     #backbone_input = torch.cat([obs_prop_scan, obs_priv_explicit, latent], dim=1)
        #     backbone_input = torch.cat([obs_scan_actor_backbone_input, obs_priv_explicit, latent], dim=1)

        #     backbone_output = self.actor_backbone(backbone_input)

        #     return backbone_output

    def infer_scan_latent(self,obs):
        return self.scan_encoder(obs)
    
    def infer_priv_latent(self, obs):

        #priv = obs[:, self.num_prop + self.num_scan + self.num_priv_explicit: self.num_prop + self.num_scan + self.num_priv_explicit + self.num_priv_latent]
       # print(f"obs shape: {obs.shape}")
        #print(f"obs: {obs}")

        priv = obs[:,self.num_scan + self.num_priv_explicit: self.num_scan + self.num_priv_explicit + self.num_priv_latent]

        #print(f"priv latent input: {priv}")

        #print(f"priv: {priv}")
        return self.priv_encoder(priv)
    
    def infer_hist_latent(self, obs):
        #hist = obs[:, -self.num_hist*self.num_prop:]
        hist = obs[:,self.num_scan + self.num_priv_explicit + self.num_priv_latent: self.num_scan + self.num_priv_explicit + self.num_priv_latent + self.num_hist*self.num_prop]
        
        #print(f"hist: {hist}")
        #print(f"proprio hist: {hist}")

        num_envs = hist.shape[0]
        H = self.num_hist           # e.g. 10
        J = self.num_prop // 2      # 29 joints
        P = self.num_prop           # 58 = 29 pos + 29 vel

        # split into position and velocity blocks
        pos = hist[:, :H * J]              # (num_envs, 290)
        vel = hist[:, H * J : 2 * H * J]   # (num_envs, 290)

        # reshape into (env, time, joints)
        pos = pos.view(num_envs, H, J)     # (num_envs, 10, 29)
        vel = vel.view(num_envs, H, J)     # (num_envs, 10, 29)

        # interleave into (env, time, 58)
        hist_interleaved = torch.cat([pos, vel], dim=2)  # (num_envs, 10, 58)

        assert hist_interleaved.shape[1] == self.num_hist
        assert hist_interleaved.shape[2] == self.num_prop

        # check semantic alignment for env 0, t = 0
        #print("t0 pos:", hist_interleaved[0, self.num_hist - 1, :J])
       # print("t0 vel:", hist_interleaved[0, self.num_hist - 1, J:])

        # check semantic alignment for env 0, t = 0
       # print("t0_last pos:", hist_interleaved[0, 0, :J])
       # print("t0_last vel:", hist_interleaved[0, 0, J:])

       # print(f"after weaving hist: {hist_interleaved}")


        #print(f"shape of new: {hist_interleaved.shape}" )

        #print(f"shape of orignal: {hist.view(-1, self.num_hist, self.num_prop).shape}")

        #return self.history_encoder(hist.view(-1, self.num_hist, self.num_prop))

        return self.history_encoder(hist_interleaved)
    
    def infer_scandots_latent(self, obs):
        scan = obs[:, self.num_prop:self.num_prop + self.num_scan]
        return self.scan_encoder(scan)

class ActorCriticRMA(nn.Module):
    is_recurrent = False


    def __init__(
        self,
        obs,
        obs_groups,
        num_actions,

        ########## custom ############
        num_prop,
        num_scan,
        num_critic_extra, 
        num_priv_latent, 
        num_priv_explicit,
        num_hist,
        num_hist_for_actor_backbone_proprio,

        ########## custom end ############

        actor_obs_normalization=False,
        critic_obs_normalization=False,
        actor_hidden_dims=[256, 256, 256],
        critic_hidden_dims=[256, 256, 256],
        activation="elu",
        init_noise_std=1.0,
        noise_std_type: str = "scalar",

        ########## custom ############
        scan_encoder_dims=[256, 256, 256],
        scan_cnn = False,
        row_scan = 10,
        col_scan = 26,
        scan_cnn_output_dim = 31,
        ########## custom end ############


        **kwargs,
    ):
        if kwargs:
            print(
                "ActorCritic.__init__ got unexpected arguments, which will be ignored: "
                + str([key for key in kwargs.keys()])
            )
        super().__init__()

  

        #######c##########
        self.kwargs = kwargs
        priv_encoder_dims= kwargs['priv_encoder_dims']
        
        #########cn#######

        # get the observation dimensions
        self.obs_groups = obs_groups
        num_actor_obs = 0
        for obs_group in obs_groups["policy"]:
            assert len(obs[obs_group].shape) == 2, "The ActorCritic module only supports 1D observations."
            num_actor_obs += obs[obs_group].shape[-1]

        ############c############
        if num_hist_for_actor_backbone_proprio > num_hist:
            raise ValueError(f"'num_hist_for_actor_backbone_proprio' should be less than 'num_hist', it is of size {num_hist_for_actor_backbone_proprio}")

        num_actor_obs -= (num_scan +  num_priv_latent + (num_hist - num_hist_for_actor_backbone_proprio) *num_prop)
        #remove all obs dims from num_actor_obs that correspond to obs's that do not feed into the actor backbone.

        ############cn##########
        
        print(f"Size of obs_group: {num_actor_obs}")

        num_critic_obs = 0
        for obs_group in obs_groups["critic"]:
            assert len(obs[obs_group].shape) == 2, "The ActorCritic module only supports 1D observations."
            num_critic_obs += obs[obs_group].shape[-1]

        

        # actor

        #raise error if num_actor_obs for extreme_parkour obs's specified in Isaaclab is different than 
        # hard coded values pulled from extreme parkour 

        ####################c########################
        #Note: activation inside of original MLP for actor is handled within MLP init.
        #self.actor = MLP(num_actor_obs, num_actions, actor_hidden_dims, activation)
        self.actor = Actor(num_hist_for_actor_backbone_proprio,
                           num_actor_obs,num_prop, 
                           num_scan, num_actions, 
                           scan_encoder_dims, 
                           actor_hidden_dims, 
                           priv_encoder_dims, 
                           num_priv_latent, 
                           num_priv_explicit, 
                           num_hist, 
                           get_activation(activation),
                           scan_cnn,
                           row_scan,
                           col_scan,
                           scan_cnn_output_dim,
                           tanh_encoder_output=kwargs['tanh_encoder_output'],
                           )
        ####################cn#######################

        # actor observation normalization
        self.actor_obs_normalization = actor_obs_normalization
        if actor_obs_normalization:
            self.actor_obs_normalizer = EmpiricalNormalization(num_actor_obs)
        else:
            self.actor_obs_normalizer = torch.nn.Identity()
        print(f"Actor MLP: {self.actor}")

        # critic
        print(f"num_critic_obs:{num_critic_obs}")
        self.critic = MLP(num_critic_obs, 1, critic_hidden_dims, activation)
        # critic observation normalization
        self.critic_obs_normalization = critic_obs_normalization
        if critic_obs_normalization:
            self.critic_obs_normalizer = EmpiricalNormalization(num_critic_obs)
        else:
            self.critic_obs_normalizer = torch.nn.Identity()
        print(f"Critic MLP: {self.critic}")

        # Action noise
        self.noise_std_type = noise_std_type
        if self.noise_std_type == "scalar":
            self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
        elif self.noise_std_type == "log":
            self.log_std = nn.Parameter(torch.log(init_noise_std * torch.ones(num_actions)))
        else:
            raise ValueError(f"Unknown standard deviation type: {self.noise_std_type}. Should be 'scalar' or 'log'")

        # Action distribution (populated in update_distribution)
        self.distribution = None
        # disable args validation for speedup
        Normal.set_default_validate_args(False)

    def reset(self, dones=None):
        pass

    def forward(self):
        raise NotImplementedError

    @property
    def action_mean(self):
        return self.distribution.mean

    @property
    def action_std(self):
        return self.distribution.stddev

    @property
    def entropy(self):
        return self.distribution.entropy().sum(dim=-1)

    def update_distribution(self, obs,hist_encoding):
        # compute mean
        mean = self.actor(obs,hist_encoding)
       # print(f"Shape of obs passed ot update distirbution: {obs.shape}")
       # print(f"Shape of output of forward MLP pass {mean.shape}")
        # compute standard deviation

       # print(f"Shape of self.std: {self.std.shape}")
        if self.noise_std_type == "scalar":
            std = self.std.expand_as(mean)
        elif self.noise_std_type == "log":
           # print(f"self.log_std: {self.log_std}")
          #  print(f"self.log_std shape: {self.log_std.shape}")
            std = torch.exp(self.log_std).expand_as(mean)
           # print(f"std: {std}")
           # print(f"std shape: {std.shape}")

        else:
            raise ValueError(f"Unknown standard deviation type: {self.noise_std_type}. Should be 'scalar' or 'log'")
        # create distribution

        if torch.isnan(std).any():
            print("STD CONTAINS NaNs")
            print(std)

            print("also look at the mean:")
            print(mean)
            raise RuntimeError("std contains NaNs")

        if (std < 0).any():
            print("STD CONTAINS NEGATIVE VALUES")
            print(std)
            print("also look at the mean:")
            print(mean)
            raise RuntimeError("std contains negative values")




        #print(f"Shape of std after expand: {std.shape}")
        self.distribution = Normal(mean, std)
       

    # def act(self, obs, **kwargs):
    #     obs = self.get_actor_obs(obs)
    #     obs = self.actor_obs_normalizer(obs)
    #     self.update_distribution(obs)
    #     return self.distribution.sample()
    def act(self, obs,hist_encoding=False, **kwargs):
        obs = self.get_actor_obs(obs)
        obs = self.actor_obs_normalizer(obs)
        self.update_distribution(obs,hist_encoding)
        return self.distribution.sample()

    # def act_inference(self, obs):
    #     obs = self.get_actor_obs(obs)
    #     obs = self.actor_obs_normalizer(obs)
    #     return self.actor(obs)
    
    def act_inference(self, obs, hist_encoding=False):
        obs = self.get_actor_obs(obs)
       # print(f"Estimated Velocity: {obs[:, :3]}")

        obs = self.actor_obs_normalizer(obs)
        return self.actor(obs,hist_encoding)

    # def act_inference(self, observations, hist_encoding=False, eval=False, scandots_latent=None, **kwargs):
    #     if not eval:
    #         actions_mean = self.actor(observations, hist_encoding, eval, scandots_latent)
    #         return actions_mean
    #     else:
    #         actions_mean, latent_hist, latent_priv = self.actor(observations, hist_encoding, eval=True)
    #         return actions_mean, latent_hist, latent_priv

    def evaluate(self, obs, **kwargs):
        obs = self.get_critic_obs(obs)
        obs = self.critic_obs_normalizer(obs)
        return self.critic(obs)

    def get_actor_obs(self, obs):
        obs_list = []
        for obs_group in self.obs_groups["policy"]:
            obs_list.append(obs[obs_group])
        return torch.cat(obs_list, dim=-1)

    def get_critic_obs(self, obs):
        obs_list = []
        for obs_group in self.obs_groups["critic"]:
            obs_list.append(obs[obs_group])
        return torch.cat(obs_list, dim=-1)

    def get_actions_log_prob(self, actions):
        return self.distribution.log_prob(actions).sum(dim=-1)

    def update_normalization(self, obs):
        if self.actor_obs_normalization:
            actor_obs = self.get_actor_obs(obs)
            self.actor_obs_normalizer.update(actor_obs)
        if self.critic_obs_normalization:
            critic_obs = self.get_critic_obs(obs)
            self.critic_obs_normalizer.update(critic_obs)

    def load_state_dict(self, state_dict, strict=True):
        """Load the parameters of the actor-critic model.

        Args:
            state_dict (dict): State dictionary of the model.
            strict (bool): Whether to strictly enforce that the keys in state_dict match the keys returned by this
                           module's state_dict() function.

        Returns:
            bool: Whether this training resumes a previous training. This flag is used by the `load()` function of
                  `OnPolicyRunner` to determine how to load further parameters (relevant for, e.g., distillation).
        """

        super().load_state_dict(state_dict, strict=strict)
        return True  # training resumes

def get_activation(act_name):
    if act_name == "elu":
        return nn.ELU()
    elif act_name == "selu":
        return nn.SELU()
    elif act_name == "relu":
        return nn.ReLU()
    elif act_name == "crelu":
        return nn.ReLU()
    elif act_name == "lrelu":
        return nn.LeakyReLU()
    elif act_name == "tanh":
        return nn.Tanh()
    elif act_name == "sigmoid":
        return nn.Sigmoid()
    else:
        print("invalid activation function!")
        return None