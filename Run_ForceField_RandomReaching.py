#SETUP FROM ORIGINAL NOTEBOOK, INCLDUING GRU TRAINING FUNCTIONS.
#select the correct virtual environment using venv, this is located in this folder: "/user/home/as15635/NeuroMatch/NeuroMatch25/.venv"
#set env:

# set the random seed for reproducibility
import random
import dotenv
import pathlib
import os
import logging
import pickle

# comment the next three lines if you want to see all training logs
pl_loggers = [logging.getLogger(name) for name in logging.root.manager.loggerDict if 'pytorch_lightning' in name]
for pl_log in pl_loggers:
    logging.getLogger(pl_log.name).setLevel(logging.WARNING)

random.seed(2024)

dotenv.load_dotenv(override=True)
HOME_DIR = os.getenv("HOME_DIR")
if HOME_DIR is None:
    HOME_DIR = ""
print(HOME_DIR, flush=True)

import torch

# LOAD RANDOM REACHING TASK and GRU and Training objects
from ctd.task_modeling.task_env.task_env import RandomTarget
from motornet.effector import RigidTendonArm26
from motornet.muscle import MujocoHillMuscle

from ctd.task_modeling.model.rnn import GRU_RNN
from ctd.task_modeling.datamodule.task_datamodule import TaskDataModule
from ctd.task_modeling.task_wrapper.task_wrapper import TaskTrainedWrapper
from pytorch_lightning import Trainer
from motornet.environment import Environment
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import torch
from torch import nn
from typing import Union, Optional, Any
from torch import Tensor
from numpy import ndarray
from ctd.task_modeling.task_env.loss_func import RandomTargetLoss

#take in arguments from command line, structured as follows: python Run_ForceField_RandomReaching.py --task_id ${SLURM_ARRAY_TASK_ID} --num_epochs 50 --num_samples 1000 --batch_size 256 --learning_rate 1e-8 --weight_decay 0.0001 --latent_size 128 --force_field_strength_x "0.1" --force_field_strength_y "0.0"
print("loading command line arguments", flush=True)
import argparse
parser = argparse.ArgumentParser(description='Run Force Field Random Reaching Task')
parser.add_argument('--task_id', type=int, default=0, help='Task ID for the SLURM array job')
parser.add_argument('--num_tasks', type=int, default=1, help='Total number of tasks to run')
parser.add_argument('--num_epochs', type=int, default=1000, help='Number of epochs to train the model')
parser.add_argument('--batch_size', type=int, default=64, help='Batch size for training')
parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate for the optimizer')
parser.add_argument('--latent_size', type=int, default=128, help='Hidden size for the GRU model')
parser.add_argument('--weight_decay', type=float, default=1e-8, help='Weight decay for the optimizer')
parser.add_argument('--num_samples', type=int, default=1000, help='Number of samples to generate for the dataset')
parser.add_argument('--force_field_strength_x', type=str, default="0.1", help='Strength of the force field to apply in the task')
parser.add_argument('--force_field_strength_y', type=str, default="0.0", help='Strength of the force field to apply in the task in the y direction')
parser.add_argument('--force_field_bool', type=str, default="FALSE", help='Whether to apply a force field in the task or not. If set to "TRUE", a force field will be applied. If set to "FALSE", no force field will be applied.')
args = parser.parse_args()
# Convert force field strengths from string to float
print("Converting force field strengths from string to float", flush=True)
args.force_field_strength_x = float(args.force_field_strength_x)
args.force_field_strength_y = float(args.force_field_strength_y)
#set force field:
print("Defining force field", flush=True)
force_field= np.array([[args.force_field_strength_x, args.force_field_strength_y]])

# DEFINE NEW FORCE FIELD RANDOMTARGET CLASS:
# forcefield class version:
print("Force field is", force_field, flush=True)
print("creating class RandomTarget_forcefield", flush=True)
class RandomTarget_forcefield(Environment):
    """A reach to a random target from a random starting position with a delay period.

    Args:
        network: :class:`motornet.nets.layers.Network` object class or subclass.
        This is the network that will perform the task.

        name: `String`, the name of the task object instance.
        deriv_weight: `Float`, the weight of the muscle activation's derivative
        contribution to the default muscle L2 loss.

        force_field: 'np.array', the force field to apply to the effector, of same dimension as the skeleton space_dim. (2d in this case)
            example 1N force to the right: np.array([[1, 0]]).

        **kwargs: This is passed as-is to the parent :class:`Task` class.
    """

    def __init__(self, *args, **kwargs):
        #putting forcefield here so it isnt used as kwarg for parent class.
        self.force_field = kwargs.pop("force_field", np.array([[0, 0]]))
        print("force field is", self.force_field, flush=True)
        super().__init__(*args, **kwargs)

        self.obs_noise[: self.skeleton.space_dim] = [
            0.0
        ] * self.skeleton.space_dim  # target info is noiseless

        self.dataset_name = "RandomTarget"
        self.n_timesteps = np.floor(self.max_ep_duration / self.effector.dt).astype(int)
        self.input_labels = ["TargetX", "TargetY", "GoCue"] #recieves three input streams, 
                                                            #target x and y specify desired location.
                                                            #go cue is a binary signal that indicates when the movement should start.
        self.output_labels = ["Pec", "Delt", "Brad", "TriLong", "Biceps", "TriLat"] #each muscle's activation is an output.
        self.context_inputs = spaces.Box(low=-2, high=2, shape=(3,), dtype=np.float32) #the three inputs, each range -2 to 2. 
        self.coupled_env = True
        self.state_label = "fingertip"

        pos_weight = kwargs.get("pos_weight", 1.0) #i think this is for the position loss,
        act_weight = kwargs.get("act_weight", 1.0) #and for the activation loss.

        self.bump_mag_low = kwargs.get("bump_mag_low", 5)
        self.bump_mag_high = kwargs.get("bump_mag_high", 10)

        # self.force_field = kwargs.get("force_field", np.array([[0, 0]]))

        self.loss_func = RandomTargetLoss(
            position_loss=nn.MSELoss(), pos_weight=pos_weight, act_weight=act_weight
        )

    def step(self, action, deterministic=False, **kwargs):
        # Add the force field to any existing endpoint_load
        force_field_tensor = torch.tensor(self.force_field, dtype=torch.float32).to(self.device)
        
        if 'endpoint_load' in kwargs:
            kwargs['endpoint_load'] = kwargs['endpoint_load'] + force_field_tensor
        else:
            kwargs['endpoint_load'] = force_field_tensor
        
        return super().step(action, deterministic=deterministic, **kwargs)

    def generate_dataset(self, n_samples):
        # Make target circular, change loss function to be pinned at zero
        initial_state = []
        inputs = np.zeros((n_samples, self.n_timesteps, 3))

        goal_list = []
        go_cue_list = []
        target_on_list = []
        catch_trials = []
        ext_inputs_list = []

        for i in range(n_samples):
            catch_trial = np.random.choice([0, 1], p=[0.8, 0.2])
            bump_trial = np.random.choice([0, 1], p=[0.5, 0.5])
            move_bump_trial = np.random.choice([0, 1], p=[0.5, 0.5])

            target_on = np.random.randint(10, 30)
            go_cue = np.random.randint(target_on, self.n_timesteps)
            if move_bump_trial:
                bump_time = np.random.randint(go_cue, go_cue + 40)
            else:
                bump_time = np.random.randint(0, self.n_timesteps - 30)
            bump_duration = np.random.randint(15, 30)
            bump_theta = np.random.uniform(0, 2 * np.pi)
            bump_mag = np.random.uniform(self.bump_mag_low, self.bump_mag_high)

            target_on_list.append(target_on)

            info = self.generate_trial_info()
            initial_state.append(info["ics_joint"])
            initial_state_xy = info["ics_xy"]


            #this is the environmental force field input. leave it set to zero for now so we can add the bump first.
            env_inputs_mat = np.zeros((self.n_timesteps, 2))

            #then it has the bump added.
            if bump_trial:
                bump_end = min(bump_time + bump_duration, self.n_timesteps)
                env_inputs_mat[bump_time:bump_end, :] = np.array(
                    [bump_mag * np.cos(bump_theta), bump_mag * np.sin(bump_theta)]
                )

            #we add the force field here, so it can then get loaded into the input_env as normal.
            env_inputs_mat += self.force_field[0,:]

            goal_matrix = torch.zeros((self.n_timesteps, self.skeleton.space_dim))
            if catch_trial:
                go_cue = -1
                goal_matrix[:, :] = initial_state_xy
            else:
                inputs[i, go_cue:, 2] = 1

                goal_matrix[:go_cue, :] = initial_state_xy
                goal_matrix[go_cue:, :] = torch.squeeze(info["goal"])

            go_cue_list.append(go_cue)
            inputs[i, target_on:, 0:2] = info["goal"]

            catch_trials.append(catch_trial)
            goal_list.append(goal_matrix)
            ext_inputs_list.append(env_inputs_mat)

        go_cue_list = np.array(go_cue_list)
        target_on_list = np.array(target_on_list)
        env_inputs = np.stack(ext_inputs_list, axis=0)
        extra = np.stack((target_on_list, go_cue_list), axis=1)
        conds = np.array(catch_trials)

        initial_state = torch.stack(initial_state, axis=0)
        goal_list = torch.stack(goal_list, axis=0)
        dataset_dict = {
            "ics": initial_state,
            "inputs": inputs,
            "inputs_to_env": env_inputs,
            "targets": goal_list,
            "conds": conds,
            "extra": extra,
            "true_inputs": inputs,
        }
        extra_dict = {}
        return dataset_dict, extra_dict

    def generate_trial_info(self):
        """
        Generate a trial for the task.
        This is a reach to a random target from a random starting
        position with a delay period.
        """
        sho_limit = [-90, 180]#[0, 135]  # mechanical constraints - used to be -90 180
        elb_limit = [-90, 180]#[0, 155]
        sho_ang = np.deg2rad(np.random.uniform(sho_limit[0] + 30, sho_limit[1] - 30))
        elb_ang = np.deg2rad(np.random.uniform(elb_limit[0] + 30, elb_limit[1] - 30))

        sho_ang_targ = np.deg2rad(
            np.random.uniform(sho_limit[0] + 30, sho_limit[1] - 30)
        )
        elb_ang_targ = np.deg2rad(
            np.random.uniform(elb_limit[0] + 30, elb_limit[1] - 30)
        )

        angs = torch.tensor(np.array([sho_ang, elb_ang, 0, 0]))
        ang_targ = torch.tensor(np.array([sho_ang_targ, elb_ang_targ, 0, 0]))

        target_pos = self.joint2cartesian(
            torch.tensor(ang_targ, dtype=torch.float32, device=self.device)
        ).chunk(2, dim=-1)[0]

        start_xy = self.joint2cartesian(
            torch.tensor(angs, dtype=torch.float32, device=self.device)
        ).chunk(2, dim=-1)[0]

        info = dict(
            ics_joint=angs,
            ics_xy=start_xy,
            goal=target_pos,
        )
        return info

    def set_goal(
        self,
        goal: torch.Tensor,
    ):
        """
        Sets the goal of the task. This is the target position of the effector.
        """
        self.goal = goal

    def get_obs(self, action=None, deterministic: bool = False) -> Union[Tensor, ndarray]:
        self.update_obs_buffer(action=action)

        obs_as_list = [
            self.obs_buffer["vision"][0],
            self.obs_buffer["proprioception"][0],
        ] + self.obs_buffer["action"][: self.action_frame_stacking]

        obs = torch.cat(obs_as_list, dim=-1)

        if deterministic is False:
            obs = self.apply_noise(obs, noise=self.obs_noise)

        return obs if self.differentiable else self.detach(obs)

    def reset(
        self,
        batch_size: int = 1,
        options: Optional[dict[str, Any]] = None,
        seed: Optional[int] = None,
    ) -> tuple[Any, dict[str, Any]]:

        """
        Uses the :meth:`Environment.reset()` method of the parent class
        :class:`Environment` that can be overwritten to change the returned data.
        Here the goals (`i.e.`, the targets) are drawn from a random uniform
        distribution across the full joint space.
        """
        sho_limit = np.deg2rad([0, 135])  # mechanical constraints - used to be -90 180
        elb_limit = np.deg2rad([0, 155])
        # Make self.obs_noise a list
        self._set_generator(seed=seed)
        # if ic_state is in options, use that
        if options is not None and "deterministic" in options.keys():
            deterministic = options["deterministic"]
        else:
            deterministic = False
        if options is not None and "ic_state" in options.keys():
            ic_state_shape = np.shape(self.detach(options["ic_state"]))
            if ic_state_shape[0] > 1:
                batch_size = ic_state_shape[0]
            ic_state = options["ic_state"]
        else:
            ic_state = self.q_init

        if options is not None and "target_state" in options.keys():
            self.goal = options["target_state"]
        else:
            sho_ang = np.random.uniform(
                sho_limit[0] + 20, sho_limit[1] - 20, size=batch_size
            )
            elb_ang = np.random.uniform(
                elb_limit[0] + 20, elb_limit[1] - 20, size=batch_size
            )
            sho_vel = np.zeros(batch_size)
            elb_vel = np.zeros(batch_size)
            angs = np.stack((sho_ang, elb_ang, sho_vel, elb_vel), axis=1)
            self.goal = self.joint2cartesian(
                torch.tensor(angs, dtype=torch.float32, device=self.device)
            ).chunk(2, dim=-1)[0]

        options = {
            "batch_size": batch_size,
            "joint_state": ic_state,
        }
        self.effector.reset(options=options)

        self.elapsed = 0.0

        action = torch.zeros((batch_size, self.action_space.shape[0])).to(self.device)

        self.obs_buffer["proprioception"] = [self.get_proprioception()] * len(
            self.obs_buffer["proprioception"]
        )
        self.obs_buffer["vision"] = [self.get_vision()] * len(self.obs_buffer["vision"])
        self.obs_buffer["action"] = [action] * self.action_frame_stacking

        action = action if self.differentiable else self.detach(action)

        obs = self.get_obs(deterministic=deterministic)
        info = {
            "states": self._maybe_detach_states(),
            "action": action,
            "noisy action": action,
            "goal": self.goal if self.differentiable else self.detach(self.goal),
        }
        return obs, info

n_epochs= args.num_epochs
print("RandomTarget_forcefield class created.", flush=True)

if args.force_field_bool == "FALSE":
    print("Running no force field random reaching task", flush=True)

    ### Run force field free training:
    

    # Create the analysis object:
    print("Creating RandomTarget task environment without force field", flush=True)
    rt_task_env_no_force_field = RandomTarget(effector = RigidTendonArm26(muscle = MujocoHillMuscle()))
    print("RandomTarget task environment without force field created", flush=True)
    # Step 1: Instantiate the model
    print("Instantiating GRU_RNN model", flush=True)
    rnn = GRU_RNN(latent_size = 128) # Look in ctd/task_modeling/models for alternative choices!
    print("GRU_RNN model instantiated", flush=True)
    print("Instantiating task environment", flush=True)
    # Step 2: Instantiate the task environment
    task_env = rt_task_env_no_force_field
    print("Task environment instantiated", flush=True)
    print("Instantiating task datamodule", flush=True)
    # Step 3: Instantiate the task datamodule
    task_datamodule = TaskDataModule(task_env, n_samples = 1000, batch_size = 256)
    print("Task datamodule instantiated", flush=True)
    print("Instantiating task wrapper", flush=True)
    # Step 4: Instantiate the task wrapper
    task_wrapper = TaskTrainedWrapper(learning_rate=1e-3, weight_decay = 1e-8)
    print("Task wrapper instantiated", flush=True)

    print("Initializing model with input and output sizes", flush=True)
    # Step 5: Initialize the model with the input and output sizes
    rnn.init_model(
        input_size = task_env.observation_space.shape[0] + task_env.context_inputs.shape[0],
        output_size = task_env.action_space.shape[0]
        )
    print("Model initialized with input and output sizes", flush=True)
    # Step 6:  Set the environment and model in the task wrapper
    print("Setting environment and model in task wrapper", flush=True)
    task_wrapper.set_environment(task_env)
    task_wrapper.set_model(rnn)
    print("Environment and model set in task wrapper", flush=True)

    print("Defining PyTorch Lightning Trainer object", flush=True)
    # Step 7: Define the PyTorch Lightning Trainer object (put `enable_progress_bar=True` to observe training progress)
    trainer = Trainer(accelerator= "cpu",max_epochs=n_epochs,enable_progress_bar=True)
    print("PyTorch Lightning Trainer object defined", flush=True)

    print("Fitting the model", flush=True)
    # Step 8: Fit the model
    trainer.fit(task_wrapper, task_datamodule)
    print("Model fitted", flush=True)

    print("Saving model and datamodule", flush=True)
    save_dir = pathlib.Path(HOME_DIR) / f"models_randtarg{n_epochs}_noforcefield"
    save_dir.mkdir(exist_ok=True)
    with open(save_dir / "model.pkl", "wb") as f:
        pickle.dump(task_wrapper, f)

    # save datamodule as .pkl
    with open(save_dir / "datamodule_sim.pkl", "wb") as f:
        pickle.dump(task_datamodule, f)

    print("Model and datamodule saved", flush=True)
elif args.force_field_bool == "TRUE":

    print("Force field training is set to TRUE, proceeding with force field training.", flush=True)

    ### Run force field training:

    #force field is defined at the start, from cl arguments.
    print("Creating RandomTarget task environment with force field", flush=True)
    rt_task_env_force_field = RandomTarget_forcefield(effector = RigidTendonArm26(muscle = MujocoHillMuscle()),force_field=force_field)
    print("RandomTarget task environment with force field created", flush=True)
    # Step 1: Instantiate the model
    print("Instantiating GRU_RNN model", flush=True)
    rnn = GRU_RNN(latent_size = 128)
    print("GRU_RNN model instantiated", flush=True)

    # Step 2: Instantiate the task environment
    print("Instantiating task environment with force field", flush=True)
    task_env = rt_task_env_force_field
    print("Task environment with force field instantiated", flush=True)

    # Step 3: Instantiate the task datamodule
    print("Instantiating task datamodule with force field", flush=True)
    task_datamodule = TaskDataModule(task_env, n_samples = 1000, batch_size = 256)
    print("Task datamodule with force field instantiated", flush=True)

    # Step 4: Instantiate the task wrapper
    print("Instantiating task wrapper with force field", flush=True)
    task_wrapper = TaskTrainedWrapper(learning_rate=1e-3, weight_decay = 1e-8)
    print("Task wrapper with force field instantiated", flush=True)

    # Step 5: Initialize the model with the input and output sizes
    print("Initializing model with input and output sizes for force field task", flush=True)
    rnn.init_model(
        input_size = task_env.observation_space.shape[0] + task_env.context_inputs.shape[0],
        output_size = task_env.action_space.shape[0]
        )
    print("Model initialized with input and output sizes for force field task", flush=True)


    # Step 6:  Set the environment and model in the task wrapper
    print("Setting environment and model in task wrapper for force field task", flush=True)
    task_wrapper.set_environment(task_env)
    task_wrapper.set_model(rnn)
    print("Environment and model set in task wrapper for force field task", flush=True)

    # Step 7: Define the PyTorch Lightning Trainer object (put `enable_progress_bar=True` to observe training progress)
    print("Defining PyTorch Lightning Trainer object for force field task", flush=True)
    trainer = Trainer(accelerator= "cpu",max_epochs=n_epochs,enable_progress_bar=True)
    print("PyTorch Lightning Trainer object defined for force field task", flush=True)

    # Step 8: Fit the model
    print("Fitting the model for force field task", flush=True)
    trainer.fit(task_wrapper, task_datamodule)
    print("Model fitted for force field task", flush=True)

    print("Saving model and datamodule for force field task", flush=True)
    save_dir = pathlib.Path(HOME_DIR) / f"models_randtarg{n_epochs}_forcefield_x{force_field[0,0]}_y{force_field[0,1]}"
    save_dir.mkdir(exist_ok=True)
    with open(save_dir / "model.pkl", "wb") as f:
        pickle.dump(task_wrapper, f)

    # save datamodule as .pkl
    with open(save_dir / "datamodule_sim.pkl", "wb") as f:
        pickle.dump(task_datamodule, f)
    print("Model and datamodule saved for force field task", flush=True)