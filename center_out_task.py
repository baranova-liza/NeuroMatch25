import torch
import numpy as np
from itertools import combinations
from typing import Any, Optional, Union
# Ensure all necessary motornet imports are here
# You likely already have these if your RandomTarget works:
from ctd.task_modeling.task_env.task_env import RandomTarget
from motornet.environment import Environment # If RandomTarget inherits from Environment directly
from motornet.effector import Effector
from motornet.muscle import MujocoHillMuscle # The problematic class, but we won't modify it directly


class RandomTargetCenterOut(RandomTarget):

    def __init__(self, *args, **kwargs):
        self.input_components = ['ics_joint', 'goal', 'phase']  # <-- set something temporary
        super().__init__(*args, **kwargs)  # this calls reset()
        self.model = None

    def set_model(self, model):
        """Set model after init and determine matching input components."""
        self.model = model
        self._match_model_input()

    def _match_model_input(self):
        dummy = self.generate_trial_info()
        component_sizes = {k: v.numel() for k, v in dummy.items() if isinstance(v, torch.Tensor)}
        component_sizes['phase'] = 1  # Always include

        try:
            in_features = self.model.readout.in_features
        except AttributeError:
            in_features = self.model.input_size

        for r in range(1, len(component_sizes) + 1):
            for combo in combinations(component_sizes, r):
                total = sum(component_sizes[k] for k in combo)
                if total == in_features:
                    self.input_components = list(combo)
                    print(f"[INFO] Auto-matched inputs: {self.input_components}")
                    return

        raise ValueError(f"No valid input combination found to match model input size {in_features}")

    def generate_trial_info(self):
        """
        Generate a trial for the task.
        This is a reach to a random target from a random starting
        position with a delay period.
        """
        sho_limit = [-180, 180]  # [0, 135]  # mechanical constraints - used to be -90 180
        # sho_limit = [0, 135]  # mechanical constraints - used to be -90 180
        elb_limit = [-180, 180]  # [0, 155]
        # elb_limit = [0, 155]
        sho_ang = np.deg2rad(np.random.uniform(sho_limit[0] + 30, sho_limit[1] - 30))
        elb_ang = np.deg2rad(np.random.uniform(elb_limit[0] + 30, elb_limit[1] - 30))

        sho_ang_targ = np.deg2rad(
            np.random.uniform(sho_limit[0] + 30, sho_limit[1] - 30)
        )
        elb_ang_targ = np.deg2rad(
            np.random.uniform(elb_limit[0] + 30, elb_limit[1] - 30)
        )
        sho_ang = np.deg2rad(90)
        elb_ang = np.deg2rad(180)
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

    # def generate_trial_info(self):
    #     sho_ang = np.deg2rad(90)
    #     elb_ang = np.deg2rad(90)
    #     radius = 0.2
    #     angle = np.random.uniform(0, 2 * np.pi)
    #     x = radius * np.cos(angle)
    #     y = radius * np.sin(angle)

    #     target_pos = torch.tensor([[x, y]], dtype=torch.float32, device=self.device)
    #     angs = torch.tensor([sho_ang, elb_ang, 0, 0], dtype=torch.float32, device=self.device)

    #     # Override starting position manually to (0, 0)
    #     start_xy = torch.tensor([[0.0, 0.0]], dtype=torch.float32, device=self.device)
    #     return dict(
    #         ics_joint=angs,
    #         ics_xy=start_xy,
    #         goal=target_pos
    #     )

    def construct_input_vector(self, obs_dict):
        if not hasattr(self, "input_components"):
            raise RuntimeError("`input_components` is not set. Did you forget to call `set_model()`?")
        vecs = []
        for key in self.input_components:
            if key == 'phase':
                vecs.append(torch.tensor([[0.0]], dtype=torch.float32, device=self.device))
            else:
                val = obs_dict[key]
                if val.ndim == 1:
                    val = val.unsqueeze(0)
                vecs.append(val)
        return torch.cat(vecs, dim=-1)

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



'''
class RandomTargetCenterOut(RandomTarget):

    def __init__(self, *args, **kwargs):
        self.input_components = ['ics_joint', 'goal', 'phase']  # <-- set something temporary
        super().__init__(*args, **kwargs)  # this calls reset()
        self.model = None
    def set_model(self, model):
        """Set model after init and determine matching input components."""
        self.model = model
        self._match_model_input()

    def _match_model_input(self):
        dummy = self.generate_trial_info()
        component_sizes = {k: v.numel() for k, v in dummy.items() if isinstance(v, torch.Tensor)}
        component_sizes['phase'] = 1  # Always include
        in_features = getattr(self.model, "input_size", None)
        if in_features is None:
            raise ValueError("Model has no `.input_size` attribute.")
        print(f"[DEBUG] Model expects input size: {in_features}")
        print(f"[DEBUG] Component sizes: {component_sizes}")
        for r in range(1, len(component_sizes)+1):
            for combo in combinations(component_sizes, r):
                total = sum(component_sizes[k] for k in combo)
                if total == in_features:
                    self.input_components = list(combo)
                    print(f"[INFO] Auto-matched inputs: {self.input_components}")
                    return

        raise ValueError(f"No valid input combination found to match model input size {in_features}")

    def generate_trial_info(self):
        sho_ang = np.deg2rad(90)
        elb_ang = np.deg2rad(90)
        radius = 0.2

        angle = np.random.uniform(0, 2 * np.pi)
        x = radius * np.cos(angle)
        y = radius * np.sin(angle)

        target_pos = torch.tensor([[x, y]], dtype=torch.float32, device=self.device)

        # working version
        angs = torch.tensor([sho_ang, elb_ang, 0, 0], dtype=torch.float32, device=self.device)
        # working version
        # Override starting position manually to (0, 0)
        start_xy = torch.tensor([[0.0, 0.0]], dtype=torch.float32, device=self.device)

        # Add dummy context to match training setup
        context = torch.zeros((1, 8), dtype=torch.float32, device=self.device)  # or (1, 9) if needed

        return dict(
            ics_joint=angs,
            ics_xy=start_xy,
            goal=target_pos,
            context=context  #
        )





    def construct_input_vector(self, obs_dict):
        if not hasattr(self, "input_components"):
            raise RuntimeError("`input_components` is not set. Did you forget to call `set_model()`?")
        vecs = []
        for key in self.input_components:
            if key == 'phase':
                vecs.append(torch.tensor([[0.0]], dtype=torch.float32, device=self.device))
            else:
                val = obs_dict[key]
                if val.ndim == 1:
                    val = val.unsqueeze(0)
                vecs.append(val)
            print(f"[DEBUG] {key}: shape {val.shape}")
        return torch.cat(vecs, dim=-1)

    def reset(self, options=None):
        self.elapsed = 0.0
        self.trial_info = self.generate_trial_info()
        self.timestep = 0
        self.done = False
        obs_vector = self.construct_input_vector(self.trial_info)
        return obs_vector, self.trial_info
'''

'''
class RandomTargetCenterOut(RandomTarget):
    """Write description if it works
    """
    def generate_trial_info(self):
        """
        Generate a trial for the task.
        This is a reach to a random target from a random starting
        position with a delay period.
        """

        sho_ang = np.deg2rad(90)  # shoulder angle
        elb_ang = np.deg2rad(90)  # elbow angle

        # targets around the center
        n_dirs = 1
        # it might be too close?
        # 0.1 created None values
        radius = 0.2

        angle = np.random.choice(np.linspace(0, 2 * np.pi, n_dirs, endpoint=False))
        # coordinates of new target positions
        x = radius * np.cos(angle)
        y = radius * np.sin(angle)
        # new target position
        target_pos = torch.tensor([[x, y]], dtype=torch.float32, device=self.device)

        angs = torch.tensor(np.array([sho_ang, elb_ang, 0, 0]))

        start_xy = self.joint2cartesian(
            torch.tensor(angs, dtype=torch.float32, device=self.device)).chunk(2, dim=-1)[0]

        info = dict(
            ics_joint=angs,
            ics_xy=start_xy,
            goal=target_pos)
        return info


'''