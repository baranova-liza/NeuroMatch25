
import torch
import numpy as np
from itertools import combinations

from ctd.task_modeling.task_env.task_env import RandomTarget
import torch
import numpy as np
from itertools import combinations

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
        angs = torch.tensor([sho_ang, elb_ang, 0, 0], dtype=torch.float32, device=self.device)

        # Override starting position manually to (0, 0)
        start_xy = torch.tensor([[0.0, 0.0]], dtype=torch.float32, device=self.device)
        return dict(
            ics_joint=angs,
            ics_xy=start_xy,
            goal=target_pos
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
        return torch.cat(vecs, dim=-1)

    def reset(self, options=None):
        self.elapsed = 0.0
        self.trial_info = self.generate_trial_info()
        self.timestep = 0
        self.done = False
        obs_vector = self.construct_input_vector(self.trial_info)
        return obs_vector, self.trial_info


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