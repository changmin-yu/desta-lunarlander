from typing import Optional, List
from gym import Env, spaces
import numpy as np
import matplotlib.pyplot as plt


GOAL_POS = np.array([2., 2.])
GOAL_RAD = 0.2
GOAL_RAD_2 = GOAL_RAD * GOAL_RAD
OB_POS = np.array([1.1, 1.])
OBS_RAD = 0.3
OBS_RAD_2 = OBS_RAD * OBS_RAD
ACTION_LIM = 0.2
START_POS = np.zeros((2,))

plt.ion()

class TestSafetyGym(Env):
    """
    A test safety environment for debugging.

    -----------------------------
    |                           |
    |          G                |
    |                           |
    |      x                    |
    | +                         |
    -----------------------------

    The agent starts at '+' and has to make its way to 'G' avoiding 'x'.

    The action is a 2-d velocity vector.
    As in safety gym, the "cost" is returned by `step` in the info dictionary.
    """
    def __init__(self):
        ones_dtype = np.ones((2,), dtype=np.float32)
        actions_lim = ACTION_LIM * ones_dtype
        obs_lim = np.inf * ones_dtype

        self.timestep = 0
        self.pos = START_POS
        self.done = True
        self.action_space = spaces.Box(-actions_lim, actions_lim)
        self.observation_space = spaces.Box(-obs_lim, obs_lim)
        self.trajectory = [np.array(self.pos)]
        self.count = 0
        self._setup_plot()

    def step(self, action):
        assert not self.done, "Calling step on an incomplete environment"
        assert action.shape == (2,)
        self.timestep += 1
        self.pos, reward, self.done, cost = self.pure_step(self.pos, action)
        info = {'cost': cost}
        if self.timestep > 50:
            self.done = True
        assert self.pos.shape == (2,)
        new_pos = np.array(self.pos)
        self.trajectory.append(new_pos)
        return new_pos, reward - cost * 100, self.done, info

    def reset(self):
        self.timestep = 0
        self.pos = START_POS
        self.trajectory = [np.array(self.pos)]
        self.done = False
        return np.array(self.pos)

    def render(self, mode='human', interventions: Optional[List[bool]] = None,
               cumulative_cost: float = 0):
        # only render at the end of episodes
        if self.done:
            self._plot(interventions, cumulative_cost)
            self.count += 1

    def _setup_plot(self):
        self.fig, axes = plt.subplots()
        axes.set_ylim((-1, 3))
        axes.set_xlim((-1, 3))
        axes.add_artist(plt.Circle(OB_POS, OBS_RAD, color='r'))
        axes.add_artist(plt.Circle(GOAL_POS, GOAL_RAD, color='g'))
        axes.add_artist(plt.Circle(START_POS, 0.05, color='b'))
        self.line, = axes.plot([0], [0])
        self.interventions = axes.plot([0], [0], 'r+')[0]
        self.text = axes.text(0., -0.8, 'Cumulative Safety Violations: 0', fontsize=14)

    def _plot(self, interventions: Optional[List[bool]] = None, cumulative_cost: float = 0):
        trajectory = np.array(self.trajectory)
        if interventions is not None:
            if any(interventions):
                intervention_points = np.array(
                    [point for point, intervention in zip(trajectory, interventions) if intervention])
                self.interventions.set_data(intervention_points[:, 0], intervention_points[:, 1])
            else:
                self.interventions.set_data([0], [0])
        self.line.set_data(trajectory[:, 0], trajectory[:, 1])
        self.text.set_text(f'Cumulative Safety Violations: {cumulative_cost}')
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        plt.pause(0.01)
        self.fig.savefig(f'images/{self.count:04d}.png')

    @staticmethod
    def pure_step(state, action):
        """A stateless step function (can be used as a model)"""
        action = ACTION_LIM * action / np.linalg.norm(action)
        next_state = state + action
        cost = int(np.sum(np.square(next_state - OB_POS)) <= OBS_RAD_2)
        goal_dist = np.sum(np.square(next_state - GOAL_POS))
        done = goal_dist <= GOAL_RAD_2
        return next_state, -goal_dist, done, cost

def main():
    """Small demo function"""
    env = TestSafetyGym()
    for i in range(10):
        done = False
        state = env.reset()
        while not done:
            action = env.action_space.sample()
            _, _, done, _ = env.step(action)
            env.render()


if __name__ == "__main__":
    main()
