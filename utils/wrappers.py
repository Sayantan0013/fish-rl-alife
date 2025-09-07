
from gymnasium import spaces
from gymnasium.core import Wrapper
from itertools import cycle
import numpy as np

OBSERVER_DATA_SIZE = 2


class EnvWrapper(Wrapper):
    def __init__(self, env):
        self.env = env
        self.env.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,))
        self.n = OBSERVER_DATA_SIZE + args.observe_time + 2 * args.observe_position +\
            self.env.observable_sharks * 3 +\
            self.env.observable_fishes * 3 +\
            self.env.observable_walls * 2
        self.env.observation_space = spaces.Box(
            low=-1.0, high=1.0, shape=(self.n,)
        )
        self.env.reward_range = (-float('inf'), float('inf'))
        self.env.spec = None
        self.env.metadata = {'render.modes': ['human']}
        self.env.num_envs = 1
        Wrapper.__init__(self, env=env)

    def step(self, action, *args, **kwargs):
        sharks = list(self.env.sharks)
        if not sharks:
            return ([0.] * self.n, 0, True, {})
        shark = sharks[0]
        action = (action[0], action[1], False)
        obs, reward, done = self.env.step({shark.name: action})
        shark = next(iter(done.keys()))
        return (
            obs.get(shark, np.array([0.] * self.n)),
            reward[shark],
            done[shark],
            {}
        )

    def reset(self, *args, **kwargs):
        obs = self.env.reset()
        shark = next(iter(obs.keys()))
        return obs[shark]

class MultiAgentEnvWrapper(Wrapper):
    def __init__(self, env, args):
        self.env = env
        self.num_sharks = len(list(env.sharks))
        self.env.action_space = spaces.Box(low=-1.0, high=1.0, shape=(self.env.max_sharks,2))
        self.n = OBSERVER_DATA_SIZE + args.observe_time + 2 * args.observe_position +\
            self.env.observable_sharks * (3 if args.angle else 5) +\
            self.env.observable_fishes * (3 if args.angle else 5) +\
            self.env.observable_walls * 2
        self.env.observation_space = spaces.Box(
            low=-1.0, high=1.0, shape=(self.env.max_sharks* self.n,)
        )
        self.env.reward_range = (-float('inf'), float('inf'))
        self.env.spec = None
        self.env.metadata = {'render.modes': ['human']}
        self.env.num_envs = 1

        self.last_obs = None
        Wrapper.__init__(self, env=env)

        self.step_count = 0



    def step(self, actions, *args, **kwargs):

        # self.env.sharks is a set, so the careful reader might lament the lack
        # of order in sets here. But this is fine. Sets still have an order,
        # it's just non-intuitive for the user - it's simply the hash order.
        # They are kept in that hash order in memory and they are returned in
        # that order. Thus, since I don't care about which shark I'm using for
        # training, as far as it's always the same one, it works out. For
        # reference, I always use the first shark returned in the set. That
        # first shark could be *any* shark from the set. But at least it's
        # always the same shark.
        # NOTE: If sharks are allowed to procreate, the assumptions do not hold
        # any longer. Adding new sharks in the middle of an episode may change
        # the order (e.g. new shark becomes the first shark in the internal
        # hash table, suddenly the shark we train with has changed).
        sharks = list(self.env.sharks)

        if not sharks:
            # TODO .. yikes
            return ([0.] * self.n, 0, True, {})
        joint_action = {}
        if(len(actions.shape)>2):
            actions = actions[0]

        for i, shark in enumerate(sharks):
            # if i != 0:
            #     action = model_inference(self.model, self.last_obs[shark.name])
            action = (actions[i][0], actions[i][1], False)
            joint_action[shark.name] = action

        obs, reward, done = self.env.step(joint_action)
        self.last_obs = obs
        observations = []
        dones = []
        rewards = []
        for shark in sharks:
            observations.append(obs.get(shark.name, np.array([0.] * self.n)))
            dones.append(done.get(shark.name,False))
            rewards.append(reward.get(shark.name,0.0))
        self.step_count += 1

        return (
            np.concatenate(observations),
            np.mean(rewards),
            np.all(dones),
            False,
            {}
        )

    def reset(self, *args, **kwargs):
        obs = self.env.reset()
        self.step_count = 0
        sharks = list(self.env.sharks)
        self.last_obs = obs
        observations = []
        for shark in sharks:
            observations.append(obs.get(shark.name, np.array([0.] * self.n)))
        return np.concatenate(observations), {}

    def render(self, render_mode='human', **kwargs):
        return self.env.render(render_mode=render_mode, **kwargs)


class MultiAgentEnvAECWrapper(Wrapper):
    def __init__(self, env):
        self.env = env
        self.num_sharks = len(list(self.env.sharks))
        self._agents = None
        self.selected_agent = None
        self.env.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,))
        self.n = OBSERVER_DATA_SIZE + self.env.observable_sharks * 3 +\
            self.env.observable_fishes * 3 +\
            self.env.observable_walls * 2
        self.env.observation_space = spaces.Box(
            low=-1.0, high=1.0, shape=(self.n,)
        )
        self.env.reward_range = (-float('inf'), float('inf'))
        self.env.spec = None
        self.env.metadata = {'render.modes': ['human']}
        self.env.num_envs = 1

        self.joint_action : dict = {}
        self.last_obs : dict = {}
        self.last_reward : dict = {}
        self.last_done : dict = {}
        Wrapper.__init__(self, env=env)
        self.log_dir = "./ppo_tensorboard_logs/"

        self.step_count = 0

    def iter_agent(self):
        self.selected_agent = next(self._agents)

    def reset(self, *args, **kwargs):
        self.last_obs = self.env.reset()
        self.joint_action = {}
        self._agents = cycle(self.env.sharks)
        self.iter_agent()
        self.step_count = 0
        obs = self.last_obs.pop(self.selected_agent.name,np.array([0.] * self.n))
        self.iter_agent()
        return obs, {}

    def step(self, action, *args, **kwargs):

        self.joint_action[self.selected_agent.name] = (action[0], action[1], False)

        self.iter_agent()

        if len(self.last_obs) == 0:
            # print(self.joint_action.keys())
            self.last_obs, self.last_reward, self.last_done = self.env.step(self.joint_action)

        self.step_count += 1
        return (
            self.last_obs.pop(self.selected_agent.name, np.array([0.] * self.n)),
            self.last_reward.pop(self.selected_agent.name, 0.),
            self.last_done.pop(self.selected_agent.name, False),
            False,
            {}
        )
