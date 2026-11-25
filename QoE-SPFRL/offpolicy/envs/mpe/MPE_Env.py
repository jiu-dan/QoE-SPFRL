from .environment import MultiAgentEnv
from .scenarios import load


def MPEEnv(args):

    scenario = load(args.scenario_name + ".py").Scenario()
    world = scenario.make_world(args)
    env = MultiAgentEnv(world, scenario.reset_world,
                        scenario.reward, scenario.observation)

    return env
