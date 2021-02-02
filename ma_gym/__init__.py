import logging

from gym import envs
from gym.envs.registration import register

logger = logging.getLogger(__name__)

# Register openai's environments as multi agent
# This should be done before registering new environments
env_specs = [env_spec for env_spec in envs.registry.all() if 'gym.envs' in env_spec.entry_point]
for spec in env_specs:
    register(
        id='ma_' + spec.id,
        entry_point='ma_gym.envs.openai:MultiAgentWrapper',
        kwargs={'name': spec.id, **spec._kwargs}
    )

# registers env adapted from Bertsekas (2020)
grid_shape, n_agents, n_preys = [(10, 10), 4, 2]
# _game_name = 'PredatorPrey{}x{}'.format(grid_shape[0], grid_shape[1])
_game_name = 'PredatorPrey10x10'
register(
    id='{}-v4'.format(_game_name),
    entry_point='ma_gym.envs.predator_prey:PredatorPrey',
    kwargs={
        'grid_shape': grid_shape,
        'n_agents': n_agents,
        'n_preys': n_preys,
    }
)
