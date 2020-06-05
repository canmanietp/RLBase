import sys
import collections
import numpy as np

sys.path.append('../envs/PyGame-Learning-Environment/')
from ple.games.snake import Snake
from ple.ple import PLE


def flatten(l):
    if isinstance(l, dict):
        for el in l.values():
            if isinstance(el, collections.abc.Iterable) and not isinstance(el, (str, bytes)):
                yield from flatten(el)
            else:
                yield el
    else:
        for el in l:
            if isinstance(el, collections.abc.Iterable) and not isinstance(el, (str, bytes)):
                yield from flatten(el)
            else:
                yield el


def pygame_obs_into_state(obs, abstraction):
    state = list(flatten(obs))
    if abstraction is None:
        return state
    else:
        return list(np.array(state)[abstraction])


_pygame = Snake()
env = PLE(_pygame, fps=30, display_screen=True, force_fps=False)
env.init()
obs = env.getGameState()
# 'snake_head_x'
# 'snake_head_y'
# 'food_x'
# 'food_y'
# 'snake_body_0'
# 'snake_body_1'
# 'snake_body_2'
# 'snake_body_pos_00'
# 'snake_body_pos_01'
# 'snake_body_pos_10'
# 'snake_body_pos_11'
# 'snake_body_pos_20'
# 'snake_body_pos_21'
# ^ the body positions are for the number of segments the snake has

current_state = pygame_obs_into_state(obs, None)
act = env.getActionSet()[0]
reward = env.act(act)
next_state_obs = env.getGameState()
next_state = pygame_obs_into_state(next_state_obs, None)
done = env.game_over()

print(current_state, act, reward, next_state, done)

# STATE SPACE ISN'T DISCRETE!!!!