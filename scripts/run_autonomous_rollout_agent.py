import time
import multiprocessing as mp

import numpy as np
import gym
import ma_gym  # register new envs on import

from src.constants import SpiderAndFlyEnv

N_AGENTS = 4

lock = mp.Lock()
can_act = mp.Condition(lock=lock)
can_update = mp.Condition(lock=lock)


def agent(i, done_n, actions, obs, is_updating, agents_acted, training_set):
    np.random.seed(i)

    while True:
        print(f'Agent {i} started.')
        print(f'Total acted: {agents_acted.value}.')

        # on entrance
        with lock:
            while is_updating.value or agents_acted.value == 4:
                can_act.wait()

        print(f'Agent {i} resumed.')

        # check if in the game
        if done_n[i]:
            break

        # select best action best on observation
        action = np.random.randint(0, 4)

        # pass action to shared memory
        actions[i] = action

        # put new experiences in the training set for NN (coordination)
        training_set.put(action)

        # on exit
        with lock:
            print(f'Agent {i} adding.')

            agents_acted.value += 1

            if agents_acted.value == 4:
                can_update.notify()

        print(f'Total acted: {agents_acted.value}.')

        with lock:
            can_act.wait()


def admin(env, done_n, actions, obs, is_updating, agents_acted, training_set):
    while True:
        # on entrance
        with lock:
            while agents_acted.value < N_AGENTS:
                can_update.wait()

            is_updating.value = True

        print('admin')

        # update Env variables -> update shared variables
        act_n = [a for a in actions]
        obs_n_new, reward_n_new, done_n_new, info = env.step(act_n)

        for i in range(len(done_n)):
            done_n[i] = done_n_new[i]

        # render the env
        time.sleep(0.1)
        env.render()

        # update NN with new experiences
        if training_set.qsize() > 100:
            while not training_set.empty():
                sample = training_set.get()

            print('Updating the weights of the Neural Net! [TODO]')

        # on exit
        with lock:
            is_updating.value = False
            agents_acted.value = 0
            can_act.notify_all()

        # end when all done
        if all(done_n):
            time.sleep(2.)
            env.close()
            break


if __name__ == '__main__':
    np.random.seed(1)

    # create Spider-and-Fly game
    env = gym.make(SpiderAndFlyEnv)

    # init env
    obs_n = env.reset()
    n_agents = env.n_agents
    n_preys = env.n_preys

    # shared variables
    done_n = mp.Array('i', [False] * n_agents)
    actions = mp.Array('i', [0] * n_agents)
    obs = mp.Array('d', obs_n[0])

    is_updating = mp.Value('i', False)
    agents_acted = mp.Value('i', 0)

    training_set = mp.Queue()

    for i in range(n_agents):
        p_agent = mp.Process(target=agent,
                             args=(i, done_n, actions, obs, is_updating, agents_acted, training_set))
        p_agent.start()

    p_admin = mp.Process(target=admin,
                         args=(env, done_n, actions, obs, is_updating, agents_acted, training_set))
    p_admin.start()
