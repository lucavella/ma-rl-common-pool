import numpy as np
import matplotlib.pyplot as plt
import environment
import model



def render_observation(observation):
    plt.figure(2)
    plt.imshow(observation.transpose(1,2,0))
    plt.axis('off')
    plt.show(block=False)
    plt.figure(1)


if __name__ == '__main__':
    n_agents = 10
    game_env = environment.CommonPoolEnv(model.DEFAULT_MAP, n_agents)
    observations, *_ = game_env.reset()
    game_env.render()
    render_observation(observations[0])

    while len(game_env.agents) > 0:
        rand_actions = np.random.randint(model.N_ACTIONS, size=n_agents)
        actions = {agent_id: action for agent_id, action in enumerate(rand_actions)}
        observations, rew, *_ = game_env.step(actions)

        print(rew)
        game_env.render()
        render_observation(observations[0])