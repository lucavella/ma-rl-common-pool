import matplotlib.pyplot as plt
import environment
import model
import agent



MAX_STEPS = 1000


def render_observation(observation):
    plt.figure(2)
    plt.imshow(observation.transpose(1,2,0))
    plt.axis('off')
    plt.show(block=False)
    plt.figure(1)


if __name__ == '__main__':
    agent_ids = list(range(10))
    game_env = environment.CommonPoolEnv(model.OPEN_MAP, agent_ids, MAX_STEPS)
    pool = agent.AgentPool(game_env, [
        agent.RandomAgent(agent_id, model.N_ACTIONS)
        for agent_id in agent_ids
    ])

    pool.reset()
    pool.render()
    render_observation(pool.observations[0])

    print(pool.game_over)

    while not pool.game_over:
        pool.step()
        print(pool.rewards)
        pool.render()
        render_observation(pool.observations[0])