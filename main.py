import gym
from gomoku import GomokuEnv
from ai import AI, BOARD_SIZE
from time import sleep

player_color = 'black'

# default 'beginner' level opponent policy 'random'
env = GomokuEnv(player_color=player_color,
                opponent='beginner', board_size=BOARD_SIZE)


ai = AI(player_color)
ai.load()
total_win = total_draw = total_lose = 0
TRAIN_ROUND = 100
for i in range(TRAIN_ROUND):
    done = False
    env.reset()
    try:
        ai.reset()
        while not done:
            action = ai.play(env.state.board.board_state,
                             env.state.board.last_action)
            # print('prev action', env.state.board.last_action)
            observation, reward, done, info = env.step(action)
            # sleep(3)
            if done:
                env.render(mode='human')
                print('round', i, "Game is Over", reward)
                if reward > 0:
                    total_win += 1
                elif reward < 0:
                    total_lose += 1
                else:
                    total_draw += 1
                ai.reward(reward)
                break
            ai.save()
    except KeyboardInterrupt:
        break
    except Exception as e:
        print(e)
print('win:', total_win / TRAIN_ROUND, 'draw:', total_draw /
      TRAIN_ROUND, 'lose:', total_lose/TRAIN_ROUND)
