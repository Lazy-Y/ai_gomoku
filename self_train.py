import gym
from gomoku import GomokuEnv, GomokuState, Board
from ai import AI, BOARD_SIZE
from time import sleep
from util import gomoku_util

player_color = 'black'

# default 'beginner' level opponent policy 'random'
# env = GomokuEnv(player_color=player_color,
#                 opponent='medium', board_size=BOARD_SIZE)
# env.reset()


def get_oppo_color(color):
    return 'black' if color == 'white' else 'black'


ai = AI(player_color)
ai.load()

oppo = AI(get_oppo_color(player_color))
oppo.load('policy_v3.h5')
for j in range(1):
    total_win = 0.
    total_draw = 0.
    total_lose = 0.
    TRAIN_ROUND = 10
    for i in range(TRAIN_ROUND):
        done = False
        state = GomokuState(
            Board(BOARD_SIZE), gomoku_util.BLACK)
        turn = 'black'
        steps = 0
        prev_action = None
        while True:
            player = ai if turn == 'black' else oppo
            action = player.play(state.board.board_state, prev_action)
            prev_action = action
            state = state.act(action)
            steps += 1
            exist, win_color = gomoku_util.check_five_in_row(
                state.board.board_state)  # 'empty', 'black', 'white'
            done = state.board.is_terminal()
            print(state.board)
            if done:
                # print(state.board)
                print('round', i, 'completed takes', steps, 'steps')
                if win_color == player_color:
                    print('win')
                    ai.reward(1)
                    total_win += 1
                elif win_color == 'empty':
                    total_draw += 1
                    print('draw')
                else:
                    total_lose += 1
                    ai.reward(-1)
                    print('lose')
                break
            turn = get_oppo_color(turn)
        ai.save()
    print('win:', total_win / TRAIN_ROUND, 'draw:', total_draw /
          TRAIN_ROUND, 'lose:', total_lose/TRAIN_ROUND)
