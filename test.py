from gomoku import GomokuState, Board
from util import gomoku_util
import tensorflow as tf
from collections import defaultdict
# s = GomokuState(Board(15), gomoku_util.BLACK)
# s = s.act(0)
# s = s.act(1)

# print(s.board.board_state)


m = defaultdict(lambda: (0, 0))
print(m['a'])
