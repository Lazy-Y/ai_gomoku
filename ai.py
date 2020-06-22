from gomoku import Board
from tensorflow.keras import losses, Sequential, Model, optimizers
from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, Dense, Activation, Softmax, Flatten
from util import gomoku_util
from collections import defaultdict
import numpy as np
import os
import time
import functools
import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

BOARD_SIZE = 9
ACTIONS = list(range(BOARD_SIZE**2))
SEARCH_SIZE = 5
REWARD_DECAY = 0.99


def board_to_tensor(board, color):
    board = np.array(board)
    if color == 'black':
        board[board == 2] = -1
    else:
        board[board == 1] = -1
        board[board == 2] = 1
    return tf.convert_to_tensor(board, dtype=tf.float32)


class PolicyNetwork(Model):
    def __init__(self):
        super(PolicyNetwork, self).__init__()
        self.network = Sequential([
            Conv2D(16, 3, padding='same'),
            Activation('tanh'),
            BatchNormalization(),
            Conv2D(16, 3, padding='same'),
            Activation('tanh'),
            BatchNormalization(),
            Flatten(),
            Dense((BOARD_SIZE * BOARD_SIZE)),
            Softmax()
        ])

    def call(self, board):
        if (board.dtype != tf.float32):
            board = tf.cast(board, dtype=tf.float32)
        board = tf.reshape(board, (1, BOARD_SIZE, BOARD_SIZE, 1))
        return self.network(board)[0]


class ValueNetwork(Model):
    def __init__(self):
        super(ValueNetwork, self).__init__()
        self.network = Sequential([
            Conv2D(16, 3, padding='same'),
            Activation('tanh'),
            BatchNormalization(),
            Conv2D(16, 3, padding='same'),
            Activation('tanh'),
            BatchNormalization(),
            Flatten(),
            Dense((1)),
        ])

    def call(self, board):
        if (board.dtype != tf.float32):
            board = tf.cast(board, dtype=tf.float32)
        board = tf.reshape(board, (1, BOARD_SIZE, BOARD_SIZE, 1))
        # print('value', board.dtype)
        return tf.tanh(self.network(board)[0, 0])


class AI:

    def __init__(self, color):
        self.color = color
        self.policy = PolicyNetwork()
        self.value = ValueNetwork()
        self.reset()

    def reset(self):
        self.actions = []
        self.oppo_actions = []
        self.board_states = []
        self.optimizer = optimizers.Adam(learning_rate=0.005)

    def save(self):
        self.policy.save_weights('policy.h5')

    def load(self, path='policy.h5'):
        self.policy.build((1, BOARD_SIZE, BOARD_SIZE, 1))
        self.policy.load_weights(path)

    def reward(self, value):
        actions = self.actions if value == 1 else self.oppo_actions
        for action, board in zip(self.actions, self.board_states):
            board_tensor = board_to_tensor(board, self.color)
            with tf.GradientTape() as tape:
                probs = self.policy(board_tensor)
                expect = np.zeros((BOARD_SIZE ** 2))
                expect[action] = 1
                expect /= expect.sum()
                loss = tf.reduce_sum(tf.square(expect - probs))
            grads = tape.gradient(loss, self.policy.trainable_variables)
            self.optimizer.apply_gradients(
                zip(grads, self.policy.trainable_variables))

    # def reward_v1(self, value):
    #     for action, board in zip(self.actions, self.board_states):
    #         board_tensor = board_to_tensor(board, self.color)
    #         with tf.GradientTape() as tape:
    #             probs = self.policy(board_tensor)
    #             if value == 1:
    #                 expect = np.zeros((BOARD_SIZE ** 2))
    #                 expect[action] = 1
    #             else:
    #                 expect = np.array(probs)
    #                 expect[action] = 0
    #                 if expect.sum() == 0:
    #                     continue
    #                 expect /= expect.sum()
    #             loss = tf.reduce_sum(tf.square(expect - probs))
    #         # print('policy network loss:', loss)
    #         grads = tape.gradient(loss, self.policy.trainable_variables)
    #         self.optimizer.apply_gradients(
    #             zip(grads, self.policy.trainable_variables))

    def play(self, board, oppo_actions=None):
        action = self.search(board)
        self.actions.append(action)
        if oppo_actions is not None:
            self.oppo_actions.append(oppo_actions)
        self.board_states.append(board)
        return action

    def get_policy_probs(self, board_tensor):
        if not tf.is_tensor(board_tensor):
            board_tensor = tf.convert_to_tensor(board_tensor)
        probs = self.policy(board_tensor)
        if tf.reduce_any(tf.math.is_nan(probs)):
            print(probs)
        if board_tensor.ndim > 1:
            board_tensor = tf.keras.backend.flatten(board_tensor)
        probs = tf.where(
            board_tensor == 0, probs, tf.zeros(BOARD_SIZE ** 2))
        probs = np.array(probs)
        s = probs.sum()
        probs /= s
        return probs

    def get_values(self, board):
        board_obj = Board(BOARD_SIZE)
        board_obj.copy(np.array(board))

    def search(self, board):
        board_obj = Board(BOARD_SIZE)
        board_obj.copy(np.array(board))
        probs = self.get_policy_probs(board_obj.board_state)
        m = defaultdict(lambda: (0, 0))
        for _ in range(SEARCH_SIZE):
            action, reward = self.search_single(board_obj, probs)
            prev_count, prev_reward = m[action]
            m[action] = (prev_count + 1., prev_reward + reward)
        max_action = 0.
        max_reward = -2.
        for action, (count, reward) in m.items():
            curr_reward = reward / count
            if max_reward < curr_reward:
                max_reward = curr_reward
                max_action = action
        return max_action

    def search_single(self, board_obj, probs):
        color = self.color
        action = np.random.choice(ACTIONS, p=probs)
        board_obj = board_obj.play(action, color)
        step = 0
        while not board_obj.is_terminal():
            color = 'white' if color == 'black' else 'black'
            action = self.choose_action(board_obj.board_state, color)
            board_obj = board_obj.play(action, color)
            step += 1
        exist, win_color = gomoku_util.check_five_in_row(
            board_obj.board_state)  # 'empty', 'black', 'white'
        if win_color == 'empty':
            reward = 0
        elif win_color == self.color:
            reward = 1
        else:
            reward = -1
        return action, reward * (REWARD_DECAY ** (step / 2))

    def choose_action(self, board, color):
        board_tensor = board_to_tensor(board, color)
        probs = self.get_policy_probs(board_tensor)
        action = np.random.choice(ACTIONS, p=probs)
        return action


# b = np.random.randint(3, size=(BOARD_SIZE, BOARD_SIZE))
# b = tf.zeros((BOARD_SIZE, BOARD_SIZE))
# ai = AI('black')
# ai.load()
# ai.policy.summary()
# print(ai.search(b))
# ai.move(b)
# b = board_to_tensor(b, 'black')
# v = ValueNetwork('black')
# r = v(b)
# print(r, )
