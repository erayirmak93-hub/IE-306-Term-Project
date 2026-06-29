from collections import deque
import random
import numpy as np


class ReplayBuffer:

    def __init__(self, capacity=100000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done, next_action_mask):
        self.buffer.append(
            (
                state,
                action,
                reward,
                next_state,
                done,
                next_action_mask
            )
        )

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)

        states, actions, rewards, next_states, dones, next_action_masks = zip(*batch)

        return (
            np.array(states),
            np.array(actions),
            np.array(rewards),
            np.array(next_states),
            np.array(dones),
            np.array(next_action_masks)
        )

    def __len__(self):
        return len(self.buffer)


if __name__ == "__main__":

    buffer = ReplayBuffer()

    for i in range(100):
        buffer.push(
            np.zeros(581),
            i,
            1.0,
            np.ones(581),
            False,
            np.ones(169)
        )

    print("Buffer size:", len(buffer))

    sample = buffer.sample(32)

    states, actions, rewards, next_states, dones, next_action_masks = sample

    print("Sample states:", states.shape)
    print("Sample action masks:", next_action_masks.shape)