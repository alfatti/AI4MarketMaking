import numpy as np


def simulate_markov_batch(Q, initial_states, times, num_simulations):
    num_states = len(Q)
    num_times = len(times)

    states_at_times = np.zeros((num_simulations, num_times), dtype=int)

    for sim in range(num_simulations):
        state = initial_states
        states_at_times[sim, 0] = state

        for i in range(1, num_times):
            current_time = times[i - 1]
            end_time = times[i]
            while current_time < end_time:
                rate = -Q[state, state]
                time_to_next = np.random.exponential(1 / rate)

                current_time += time_to_next
                if current_time < end_time:
                    transition_probs = Q[state, :] / rate
                    transition_probs[state] = 0
                    state = np.random.choice(num_states, p=transition_probs)

            states_at_times[sim, i] = state

    return states_at_times

def simulate_markov_batch_1(Q, initial_states, times, num_simulations , delta_t):
    num_states = len(Q)
    num_times = len(times)
    states_at_times = np.zeros((num_simulations, num_times), dtype=int)

    for sim in range(num_simulations):
        state = initial_states
        states_at_times[sim, 0] = state
        q = Q * delta_t
        I = np.identity(num_states)
        p = np.zeros((1,num_states))
        p[0,state] = 1
        for i in range(1, num_times):
          p = np.matmul(p,q+I)
          #print(p)
          state = np.random.choice(num_states, p=np.squeeze(p))
          states_at_times[sim, i] = state

    return states_at_times