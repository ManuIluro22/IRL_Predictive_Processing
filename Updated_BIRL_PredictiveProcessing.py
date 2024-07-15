
import torch
import torch.optim as optim
import pandas as pd
import numpy as np
import time
from multiprocessing import Pool, cpu_count

# Check if CUDA is available and set the device accordingly
device = 'cpu'#torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class DataLoader:
    def __init__(self, file_path):
        self.data = pd.read_excel(file_path)

    def get_columns_by_keyword(self, keyword):
        return [col for col in self.data.columns if keyword in col]

    def get_datasets(self, rating_keyword, fulfilled_keyword):
        rating_columns = self.get_columns_by_keyword(rating_keyword)
        fulfilled_columns = self.get_columns_by_keyword(fulfilled_keyword)
        actions_dataset = self.data[rating_columns] - 1
        states_dataset = self.data[fulfilled_columns]
        return actions_dataset, states_dataset



class ParticipantOptimizer:
    def __init__(self, states, actions,n_states,n_actions, initial_rewards, Rmin=-3, Rmax=3):
        self.states = torch.tensor(states, dtype=torch.int64, device=device)
        self.actions = torch.tensor(actions, dtype=torch.int64, device=device)
        self.n_states = n_states
        self.n_actions = n_actions

        self.best_rewards = initial_rewards
        self.Rmin = Rmin
        self.Rmax = Rmax

        self.conv = 0

        self.list_rewards = None
        self.old_rewards = self.list_rewards

        self.beta_no_match = None
        self.beta_match = None

    def calculate_rewards_individual(self, states, actions, rewards_matrix):
        rewards_tensor = torch.tensor(rewards_matrix, device=device)
        probabilities = torch.nn.functional.softmax(rewards_tensor[states], dim=1)
        selected_probabilities = probabilities.gather(1, actions.unsqueeze(1)).squeeze(1)
        loss = -torch.sum(torch.log(selected_probabilities))
        return loss, rewards_tensor

    def find_initial_rewards(self, num_trials=100):
        best_loss = float('inf')
        best_rewards = None

        for _ in range(num_trials):
            trial_rewards = np.random.uniform(self.Rmin, self.Rmax, (self.n_states, self.n_actions))
            loss, _ = self.calculate_rewards_individual(self.states, self.actions, trial_rewards)
            if loss < best_loss:
                best_loss = loss
                best_rewards = trial_rewards
        self.best_rewards = best_rewards


    def _get_rewards_list(self, rewards =None):
        if rewards is None:
            rewards = self.best_rewards

        rewards_list = []
        for state, action in zip(self.states, self.actions):
            rewards_list.append(rewards[state][action])

        self.list_rewards = torch.tensor(rewards_list, dtype=torch.float32)

    def optimize(self, num_iterations=100, learning_rate=0.05):


        self.find_initial_rewards()

        for m in range(num_iterations):
            beta_unconstrained_match = torch.tensor([-2.8], requires_grad=True, device=device)
            beta_unconstrained_no_match = torch.tensor([-2.8], requires_grad=True, device=device)
            optimizer = optim.Adam([beta_unconstrained_match, beta_unconstrained_no_match], lr=learning_rate)
            self._get_rewards_list()
            for epoch in range(125):
                optimizer.zero_grad()
                beta_match = torch.sigmoid(beta_unconstrained_match)
                beta_no_match = torch.sigmoid(beta_unconstrained_no_match)

                loss, _, probability_data= self.simulate(beta_match, beta_no_match)
                loss.backward()
                optimizer.step()

            self.RandomWalk(0.05,beta_match,beta_no_match,probability_data)
            if (self.conv == 50):
                break

        self.beta_match = beta_match
        self.beta_no_match = beta_no_match



    def perturb_rewards(self, scale=0.02):
        perturbation = np.random.uniform(-scale, scale, self.best_rewards.shape)
        new_rewards = np.clip(self.best_rewards + perturbation, self.Rmin, self.Rmax)
        return new_rewards
    def RandomWalk(self, distance, beta_match, beta_no_match, old_probability):
        better = False
        self.old_rewards = self.list_rewards
        self.conv = 0
        while(not better and self.conv < 50):
            new_rewards_matrix = self.perturb_rewards(distance)
            self._get_rewards_list(new_rewards_matrix)

            Q_values = torch.tensor(new_rewards_matrix, requires_grad=True, device=device)
            Q_updated = Q_values.clone()
            new_probability_data = 1
            new_loss = 0

            for i in range(1, len(self.states)):

                state = self.states[i]
                action = self.actions[i]
                reward = self.list_rewards[i - 1]
                # Update Q_values on a new tensor to avoid in-place operations
                probabilities = torch.nn.functional.softmax(Q_updated, dim=1)  # Assuming Q_updated is indexed appropriately
                selected_probability = probabilities[state].gather(0, action)
                new_probability_data *= selected_probability
                new_loss -= selected_probability*torch.log(selected_probability)

                ## Maybe different beta for sequence s0->s0, s0->s1, s1->s0, s1->s1
                if (self.states[i - 1] == 0):
                    beta = beta_no_match
                else:
                    beta = beta_match
                new_Q_value = beta * (Q_updated[state, action].clone() - reward) + Q_updated[state, action].clone()
                Q_updated[state, action] = new_Q_value
            if new_probability_data / old_probability < 1:
                prob = new_probability_data / old_probability / 2
            else:
                prob = new_probability_data / old_probability

            if np.random.random() < min(1, prob):
                better = True
                self.best_rewards = new_rewards_matrix
            self.conv += 1



    def simulate(self, beta_match, beta_no_match):

        Q_values = torch.tensor(self.best_rewards, requires_grad=True, device=device)
        Q_updated = Q_values.clone()
        loss = 0
        probability_data = 1
        for i in range(1, len(self.states)):
            state = self.states[i]
            action = self.actions[i]
            reward = self.list_rewards[i-1]  # Assuming rewards align with states/actions
            probabilities = torch.nn.functional.softmax(Q_updated, dim=1)
            selected_probability = probabilities[state].gather(0, action)
            probability_data *= selected_probability
            loss -= selected_probability*torch.log(selected_probability)
            beta = beta_no_match if self.states[i-1] == 0 else beta_match
            new_Q_value = beta * (Q_updated[state, action].clone() - reward) + Q_updated[state, action].clone()
            Q_updated[state, action] = new_Q_value

        return loss, Q_updated,probability_data

def optimize_participant(participant_index, states, actions, n_states, n_actions):
    start_time = time.time()
    initial_rewards = np.random.uniform(-3, 3, (n_states, n_actions))
    optimizer = ParticipantOptimizer(states, actions, n_states, n_actions, initial_rewards)
    optimizer.optimize()
    end_time = time.time()
    execution_time = end_time - start_time
    print(f"Participant {participant_index} completed in {execution_time:.2f} seconds")
    return participant_index, optimizer.best_rewards, optimizer.beta_match.item(), optimizer.beta_no_match.item(), execution_time

# Example Usage
if __name__ == '__main__':
    data_loader = DataLoader('RETOS_BEBRASK_long.xlsx')
    actions_dataset, states_dataset = data_loader.get_datasets('Rating0', 'Fulfilled')

    participants = states_dataset.index
    n_states = 2  # Example, replace with the actual number of states
    n_actions = 4  # Example, replace with the actual number of actions

    results = pd.DataFrame(index=participants, columns=['best_rewards', 'beta_match', 'beta_no_match'])
    total_start_time = time.time()

    for participant in participants:
        states = states_dataset.loc[participant]
        actions = actions_dataset.loc[participant]
        participant_index, best_rewards, beta_match, beta_no_match, _ = optimize_participant(participant, states, actions, n_states, n_actions)
        results.at[participant_index, 'best_rewards'] = best_rewards
        results.at[participant_index, 'beta_match'] = beta_match
        results.at[participant_index, 'beta_no_match'] = beta_no_match

    total_end_time = time.time()
    total_execution_time = total_end_time - total_start_time

    print(results)
    print(f"Total execution time: {total_execution_time:.2f} seconds")

    results.to_csv('optimization_results.csv')
    results.to_excel('optimization_results.xlsx', index=True)
    print("Results saved to CSV and Excel.")




