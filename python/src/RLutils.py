import torch
import random
from torch_geometric.data import HeteroData
import random
from torch_geometric.data import Data, Batch
import torch.nn.functional as F
from torch_geometric.nn import GATConv, to_hetero, SAGEConv
import torch.nn as nn

def create_graph(
        app_features,
        infrastructural_features,
        edges_app_to_infrastructural,
        edges_app_to_app,
        edges_infrastructural_to_infrastructural,
        edges_features_app_to_infrastructural,
        edges_features_app_to_app,
        edges_features_infrastructural_to_infrastructural
):
    data = HeteroData()
    data["application"].x = app_features
    data["infrastructure"].x = infrastructural_features
    data ["application", "app_to_infrastructural", "infrastructure"].edge_index = edges_app_to_infrastructural
    data ["application", "app_to_app", "application"].edge_index = edges_app_to_app
    data ["infrastructure", "infrastructural_to_infrastructural", "infrastructure"].edge_index = edges_infrastructural_to_infrastructural
    data ["application", "app_to_infrastructural", "infrastructure"].edges_attr = edges_features_app_to_infrastructural
    data ["application", "app_to_app", "application"].edges_attr = edges_features_app_to_app
    data ["infrastructure", "infrastructural_to_infrastructural", "infrastructure"].edges_attr = edges_features_infrastructural_to_infrastructural
    return data

class GraphReplayBuffer:

    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def push(self, graph_observation, actions, rewards, next_graph_observation):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (graph_observation, actions, rewards, next_graph_observation)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        sample = random.sample(self.buffer, batch_size)
        observations = [s[0] for s in sample]
        actions = [s[1] for s in sample]
        rewards = [s[2] for s in sample]
        next_graph_observations = [s[3] for s in sample]
        return (Batch.from_data_list(observations), torch.cat(actions), torch.cat(rewards), Batch.from_data_list(next_graph_observations))

    def __len__(self):
        return len(self.buffer)

class GCN(torch.nn.Module):
    def __init__(self, hidden_dim, output_dim):
        super(GCN, self).__init__()
        self.conv1 = GATConv((-1, -1), hidden_dim, add_self_loops=False, bias=True)
        # self.conv2 = GATConv(hidden_dim, hidden_dim, add_self_loops=False, bias=True)
        # self.conv3 = GATConv(hidden_dim, hidden_dim, add_self_loops=False, bias=True)
        self.lin1 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.lin2 = torch.nn.Linear(hidden_dim, output_dim)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = torch.relu(x)
        # x = self.conv2(x, edge_index)
        # x = torch.relu(x)
        # x = self.conv3(x, edge_index)
        # x = torch.relu(x)
        x = self.lin1(x)
        x = torch.relu(x)
        x = self.lin2(x)
        return x

class DQNTrainer:
    def __init__(self, output_size):
        self.output_size = output_size
        self.model = GCN(hidden_dim=32, output_dim=output_size)
        self.target_model = GCN(hidden_dim=32, output_dim=output_size)
        self.target_model.load_state_dict(self.model.state_dict())
        self.optimizer = torch.optim.RMSprop(self.model.parameters(), lr=0.0001)
        self.replay_buffer = GraphReplayBuffer(6000)
        self.ticks = 0

    def add_experience(self, graph_observation, actions, rewards, next_graph_observation):
        self.replay_buffer.push(graph_observation, actions, rewards, next_graph_observation)

    def select_action(self, graph_observation, epsilon):
        if random.random() < epsilon:
            return [torch.tensor(random.randint(0, self.output_size - 1)) for _ in range(graph['application'].x.shape[0])]
        else:
            self.model.eval()
            with torch.no_grad():
                return self.model(graph_observation.x_dict, graph_observation.edge_index_dict)['application'].max(dim=1)[1]

    def toHetero(self, data):
        metadata = data.metadata()
        self.model = to_hetero(self.model, metadata, aggr='sum')
        self.target_model = to_hetero(self.target_model, metadata, aggr='sum')

    def train_step_dqn(self, batch_size, gamma=0.99, update_target_every=10):
        if len(self.replay_buffer) < batch_size:
            return 0

        self.model.train()
        self.optimizer.zero_grad()
        obs, actions, rewards, nextObs = self.replay_buffer.sample(batch_size)
        values = self.model(obs.x_dict, obs.edge_index_dict)['application'].gather(1, actions.unsqueeze(1))
        nextValues = self.target_model(nextObs.x_dict, nextObs.edge_index_dict)['application'].max(dim=1)[0].detach()
        targetValues = rewards + gamma * nextValues
        loss = nn.SmoothL1Loss()(values, targetValues.unsqueeze(1))
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_value_(self.model.parameters(), 1)
        self.optimizer.step()

        if self.ticks % update_target_every == 0:
            del self.target_model
            metadata = obs.metadata()
            self.target_model = GCN(hidden_dim=32, output_dim=self.output_size)
            self.target_model = to_hetero(self.target_model, metadata, aggr='sum')
            self.target_model.load_state_dict(self.model.state_dict())
        self.ticks += 1
        return loss.item()

    def model_snapshot(self, dir, iter):
        torch.save(self.model, f'{dir}/network-iteration-{iter}')

class BatteryRewardFunction:
    def compute_difference(self, observation, next_observation):
        battery_status_t1 = observation["application"].x[:, 0]
        battery_status_t2 = next_observation["application"].x[:, 0]
        rewards = battery_status_t2 - battery_status_t1
        return rewards

    def compute_threshold(self, observation, next_observation):
        battery_status = next_observation["application"].x[:, 0]
        rewards = torch.where(battery_status > 50, torch.tensor(0.), torch.tensor(-10.))
        return rewards

# Just a quick test
if __name__ == '__main__':

    graph = create_graph(
        app_features=torch.tensor([[100.0, 2.0], [100.0, 1.0], [3.0, 1.0]]),
        infrastructural_features=torch.tensor([[1.0, 1.0, 1.0], [2.0, 2.0, 2.0], [3.0, 3.0, 3.0]]),
        edges_app_to_infrastructural=torch.tensor([[0, 1, 0], [1, 2, 2]]),
        edges_app_to_app=torch.tensor([[0], [1]]),
        edges_infrastructural_to_infrastructural=torch.tensor([[1], [2]]),
        edges_features_app_to_infrastructural=torch.tensor([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]]),
        edges_features_app_to_app=torch.tensor([[1.0, 2.0]]),
        edges_features_infrastructural_to_infrastructural=torch.tensor([[2.0, 1.0]])
    )

    print(graph['application'].x.shape[0])

    graph2 = create_graph(
        app_features=torch.tensor([[100.0, 2.0], [70.0, 1.0], [30.0, 1.0]]),
        infrastructural_features=torch.tensor([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]]),
        edges_app_to_infrastructural=torch.tensor([[0, 1, 0], [1, 2, 2]]),
        edges_app_to_app=torch.tensor([[0], [1]]),
        edges_infrastructural_to_infrastructural=torch.tensor([[1], [2]]),
        edges_features_app_to_infrastructural=torch.tensor([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]]),
        edges_features_app_to_app=torch.tensor([[1.0, 2.0]]),
        edges_features_infrastructural_to_infrastructural=torch.tensor([[2.0, 1.0]])
    )

    print('---------------------------------- Checking GCN ----------------------------------')

    # Checks that the GCN is correctly created
    model = GCN(hidden_dim=32, output_dim=8)
    model = to_hetero(model, graph.metadata(), aggr='sum')
    output = model(graph.x_dict, graph.edge_index_dict)
    print(output['application'])
    print('OK!')

    print('-------------------------------- Checking Learning -------------------------------')
    # Checks learning step
    trainer = DQNTrainer(8)
    for i in range(10):
        trainer.add_experience(graph, torch.tensor([1, 2, 3]), torch.tensor([1.0, 0.0, -10.0]), graph2)
    trainer.toHetero(graph)
    trainer.train_step_dqn(batch_size=5, gamma=0.99, update_target_every=10)
    print(trainer.select_action(graph, 0.0))
    print('OK!')

    print('-------------------------------- Checking RF ---------------------------------')
    reward_function = BatteryRewardFunction()
    diff = reward_function.compute_difference(graph, graph2)
    th = reward_function.compute_threshold(graph, graph2)
    print(diff)
    print(th)