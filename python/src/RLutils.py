import torch
from torch_geometric.data import HeteroData

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


graph = create_graph(
    app_features=torch.tensor([[1, 2], [2, 1], [3 , 1]]),
    infrastructural_features=torch.tensor([[1, 1], [2, 2], [3 , 3]]),
    edges_app_to_infrastructural=torch.tensor([[0, 1, 0], [1, 2, 2]]),
    edges_app_to_app=torch.tensor([[0], [1]]),
    edges_infrastructural_to_infrastructural=torch.tensor([[1], [2]]),
    edges_features_app_to_infrastructural=torch.tensor([[1, 1], [2, 2], [3, 3]]),
    edges_features_app_to_app=torch.tensor([[1,2,3,4]]),
    edges_features_infrastructural_to_infrastructural=torch.tensor([[2]])
)
