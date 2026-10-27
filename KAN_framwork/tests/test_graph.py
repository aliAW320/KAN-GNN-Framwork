import pytest
import torch
from torch_geometric.nn import GATConv, GATv2Conv, GCNConv, MessagePassing, SAGEConv

from kangnn import GraphLayer, LayerConfig, TransformConfig

OPERATORS = ["gcn", "sage", "gat", "gatv2"]


def graph():
    # Directed graph, unequal degrees, existing loop, and isolated node 4.
    return torch.randn(5, 3), torch.tensor([[0, 1, 2, 0, 2], [1, 2, 0, 2, 2]])


@pytest.mark.parametrize("operator", OPERATORS)
@pytest.mark.parametrize("loops", [True, False])
def test_matches_pyg_baseline(operator, loops):
    x, edges = graph()
    cfg = LayerConfig(operator=operator, self_loops=loops,
                      message=TransformConfig(bias=False), attention=TransformConfig(bias=False))
    ours = GraphLayer(3, 2, cfg)
    assert isinstance(ours, MessagePassing)
    if operator == "gcn":
        reference = GCNConv(3, 2, bias=False, add_self_loops=loops)
        reference.lin.weight.data.copy_(ours.source[0].weight)
    elif operator == "sage":
        reference = SAGEConv(3, 2, bias=False)
        reference.lin_l.weight.data.copy_(ours.source[0].weight)
        reference.lin_r.weight.data.copy_(ours.root[0].weight)
    elif operator == "gat":
        reference = GATConv(3, 2, bias=False, add_self_loops=loops)
        reference.lin.weight.data.copy_(ours.source[0].weight)
        reference.att_src.data.copy_(ours.att_source[0][0].weight.view(1, 1, 2))
        reference.att_dst.data.copy_(ours.att_target[0][0].weight.view(1, 1, 2))
    else:
        reference = GATv2Conv(3, 2, bias=False, add_self_loops=loops)
        # PyG applies lin_l to sources, lin_r to destinations.
        reference.lin_l.weight.data.copy_(ours.source[0].weight)
        reference.lin_r.weight.data.copy_(ours.target[0].weight)
        reference.att.data.copy_(ours.att_source[0][0].weight.view(1, 1, 2))
    torch.testing.assert_close(ours(x, edges), reference(x, edges), atol=1e-6, rtol=1e-5)


def test_weighted_gcn_matches_reference():
    x, edges = graph()
    weights = torch.tensor([0.5, 1., 2., 1.5, 0.7])
    ours = GraphLayer(3, 2, LayerConfig(message=TransformConfig(bias=False)))
    reference = GCNConv(3, 2, bias=False)
    reference.lin.weight.data.copy_(ours.source[0].weight)
    torch.testing.assert_close(ours(x, edges, weights), reference(x, edges, weights))


@pytest.mark.parametrize("operator", OPERATORS)
@pytest.mark.parametrize("basis", [None, "bspline", "fourier", "relu", "rbf"])
def test_message_update_permutation_and_backward(operator, basis):
    transform = TransformConfig(kind="mlp" if basis is None else "kan",
                                basis=basis or "rbf", basis_kwargs={"num_basis": 9})
    layer = GraphLayer(3, 3, LayerConfig(operator=operator, message=transform, update=transform,
                                        residual=True, heads=2 if "gat" in operator else 1))
    x, edges = graph()
    x.requires_grad_()
    y = layer(x, edges)
    assert y.shape == (5, 3) and torch.isfinite(y).all()
    permutation = torch.tensor([2, 4, 0, 1, 3])
    inverse = torch.argsort(permutation)
    torch.testing.assert_close(layer(x[permutation], inverse[edges]), y[permutation],
                               atol=1e-6, rtol=1e-5)
    y.square().sum().backward()
    assert torch.isfinite(x.grad).all()
    for parameter in layer.parameters():
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()


@pytest.mark.parametrize("operator", OPERATORS)
def test_empty_edges_and_edge_features(operator):
    cfg = LayerConfig(operator=operator, edge_dim=2, self_loops=False,
                      edge=TransformConfig(kind="kan", basis="rbf"))
    layer = GraphLayer(3, 3, cfg)
    x, edges = graph()
    attr = torch.randn(edges.size(1), 2, requires_grad=True)
    y = layer(x, edges, edge_attr=attr)
    y.sum().backward()
    assert attr.grad is not None and torch.isfinite(attr.grad).all()
    assert torch.isfinite(layer(x, torch.empty(2, 0, dtype=torch.long),
                               edge_attr=torch.empty(0, 2))).all()


@pytest.mark.parametrize("operator", ["gat", "gatv2"])
@pytest.mark.parametrize("basis", ["bspline", "fourier", "relu", "rbf"])
def test_kan_attention_and_self_loop_edge_features(operator, basis):
    layer = GraphLayer(3, 3, LayerConfig(operator=operator, heads=3, edge_dim=2,
                                        attention=TransformConfig(kind="kan", basis=basis,
                                                                  basis_kwargs={"num_basis": 9})))
    x, edges = graph()
    y = layer(x, edges, edge_attr=torch.ones(edges.size(1), 2))
    y.square().sum().backward()
    assert torch.isfinite(y).all()
    assert layer.att_source[0][0].coefficients.grad.abs().sum() > 0


def test_validation():
    x, edges = graph()
    with pytest.raises(ValueError):
        GraphLayer(3, 2)(x, edges.float())
    with pytest.raises(ValueError):
        GraphLayer(3, 2)(x, edges + 10)
    with pytest.raises(ValueError):
        GraphLayer(3, 2, LayerConfig(operator="sage"))(x, edges, torch.ones(edges.size(1)))
    with pytest.raises(ValueError):
        GraphLayer(3, 2)(x, edges, -torch.ones(edges.size(1)))
    with pytest.raises(ValueError):
        GraphLayer(3, 2, LayerConfig(edge_dim=2))(x, edges, edge_attr=torch.ones(5, 2))
    with pytest.raises(ValueError):
        LayerConfig(operator="bad")
    with pytest.raises(ValueError):
        LayerConfig(heads=2)


@pytest.mark.parametrize("operator", ["gat", "gatv2"])
def test_multihead_attention_matches_pyg(operator):
    x, edges = graph()
    ours = GraphLayer(3, 2, LayerConfig(operator=operator, heads=3,
                                       message=TransformConfig(bias=False)))
    cls = GATConv if operator == "gat" else GATv2Conv
    reference = cls(3, 2, heads=3, concat=False, bias=False)
    if operator == "gat":
        reference.lin.weight.data.copy_(ours.source[0].weight)
        reference.att_src.data.copy_(torch.stack([s[0].weight.squeeze(0)
                                                  for s in ours.att_source]).unsqueeze(0))
        reference.att_dst.data.copy_(torch.stack([s[0].weight.squeeze(0)
                                                  for s in ours.att_target]).unsqueeze(0))
    else:
        reference.lin_l.weight.data.copy_(ours.source[0].weight)
        reference.lin_r.weight.data.copy_(ours.target[0].weight)
        reference.att.data.copy_(torch.stack([s[0].weight.squeeze(0)
                                              for s in ours.att_source]).unsqueeze(0))
    torch.testing.assert_close(ours(x, edges), reference(x, edges), atol=1e-6, rtol=1e-5)


def test_dropout_modes_and_repeatability():
    from kangnn import seed_everything
    layer = GraphLayer(3, 2, LayerConfig(operator="gat", heads=2, dropout=0.5))
    x, edges = graph()
    layer.train()
    seed_everything(5)
    a = layer(x, edges)
    seed_everything(5)
    assert torch.equal(a, layer(x, edges))
    layer.eval()
    assert torch.equal(layer(x, edges), layer(x, edges))
    assert not torch.equal(a, layer(x, edges))
