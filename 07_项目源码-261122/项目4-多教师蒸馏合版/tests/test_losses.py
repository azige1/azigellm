import pytest


def test_custom_loss_returns_negative_weighted_advantage():
    torch = pytest.importorskip("torch")
    pytest.importorskip("verl")
    from tensordict import TensorDict

    from multi_teacher_distillation.losses import compute_sampled_reverse_kl

    advantages = torch.tensor([[0.5, 0.0, -3.0, 0.0]])
    token_weights = torch.tensor([[2.0, 1.0, 0.5, 0.0]])
    mask = torch.tensor([[1, 1, 1, 0]])
    data = TensorDict(
        {
            "response_mask": mask,
            "distillation_advantages": advantages,
            "distillation_token_weights": token_weights,
        },
        batch_size=[1],
    )
    losses, metrics = compute_sampled_reverse_kl(None, None, {}, data)
    assert torch.allclose(losses, -(advantages * token_weights))
    assert metrics["distillation/clipped_fraction"] == pytest.approx(1 / 3)
    assert metrics["distillation/mean_token_reliability_weight"] == pytest.approx((2.0 + 1.0 + 0.5) / 3.0)


def test_custom_loss_accepts_nested_tq_fields():
    torch = pytest.importorskip("torch")
    pytest.importorskip("verl")
    from tensordict import TensorDict

    from multi_teacher_distillation.losses import compute_sampled_reverse_kl

    rows = [torch.tensor([0.5, -0.25, 0.1]), torch.tensor([0.2])]
    weights = [torch.tensor([1.0, 2.0, 0.5]), torch.tensor([0.8])]
    masks = [torch.ones(3, dtype=torch.long), torch.ones(1, dtype=torch.long)]
    data = TensorDict(
        {
            "response_mask": torch.nested.nested_tensor(masks, layout=torch.jagged),
            "distillation_advantages": torch.nested.nested_tensor(rows, layout=torch.jagged),
            "distillation_token_weights": torch.nested.nested_tensor(weights, layout=torch.jagged),
        },
        batch_size=[2],
    )
    losses, _ = compute_sampled_reverse_kl(None, None, {}, data)
    assert losses.shape == (2, 3)
    assert torch.allclose(losses[0], torch.tensor([-0.5, 0.5, -0.05]))
    assert torch.allclose(losses[1, 0], torch.tensor(-0.16))
    assert torch.allclose(losses[1, 1:], torch.zeros(2))


def test_missing_advantages_field_is_an_error():
    torch = pytest.importorskip("torch")
    pytest.importorskip("verl")
    from tensordict import TensorDict

    from multi_teacher_distillation.losses import compute_sampled_reverse_kl

    data = TensorDict({"response_mask": torch.ones(1, 2, dtype=torch.long)}, batch_size=[1])
    with pytest.raises(ValueError):
        compute_sampled_reverse_kl(None, None, {}, data)


def test_domain_token_reweight_equalizes_gradient_token_shares():
    torch = pytest.importorskip("torch")
    from multi_teacher_distillation.losses import compute_domain_token_weights, reweight_domain_losses

    advantages = torch.tensor([
        [1.0, 1.0, 1.0, 1.0],
        [0.5, 0.5, 0.0, 0.0],
    ])
    mask = torch.tensor([
        [1, 1, 1, 1],
        [1, 1, 0, 0],
    ])
    target = {"repo_impl": 0.5, "instruction_following": 0.5}
    weights, diagnostics = compute_domain_token_weights(
        advantages, mask, ["repo_impl", "instruction_following"], target
    )
    assert diagnostics["actual_shares"]["repo_impl"] == pytest.approx(0.8)
    assert diagnostics["actual_shares"]["instruction_following"] == pytest.approx(0.2)
    assert diagnostics["share_gaps"]["repo_impl"] == pytest.approx(0.3)

    reweighted = reweight_domain_losses(-advantages, weights)
    repo_mass = reweighted[0][mask[0].bool()].abs().sum().item()
    if_mass = reweighted[1][mask[1].bool()].abs().sum().item()
    assert repo_mass / (repo_mass + if_mass) == pytest.approx(0.5)
    assert if_mass / (repo_mass + if_mass) == pytest.approx(0.5)
