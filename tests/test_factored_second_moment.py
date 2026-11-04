import itertools

import pytest
import torch

from prodigyplus import ProdigyPlusScheduleFree


SHAPES = [(64, 32), (32, 64), (32, 32)] + list(itertools.permutations((64, 32, 8)))
SHAPES += [(64, 1, 32), (64, 8, 4, 32)]


@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize(
    "dtype,factored_fp32",
    [(torch.float32, True), (torch.bfloat16, True), (torch.bfloat16, False)],
)
def test_factored_denominator_matches_dense_marginals(shape, dtype, factored_fp32):
    p = torch.nn.Parameter(torch.zeros(shape, dtype=dtype))
    p.grad = torch.ones_like(p)
    optimizer = ProdigyPlusScheduleFree(
        [p], factored=True, factored_fp32=factored_fp32, split_groups=False
    )
    group = optimizer.param_groups[0]
    state = optimizer.initialise_state(p, group)
    dr, dc = state["exp_avg_sq_metadata"]
    dense_second_moment = torch.zeros(shape)
    generator = torch.Generator().manual_seed(1234)

    for step, d in enumerate((0.03, 0.08, 0.12)):
        # Nonuniform, nonseparable inputs distinguish a wrong reduction axis.
        grad = torch.rand(shape, generator=generator).mul_(4).add_(0.1)
        grad *= torch.linspace(0.2, 3, shape[-1])
        grad = grad.to(dtype).float()
        group["d"] = d
        beta2 = 0.8
        dense_second_moment.mul_(beta2).addcmul_(
            grad, grad, value=(1 - beta2) * d ** 2
        )
        actual = optimizer.update_second_moment(state, group, grad, beta2, p)

        # Adafactor section 3: outer product of the two marginals / total mass.
        # Use sums over the full dense EMA as an independent reference.
        total = dense_second_moment.sum(dim=(dr, dc), keepdim=True)
        expected = (
            dense_second_moment.sum(dim=dr, keepdim=True)
            * dense_second_moment.sum(dim=dc, keepdim=True)
            / total
        ).sqrt()
        assert actual.shape == p.shape
        tolerance = 0.03 if not factored_fp32 else 2e-5
        torch.testing.assert_close(
            actual.float(), expected, rtol=tolerance, atol=1e-7,
            msg=lambda msg: f"step={step}, shape={shape}, axes={(dr, dc)}: {msg}",
        )

        # Reconstruction must leave the stored factors unchanged.
        row = state["exp_avg_sq_row"].clone()
        col = state["exp_avg_sq_col"].clone()
        optimizer.get_denom(state, group)
        assert torch.equal(state["exp_avg_sq_row"], row)
        assert torch.equal(state["exp_avg_sq_col"], col)


@pytest.mark.parametrize("shape", [(64, 32), (32, 64), (32, 32)])
@pytest.mark.parametrize("use_schedulefree", [True, False])
def test_rank_one_updates_match_unfactored_second_moment(shape, use_schedulefree):
    # Rank-one squared gradients have an exact factored reconstruction.
    parameters = [torch.nn.Parameter(torch.zeros(shape)) for _ in range(2)]
    optimizers = [
        ProdigyPlusScheduleFree(
            [p], d0=0.01, factored=factored, use_schedulefree=use_schedulefree,
            use_stableadamw=False, stochastic_rounding=False, split_groups=False,
        )
        for p, factored in zip(parameters, (True, False))
    ]
    gradient = torch.outer(torch.linspace(1, 3, shape[0]), torch.linspace(1, 7, shape[1]))
    for scale in (0.2, 0.5, 0.3):
        for p, optimizer in zip(parameters, optimizers):
            p.grad = gradient * scale
            optimizer.step()
        torch.testing.assert_close(parameters[0], parameters[1], rtol=2e-5, atol=1e-7)


@pytest.mark.parametrize("shape", [(64, 32), (32, 64), (64, 8, 32)])
def test_zero_factored_moments_have_finite_zero_denominator(shape):
    p = torch.nn.Parameter(torch.zeros(shape))
    p.grad = torch.zeros_like(p)
    optimizer = ProdigyPlusScheduleFree([p], split_groups=False)
    group = optimizer.param_groups[0]
    state = optimizer.initialise_state(p, group)
    denom = optimizer.update_second_moment(state, group, p.grad, 0.99, p)
    assert torch.equal(denom, torch.zeros_like(p))
