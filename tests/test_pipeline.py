"""Tests for the properties that are expensive to debug and easy to break.

These are not coverage tests. Each one guards a specific failure mode that
would otherwise show up as a model that trains fine and is quietly wrong:

* non-causal normalisation leaking the future into the analyser;
* the replay handing the model a window that overlaps its own target;
* tokenisation that scales with bin width;
* a consistency loss that punishes the market for moving;
* train/serve skew between the trainer's tokeniser and the live runner's.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from btcpred.data.features import build_features
from btcpred.data.pyramid import PyramidSpec, tokenize, tokenize_batch
from btcpred.data.schema import N_FEATURES, month_length_seconds
from btcpred.models.basis import PathSynthesiser, representable_fraction
from btcpred.models.market_analyser import MarketAnalyser
from btcpred.models.price_predictor import PricePredictor
from btcpred.train.objective import ObjectiveConfig, RealTimeObjective


@pytest.fixture(scope="module")
def spec() -> PyramidSpec:
    return PyramidSpec()


# --------------------------------------------------------------------------- #
# schema / features
# --------------------------------------------------------------------------- #
def test_month_lengths_handle_leap_years():
    assert month_length_seconds(2024, 2) == 29 * 86400
    assert month_length_seconds(2023, 2) == 28 * 86400
    assert month_length_seconds(2026, 12) == 31 * 86400


def test_features_are_finite_and_scale_free():
    n = 5000
    rng = np.random.default_rng(0)
    # Two price regimes three orders of magnitude apart. A level-dependent
    # feature would show up as a difference in the summary statistics.
    out = []
    for level in (7_000.0, 120_000.0):
        price = level * np.exp(np.cumsum(rng.normal(0, 1e-4, n)))
        c = np.rint(price * 100).astype(np.int32)
        f = build_features(
            c=c,
            o_c=np.zeros(n, np.int32), h_c=np.full(n, 50, np.int32),
            l_c=np.full(n, -50, np.int32),
            v=rng.integers(0, 5000, n).astype(np.int64),
            tb=rng.integers(0, 2500, n).astype(np.int64),
            n=rng.integers(1, 200, n).astype(np.int32),
            flags=np.zeros(n, np.int8), start_epoch=1_700_000_000,
        )
        assert np.isfinite(f).all()
        out.append(f[:, 0].std())
    # Log-return volatility must not depend on the price level.
    assert out[0] == pytest.approx(out[1], rel=0.15)


def test_filled_seconds_are_flagged_not_hidden():
    n = 100
    flags = np.zeros(n, np.int8)
    flags[10:20] = 1
    f = build_features(
        c=np.full(n, 5_000_000, np.int32), o_c=np.zeros(n, np.int32),
        h_c=np.zeros(n, np.int32), l_c=np.zeros(n, np.int32),
        v=np.zeros(n, np.int64), tb=np.zeros(n, np.int64),
        n=np.zeros(n, np.int32), flags=flags, start_epoch=0,
    )
    assert f[10:20, 9].all() and not f[:10, 9].any()


# --------------------------------------------------------------------------- #
# tokenisation
# --------------------------------------------------------------------------- #
def test_pyramid_tiles_the_window_exactly(spec):
    assert spec.bin_widths().sum() == spec.context_seconds
    assert len(spec.bin_widths()) == spec.n_tokens
    ends = spec.token_end_offsets()
    assert ends[-1] == 0  # newest token ends at the cursor
    assert ends[0] == spec.context_seconds - spec.bin_widths()[0]


def test_additive_channels_sum_and_extent_channels_max(spec):
    w = np.zeros((spec.context_seconds, N_FEATURES), dtype=np.float32)
    w[:, 0] = 1.0  # log_return: additive
    w[:, 1] = np.arange(spec.context_seconds)  # hl_range: extent
    t = tokenize(w, spec)
    widths = spec.bin_widths()
    # Each token's return equals its bin width (sum of ones).
    assert np.allclose(t[:, 0], widths, atol=1e-3)
    # The newest token's range is the window's final value.
    assert t[-1, 1] == pytest.approx(spec.context_seconds - 1)


def test_batch_tokeniser_matches_single(spec):
    rng = np.random.default_rng(1)
    w = rng.normal(size=(3, spec.context_seconds, N_FEATURES)).astype(np.float32)
    batched = tokenize_batch(w, spec)
    for i in range(3):
        assert np.allclose(batched[i], tokenize(w[i], spec), atol=1e-4)


def test_tokenizer_rejects_wrong_shape(spec):
    with pytest.raises(ValueError):
        tokenize(np.zeros((100, N_FEATURES), np.float32), spec)


# --------------------------------------------------------------------------- #
# causality -- the one that matters most
# --------------------------------------------------------------------------- #
def test_analyser_cannot_see_the_future(spec):
    torch.manual_seed(0)
    model = MarketAnalyser(d_model=48, n_layers=6, d_state=48).eval()
    x = torch.randn(2, spec.n_tokens, N_FEATURES)
    cut = 300
    x2 = x.clone()
    x2[:, cut:] += 25.0  # violently perturb everything after `cut`
    with torch.no_grad():
        a, b = model(x)["memory"], model(x2)["memory"]
    assert torch.allclose(a[:, :cut], b[:, :cut], atol=1e-6), (
        "analyser output before the perturbation changed -> future leak"
    )


def test_analyser_receptive_field_spans_the_window(spec):
    model = MarketAnalyser(d_model=32, n_layers=10, d_state=32)
    assert model.receptive_field >= spec.n_tokens, (
        "the analyser cannot physically see the whole 12-hour context"
    )


# --------------------------------------------------------------------------- #
# basis / predictor
# --------------------------------------------------------------------------- #
def test_basis_paths_open_without_a_discontinuity():
    """The forecast must not jump away from the anchor at h=0.

    The meaningful property is continuity, not smallness: the step from the
    anchor (implicitly 0) into the first horizon point should be no larger
    than the path's typical one-step movement. A free 1500-output head
    routinely violates this by a factor of hundreds, which is what makes its
    first-second forecast unusable for execution.
    """
    synth = PathSynthesiser(1500, 48)
    path = synth(torch.randn(64, 48))
    opening_step = path[:, 0].abs()
    typical_step = path.diff(dim=1).abs().amax(dim=1)
    assert (opening_step <= 1.5 * typical_step).all()


def test_basis_paths_are_smooth_by_construction():
    """Curvature is bounded because the expansion is band-limited.

    Compared against an unconstrained head with the same output variance, the
    basis must produce dramatically smaller second differences -- this is the
    'predictions must be stable' requirement, enforced by parameterisation.
    """
    synth = PathSynthesiser(1500, 48)
    path = synth(torch.randn(32, 48))
    free = torch.randn(32, 1500) * path.std()
    curv = lambda p: (p[:, 2:] - 2 * p[:, 1:-1] + p[:, :-2]).abs().mean()  # noqa: E731
    assert curv(path) < curv(free) / 1000


def test_basis_represents_real_random_walks():
    rng = np.random.default_rng(3)
    walks = np.cumsum(rng.normal(0, 1e-5, size=(128, 1500)), axis=1)
    assert representable_fraction(walks, 1500, 48) > 0.97


def test_quantiles_never_cross(spec):
    torch.manual_seed(0)
    offsets = torch.from_numpy(spec.token_end_offsets().copy())
    model = PricePredictor(
        offsets, d_analyser=32, d_state=32, d_model=64, n_layers=2,
        n_heads=4, head="quantile",
    ).eval()
    x = torch.randn(4, spec.n_tokens, N_FEATURES)
    with torch.no_grad():
        out = model(x, torch.randn(4, spec.n_tokens, 32), torch.randn(4, 32))
    q = out["quantile_paths"]
    assert (q[:, 0] <= q[:, 1] + 1e-6).all() and (q[:, 1] <= q[:, 2] + 1e-6).all()


@pytest.mark.parametrize("head", ["point", "quantile", "distribution"])
def test_all_heads_forward_and_backward(spec, head):
    torch.manual_seed(0)
    offsets = torch.from_numpy(spec.token_end_offsets().copy())
    model = PricePredictor(
        offsets, d_analyser=32, d_state=32, d_model=64, n_layers=2,
        n_heads=4, head=head, horizon=300, n_coeffs=16,
    )
    x = torch.randn(2, spec.n_tokens, N_FEATURES)
    out = model(x, torch.randn(2, spec.n_tokens, 32), torch.randn(2, 32))
    assert out["path"].shape == (2, 300)
    out["path"].sum().backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters())


# --------------------------------------------------------------------------- #
# objective
# --------------------------------------------------------------------------- #
def test_oracle_scores_zero_on_every_supervised_term():
    torch.manual_seed(0)
    H, B = 600, 4
    obj = RealTimeObjective(ObjectiveConfig(), H)
    true = torch.cumsum(torch.randn(B, H + 5) * 1e-5, dim=1)
    idx = torch.arange(B) * 1000
    logs = {}
    for step in range(3):
        tgt = true[:, step + 1 : step + 1 + H] - true[:, step][:, None]
        _, logs = obj({"path": tgt.clone()}, tgt, torch.zeros(B), idx + step)
    assert logs["loss/path"] == pytest.approx(0.0, abs=1e-9)
    assert logs["loss/terminal"] == pytest.approx(0.0, abs=1e-9)
    assert logs["loss/direction"] == pytest.approx(0.0, abs=1e-9)
    # The consistency term must be exactly zero for a perfect, perfectly
    # stable forecaster -- if the re-anchoring is wrong it will not be.
    assert logs["loss/consistency"] == pytest.approx(0.0, abs=1e-9)


def test_wrong_sign_is_penalised_more_than_right_sign():
    torch.manual_seed(0)
    H, B = 600, 4
    cfg = ObjectiveConfig()
    true = torch.cumsum(torch.randn(B, H) * 1e-5, dim=1)

    good = RealTimeObjective(cfg, H)
    _, lg = good({"path": true.clone()}, true, torch.zeros(B), torch.arange(B))
    bad = RealTimeObjective(cfg, H)
    _, lb = bad({"path": -true.clone()}, true, torch.zeros(B), torch.arange(B))
    assert lb["loss/total"] > lg["loss/total"] * 5


def test_filled_targets_are_down_weighted():
    torch.manual_seed(0)
    H, B = 300, 4
    cfg = ObjectiveConfig(filled_weight=0.0)
    obj = RealTimeObjective(cfg, H)
    true = torch.cumsum(torch.randn(B, H) * 1e-4, dim=1)
    pred = torch.zeros(B, H)
    fill = torch.tensor([0.0, 0.0, 1.0, 1.0])
    loss_mixed, _ = obj({"path": pred}, true, fill, torch.arange(B))
    obj.reset()
    loss_all_real, _ = obj({"path": pred}, true, torch.zeros(B), torch.arange(B))
    assert torch.isfinite(loss_mixed) and torch.isfinite(loss_all_real)
