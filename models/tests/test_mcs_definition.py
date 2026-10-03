import numpy as np

from models.mcs_definition import model_confidence_set, qlike_losses


def test_mcs_definition():
    actual = np.array([2.0, 1.0])
    predicted = np.array([1.0, 2.0])
    np.testing.assert_allclose(qlike_losses(actual, predicted),
                               [1 - np.log(2), 0.5 + np.log(2) - 1])

    # Shared, non-circular block draws: reproduce the bootstrap means by hand.
    losses = np.column_stack((np.arange(7.0), np.arange(7.0)))
    result = model_confidence_set(losses, ["B", "A"], block_length=3,
                                  repetitions=19, seed=42)
    assert result == model_confidence_set(losses, ["B", "A"], block_length=3,
                                          repetitions=19, seed=42)
    assert result["members"] == ("A", "B")  # identical centered histories
    assert result["trace"][0]["t_max"] == 0
    assert result["trace"][0]["p_value"] == 1

    try:
        model_confidence_set(np.column_stack((np.zeros(20), np.ones(20))),
                             ["A", "B"], repetitions=19)
    except ValueError:
        pass
    else:
        raise AssertionError("degenerate nonzero excess accepted")

    rng = np.random.default_rng(42)
    starts = rng.integers(0, 5, size=(19, 3))
    indices = np.concatenate([np.arange(s, s + 3) for s in starts[0]])[:7]
    assert len(indices) == 7 and indices.max() < 7
    assert starts.min() >= 0 and starts.max() <= 4

    # Independent small-sample calculation of centered bootstrap T_max and p.
    pair = np.array([[0., 1.], [1., 0.], [0., 2.], [2., 0.],
                     [0., 3.], [3., 0.]])
    observed = pair - pair.mean(axis=1, keepdims=True)
    observed = observed.mean(axis=0)
    draws = np.random.default_rng(3).integers(0, 5, size=(29, 3))
    sampled = np.array([pair[np.concatenate([np.arange(s, s + 2)
                                              for s in row])] for row in draws])
    sampled_excess = sampled - sampled.mean(axis=2, keepdims=True)
    sampled_excess = sampled_excess.mean(axis=1)
    se = sampled_excess.std(axis=0, ddof=1)
    expected_t = (observed / se).max()
    bootstrap_t = ((sampled_excess - observed) / se).max(axis=1)
    expected_p = (1 + np.count_nonzero(bootstrap_t >= expected_t)) / 30
    calculated = model_confidence_set(pair, ["A", "B"], block_length=2,
                                      repetitions=29, seed=3)["trace"][0]
    np.testing.assert_allclose(calculated["t_max"], expected_t)
    assert calculated["p_value"] == expected_p

    # A clear inferior candidate is eliminated; the retained singleton is not retested.
    rng = np.random.default_rng(7)
    losses = np.column_stack((rng.normal(0, 0.1, 120),
                              rng.normal(0, 0.1, 120) + 1))
    result = model_confidence_set(losses, ["good", "bad"], block_length=20,
                                  repetitions=199, seed=42)
    assert result["members"] == ("good",)
    assert len(result["trace"]) == 1
    assert result["trace"][0]["removed"] == "bad"
    assert result["adjusted_p_values"]["bad"] == result["trace"][0]["p_value"]
    assert result["adjusted_p_values"]["good"] == 1

    three = np.column_stack((rng.normal(0, 0.1, 120),
                             rng.normal(0.5, 0.1, 120),
                             rng.normal(1, 0.1, 120)))
    result = model_confidence_set(three, ["good", "middle", "worst"],
                                  repetitions=199, seed=42)
    assert result["members"] == ("good",)
    assert [step["removed"] for step in result["trace"]] == ["worst", "middle"]
    assert result["adjusted_p_values"]["middle"] == max(
        step["p_value"] for step in result["trace"])

    tied = np.column_stack((losses[:, 0], losses[:, 1], losses[:, 1]))
    result = model_confidence_set(tied, ["A", "C", "B"], repetitions=199)
    assert result["trace"][0]["removed"] == "B"

    for invalid in (np.array([[0.0, np.nan]]), np.ones((19, 2))):
        try:
            model_confidence_set(invalid, ["A", "B"])
        except ValueError:
            pass
        else:
            raise AssertionError("invalid MCS input accepted")
    for invalid in (0.0, np.inf):
        try:
            qlike_losses([1.0], [invalid])
        except ValueError:
            pass
        else:
            raise AssertionError("invalid QLIKE input accepted")


if __name__ == "__main__":
    test_mcs_definition()
