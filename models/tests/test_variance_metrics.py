import math

from models.training_blocks import variance_metrics


def test_variance_metrics():
    history = {"A": [1, 2, 3], "B": [1, 3, 5]}
    scores = variance_metrics([2, 4, 4], [1, 2, 8], ["A", "B", "B"], history)
    expected = {
        "mae": 7 / 3,
        "mase": 4 / 3,
        "mse": 7,
        "rmse": math.sqrt(7),
        "qlike": ((2 - math.log(2) - 1) * 2 + (0.5 - math.log(0.5) - 1)) / 3,
    }
    assert all(math.isclose(scores[key], value) for key, value in expected.items())
    assert variance_metrics([2], [1], ["A"], history)["qlike"] > variance_metrics(
        [2], [4], ["A"], history
    )["qlike"]
    assert math.isclose(
        variance_metrics([1e-300], [1e100], ["A"], history)["qlike"],
        400 * math.log(10) - 1,
    )
    assert variance_metrics([1 + 1e-12], [1], ["A"], history)["qlike"] > 0

    for actual, prediction, tickers, histories in (
        ([], [], [], history),
        ([1], [1, 2], ["A"], history),
        ([0], [1], ["A"], history),
        ([1], [0], ["A"], history),
        ([math.inf], [1], ["A"], history),
        ([1], [math.nan], ["A"], history),
        ([1], [1], ["A"], {"A": [1, 1]}),
        ([1], [1], ["A"], {"A": [1]}),
        ([1], [1], ["A"], {"A": [1, math.inf]}),
        ([1], [1], ["A"], {}),
    ):
        try:
            variance_metrics(actual, prediction, tickers, histories)
        except ValueError:
            pass
        else:
            raise AssertionError("invalid variance input was accepted")


if __name__ == "__main__":
    test_variance_metrics()
    print("variance metric check passed")
