from inference.calculate_realized_utility import realized_utility


def test_hand_calculated_utility():
    assert realized_utility(4, 4, 1) == 0.25
    assert realized_utility(4, 1, 1) == 0
