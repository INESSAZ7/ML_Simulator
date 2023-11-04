from typing import List


def profit(revenue: List[float], costs: List[float]) -> float:
    return sum(revenue) - sum(costs)


def margin(revenue: List[float], costs: List[float]) -> float:
    return (sum(revenue) - sum(costs)) / sum(revenue)


def markup(revenue: List[float], costs: List[float]) -> float:
    return (sum(revenue) - sum(costs)) / sum(costs)



def test_profit() -> None:
    assert profit([1, 2, 3], [1, 1, 1]) == 3
    assert profit([0, 0], [0, 0]) == 0
    #assert profit([2.2, 0], [1, 1]) == 0.2
    assert profit([2, 0], [1, 3]) == -2
    assert profit([], []) == 0


def test_margin() -> None:
    assert margin([1, 2, 3], [1, 1, 1]) == 0.5
    assert margin([2, 0], [1, 1]) == 0
    assert margin([2, 0], [1, 3]) == -1


def test_markup() -> None:
    assert markup([1, 2, 3], [1, 1, 1]) == 1
    assert markup([2, 0], [1, 1]) == 0
    assert markup([2, 0], [1, 3]) == -0.5

