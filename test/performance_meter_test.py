from scripts.performance_meter import AverageMeter


def test_avg_meter():
    am = AverageMeter(["mse", "vlb"])
    am.update({"mse": 10, "vlb": 3})
    assert am.metrics == {'mse': {'val': 10, 'count': 1, 'sum': 10, 'avg': 10.0},
                          'vlb': {'val': 3, 'count': 1, 'sum': 3, 'avg': 3.0}}
    am.update({"mse": 30, "vlb": 5})
    assert am.metrics == {'mse': {'val': 30, 'count': 2, 'sum': 40, 'avg': 20.0},
                          'vlb': {'val': 5, 'count': 2, 'sum': 8, 'avg': 4.0}}
    am.reset()
    assert am.metrics == {'mse': {'val': 0, 'count': 0, 'sum': 0, 'avg': 0},
                          'vlb': {'val': 0, 'count': 0, 'sum': 0, 'avg': 0}}


if __name__ == "__main__":
    test_avg_meter()
