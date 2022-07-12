from scripts.performance_meter import AverageMeter


def test_avg_meter():
    am = AverageMeter(["mse", "vlb"])
    print(f"Instatiated with mse and vlb: \n", am.metrics)
    am.update({"mse": 10, "vlb": 3})
    print(f"Updated with 10, and 3: \n", am.metrics)
    am.update({"mse": 30, "vlb": 5})
    print(f"Updated with 30 and 5: \n", am.metrics)
    am.reset()
    print(f"Reset: \n", am.metrics)


if __name__ == "__main__":
    test_avg_meter()
