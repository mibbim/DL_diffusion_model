import matplotlib.pyplot as plt

from scripts.trainer import Trainer
from scripts.import_dataset import load_data
import numpy as np


def plot_single(H, m):
    values = np.array(H[m]).T
    plt.plot(values[0], values[1], label=m)


def test_train():
    trainer = Trainer()
    train_loader, test_loader = load_data(train_batch_size=16,
                                          test_batch_size=16,
                                          ratio_data=1000)
    trainer.train(n_epochs=10,
                  train_dataloader=train_loader,
                  val_dataloader=test_loader,
                  valid_each=2,
                  )

    for m in trainer.history.keys():
        plot_single(trainer.history, m)
    plt.legend()
    plt.show()


if __name__ == "__main__":
    test_train()
