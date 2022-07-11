import matplotlib.pyplot as plt

from scripts.trainer import Trainer
from scripts.import_dataset import load_data


def test_train():
    trainer = Trainer()
    train_loader, test_loader = load_data(train_batch_size=16,
                                          test_batch_size=1,
                                          ratio_data=1000)
    trainer.train(n_epochs=10,
                  train_dataloader=train_loader,
                  )
    plt.plot(trainer.history["loss"])
    plt.show()


if __name__ == "__main__":
    test_train()
