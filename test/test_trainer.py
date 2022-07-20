from pathlib import Path

from scripts.trainer import Trainer
from scripts.import_dataset import load_MNIST

script_dir = Path(__file__).resolve().parent


# def plot_single(H, m):
#     values = np.array(H[m]).T
#     plt.plot(values[0], values[1], label=m)


def test_train():
    trainer = Trainer(out_path=script_dir / "out")
    train_loader, test_loader = load_MNIST(train_batch_size=16,
                                           test_batch_size=16,
                                           ratio_data=1000)
    trainer.train(n_epochs=10,
                  train_dataloader=train_loader,
                  val_dataloader=test_loader,
                  valid_each=2,
                  )

    # for m in trainer.history.keys():
    #     plot_single(trainer.history, m)
    # plt.legend()
    # plt.show()

def test_train_Unet_valeria():
    trainer = Trainer(out_path=script_dir / "out")
    train_loader, test_loader = load_MNIST(train_batch_size=32,
                                           test_batch_size=16,
                                           ratio_data=1000)
    trainer.train(n_epochs=10,
                  train_dataloader=train_loader,
                  val_dataloader=test_loader,
                  valid_each=2,
                  )

    # for m in trainer.history.keys():
    #     plot_single(trainer.history, m)
    # plt.legend()
    # plt.show()

if __name__ == "__main__":
    test_train()
