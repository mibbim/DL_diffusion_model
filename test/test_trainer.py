from datetime import datetime
from pathlib import Path

from scripts.DiffusionModel import DiffusionModel
from scripts.Unet_valeria import UNet
from scripts.trainer import Trainer
from scripts.import_dataset import load_MNIST, load_data_CSSD

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
    out_path = script_dir / "out" / "{now:%Y-%m-%d}_{now:%H:%M:%S}".format(now=datetime.now())
    trainer = Trainer(model=DiffusionModel(noise_predictor=UNet(3, 3)),
                      out_path=out_path,
                      )
    train_loader, test_loader = load_data_CSSD(train_batch_size=32,
                                               test_batch_size=64,
                                               ratio_test=0.2)
    trainer.train(n_epochs=50,
                  train_dataloader=train_loader,
                  val_dataloader=test_loader,
                  valid_each=5,
                  )

    # for m in trainer.history.keys():
    #     plot_single(trainer.history, m)
    # plt.legend()
    # plt.show()


if __name__ == "__main__":
    # test_train()
    test_train_Unet_valeria()
