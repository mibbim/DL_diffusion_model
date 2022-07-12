from torch.utils.data import Dataset
import imageio
import os
import torchvision.transforms as T

class CSSDdataset(Dataset):
    def __init__(self, main_dir, transform=None, test = False, ratio_test=0.20):
        self.main_dir = main_dir
        self.transform = transform
        # self.images = os.listdir(image_dir)

        self.all_imgs = os.listdir(main_dir)
        if test:
            self.all_imgs = self.all_imgs[:int(len(self.all_imgs) * ratio_test)]
        else:
            self.all_imgs = self.all_imgs[int(len(self.all_imgs) * ratio_test):]

    def __len__(self):
        return len(self.all_imgs)

    def __getitem__(self, index):

        img_loc = os.path.join(self.main_dir, self.all_imgs[index])
        im = imageio.imread(img_loc)
        im = T.ToPILImage()(im)

        # minimum pixel of this dataset is x = 139 and y = 238
        im = im.resize((139, 139)) #without keeping the ratio

        if self.transform:
            im = self.transform(im)

        return im