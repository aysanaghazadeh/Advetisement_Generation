from .utils.data.trian_test_split import get_train_data
from .utils.data.dataset import *
from torch.utils.data import DataLoader


def get_train_Mistral7B_Dataloader(args):
    image_urls = get_train_data(args).values
    dataset = Mistral7BTrainingDataset(args, image_urls)
    train_loader = DataLoader(dataset, batch_size=4, shuffle=False)
    return train_loader
