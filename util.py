import torch
from torchvision import transforms, datasets

# for extra speed
DEVICE = torch.device("cuda")


def preload_data():
    """ preload data onyo GPU for extra speed """
    print("preloading data")
    _tranformers = [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    transformers = transforms.Compose(_tranformers)
    TRAIN_SET = datasets.MNIST("./_mnist", train=True, download=True, transform=transformers)
    TEST_SET = datasets.MNIST("./_mnist", train=False, download=True, transform=transformers)
    train_images, train_labels = zip(*TRAIN_SET)
    test_images, test_labels = zip(*TEST_SET)
    train_images = torch.stack(train_images, dim=0).to(DEVICE)
    train_labels = torch.tensor(train_labels, device=DEVICE)
    test_images = torch.stack(test_images, dim=0).to(DEVICE)
    test_labels = torch.tensor(test_labels, device=DEVICE)
    return train_images, train_labels, test_images, test_labels


(
    TRAIN_IMAGES,
    TRAIN_LABELS,
    TEST_IMAGES,
    TEST_LABELS
) = preload_data()


class PersonalDataLoader:
    """ dataloader that preloads all data onto GPU """

    def __init__(self, train=True, batch_size: int = 128):
        if train:
            labels = TRAIN_LABELS
            data = TRAIN_IMAGES
        else:
            labels = TEST_LABELS
            data = TEST_IMAGES
        self.labels = labels
        self.batch_size = batch_size
        self.data = data
        self.device = DEVICE
        self.length = torch.cuda.LongTensor([(len(data) // batch_size) + 1], device=DEVICE)
        self.length_orgi = len(data)
        self.indices = None
        self.counter = torch.zeros([1], device=self.device).long()
        self.one = torch.ones([1], device=self.device).long()

    def __len__(self):
        return self.length

    def __iter__(self):
        self.indices = (
                torch.randperm(self.length.item() * self.batch_size, device=self.device) % self.length_orgi).view(
            -1, self.batch_size)
        self.counter -= self.counter
        return self

    def __next__(self):
        if self.counter >= self.length:
            raise StopIteration
        else:
            batch = self.__getitem__(self.counter)
            self.counter += self.one
            return batch

    def __getitem__(self, item):
        return self.data[
                   self.indices[item].squeeze()
               ], \
               self.labels[
                   self.indices[item].squeeze()
               ]
