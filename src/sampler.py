import torch
import torch.utils.data
import torchvision
import json


class ImbalancedDatasetSampler(torch.utils.data.sampler.Sampler):
    """Samples elements randomly from a given list of indices for imbalanced dataset
    Arguments:
        dataset : (list) : a list of (input, target)
        indices (list, optional): a list of indices
        num_samples (int, optional): number of samples to draw
    """

    def __init__(self, dataset, target_type, indices=None, num_samples=None):

        # if indices is not provided,
        # all elements in the dataset will be considered
        self.indices = list(range(len(dataset))) \
            if indices is None else indices

        # if num_samples is not provided,
        # draw `len(indices)` samples in each iteration
        self.num_samples = len(self.indices) \
            if num_samples is None else num_samples

        # load mean and stats
        self.stats = {}

        # f = open('data/RNN_dataset/mean_value.json', encoding='UTF-8')
        # f_ = open('tensor_data/RNN_dataset/mean_value.json', encoding='UTF-8')
        # if target_type == 'sbp':
        #     self.stats['mean'] = json.loads(f.read())['VS_sbp']
        #     self.stats['std'] = json.loads(f_.read())['VS_sbp']
        # elif target_type == 'dbp':
        #     self.stats['mean'] = json.loads(f.read())['VS_dbp']
        #     self.stats['std'] = json.loads(f_.read())['VS_dbp']
        #
        # f.close(); f_.close()

        # distribution of classes in the dataset
        label_to_count = {}
        for idx in self.indices:
            label = self._get_label(dataset, target_type, idx)
            if label in label_to_count:
                label_to_count[label] += 1
            else:
                label_to_count[label] = 1

        # weight for each sample
        weights = [1.0 / label_to_count[self._get_label(dataset, target_type, idx)]
                   for idx in self.indices]
        self.weights = torch.DoubleTensor(weights)

    def _get_label(self, dataset, target_type, idx):
        if target_type == 'regression':
            class_ = dataset[idx] * self.stats['std'] + self.stats['mean']
            if target_type == 'sbp':
                if class_ < 120:
                    return 'hypo'
                else:
                    return 'normal'

            if target_type == 'dbp':
                if class_ < 80:
                    return 'hypo'
                else:
                    return 'normal'

        else:
            return dataset[idx][0]

    #         if class_ < 112:
    #             return 'hypo'
    #         elif 112 < class_ < 131:
    #             return 'normal'
    #         elif 131 < class_ < 148:
    #             return 'pre_hyper'
    #         else:
    #             return 'hyper'
    """
        if dataset_type is torchvision.datasets.MNIST:
            return dataset.train_labels[idx].item()
        elif dataset_type is torchvision.datasets.ImageFolder:
            return dataset.imgs[idx][1]
        else:
            raise NotImplementedError
        """

    def __iter__(self):
        return (self.indices[i] for i in torch.multinomial(
            self.weights, self.num_samples, replacement=True))

    def __len__(self):
        return self.num_samples
