from sklearn.datasets import load_boston


def load_dataset():
    dataset = load_boston()
    return dataset.data, dataset.target
