
from medmnist import PathMNIST
import pandas as pd
from tensorflow.keras.utils import to_categorical
import os


class InputData:
    
    def __init__(self, x_train, x_test, train_true_labels, test_true_labels, train_noisy_labels, test_noisy_labels):
        self.x_train = x_train
        self.x_test = x_test
        self.train_true_labels = train_true_labels
        self.test_true_labels= test_true_labels
        self.train_noisy_labels= train_noisy_labels
        self.test_noisy_labels = test_noisy_labels



def _get_cur_file_path():
    return f'{os.path.dirname(os.path.abspath(__file__))}'


def _get_resources_dir():
    cur_file_dir = _get_cur_file_path()
    return f"{cur_file_dir}/resources"


def get_path_mnist_noisy_labels():
    # load x
    train_dataset = PathMNIST(split="train", download=True)
    test_dataset = PathMNIST(split="test", download=True)
    x_train = train_dataset.imgs/255
    x_test = test_dataset.imgs/255
    
    train_true_labels= to_categorical(train_dataset.labels, num_classes = 9)
    test_true_labels = to_categorical(test_dataset.labels, num_classes= 9)
    
    
    # load noisy labels
    noisy_labels_train_path= f'{_get_resources_dir}/pathminst_train_noisy_labels.csv'
    noisy_labels_test_path = f'{_get_resources_dir}/pathmnist_test_noisy_labels.csv'
    noisy_labels_test_table=  pd.read_csv(noisy_labels_test_path)
    noisy_labels_train_table = pd.read_csv(noisy_labels_train_path)
    noisy_labels_test = noisy_labels_test_table['Predicted Label'].to_numpy()
    noisy_labels_train = noisy_labels_train_table['Predicted Label'].to_numpy()
    
    noisy_train_labels = to_categorical(noisy_labels_train,num_classes=9)
    noisy_test_labels= to_categorical(noisy_labels_test,num_classes=9)
    
    data = InputData(x_train=x_train, x_test=x_test, train_true_labels=train_true_labels, test_true_labels= test_true_labels,
                     noisy_labels_train= noisy_train_labels, noisy_test_labels = noisy_test_labels)
    
    return data    