import sys
sys.path.append('/home/dsi/rotemnizhar/dev/python_scripts')
from noisy_training.file_handler import get_data_dir
from medmnist import INFO
import medmnist
import torchvision.transforms as transforms
import PIL
import torch
import torch.utils.data as data
import glob
import numpy as np

BATCH_SIZE = 32
PIXELS = 224


dataset_to_n_classes_map = {
    'octmnist': 4,
    'ham10000': 7,
    'path_mnist': 9,
    'pathmnist': 9,
    'tissuemnist': 8,
    'bloodmnist': 8,
    'organamnist': 11,
    'organcmnist': 11,
    'organsmnist': 11,
    'cifar10': 10,
    'cifar100': 100,
    'tinyimagenet': 200,
    'imagenet': 1000,
    'dermamnist': 7,
    'chestmnist': 14
}

class Data:
    def __init__(self, train_images, train_labels, valid_images, valid_labels, test_images, test_labels,
                 train_dataset, valid_dataset, test_dataset):
        self.train_images = train_images
        self.train_labels = train_labels
        self.valid_images = valid_images
        self.valid_labels = valid_labels
        self.test_images = test_images
        self.test_labels = test_labels
        self.train_dataset = train_dataset
        self.valid_dataset = valid_dataset
        self.test_dataset = test_dataset
        
class TrueLabels:
    def __init__ (self, train_labels, valid_labels, test_labels):
        self.train_labels = train_labels
        self.valid_labels = valid_labels
        self.test_labels = test_labels
        
        
class TransitionMatrix:
    def __init__ (self, train, valid, test):
        self.train_matrix = train
        self.valid_matrix = valid
        self.test_matrix = test
        

def dataset_to_tensor(dataset):
    batch_size = 32
    dataloader = data.DataLoader(dataset, batch_size=batch_size, num_workers=4, pin_memory=True)
    images_list = []
    labels_list = []

    for images, labels in dataloader:
        if 'ChestMNIST' ==  dataset.info['python_class']:  #chestmnist is one hot encoding
            labels = labels.argmax(dim=1)
        images = images.view(-1, 3, PIXELS, PIXELS)
        labels = labels.view(-1)
        images_list.append(images)
        labels_list.append(labels)

    # Concatenate all batches into single tensors
    images_tensor = torch.cat(images_list, dim=0)
    labels_tensor = torch.cat(labels_list, dim=0)
    return images_tensor, labels_tensor


def load_data(dataset_name):

    
    download = True
    info = INFO[dataset_name]
    
    DataClass = getattr(medmnist, info['python_class'])

    data_transform = transforms.Compose(
        [transforms.Resize((PIXELS, PIXELS), interpolation=PIL.Image.NEAREST), 
        transforms.ToTensor(),
        transforms.Normalize(mean=[.5], std=[.5])])

    download = True
    as_rgb = True

    info = INFO[dataset_name]
    DataClass = getattr(medmnist, info['python_class'])
    
    print ("----- get/download data... -----")
    train_dataset = DataClass(split='train', transform=data_transform, download=download, as_rgb=as_rgb)
    val_dataset = DataClass(split='val', transform=data_transform, download=download, as_rgb=as_rgb)
    test_dataset = DataClass(split='test', transform=data_transform, download=download, as_rgb=as_rgb)

    print ("----- Dataset to TensorDataset -----")
    # train_images = torch.cat([item[0] for item in train_dataset], axis = 0).reshape(-1, 3, PIXELS, PIXELS)
    # train_labels = torch.tensor([item[1].item() for item in train_dataset])

    # valid_images = torch.cat([item[0] for item in val_dataset], axis = 0).reshape(-1, 3, PIXELS, PIXELS)
    # valid_labels = torch.tensor([item[1].item() for item in val_dataset])

    # test_images = torch.cat([item[0] for item in test_dataset], axis = 0).reshape(-1, 3, PIXELS, PIXELS)
    # test_labels = torch.tensor([item[1].item() for item in test_dataset])
    
    
    train_images, train_labels = dataset_to_tensor(train_dataset)
    valid_images, valid_labels = dataset_to_tensor(val_dataset)
    test_images, test_labels = dataset_to_tensor(test_dataset)
    return Data(train_images = train_images, train_labels = train_labels, 
                valid_images = valid_images, valid_labels = valid_labels,
                test_images = test_images, test_labels = test_labels,
                train_dataset= train_dataset, valid_dataset= val_dataset, test_dataset= test_dataset)
    
def create_transition_matrix(n_classes, labels, noisy_labels):
    """
    Create a normalized transition matrix from noisy labels and true labels
    :param n_classes: number of classes
    :param labels: true labels
    :param noisy_labels: noisy labels
    :return: normalized confusion matrix
    """
    cm = np.zeros((n_classes, n_classes))

    labels_np = labels.numpy()
    noisy_labels_np = noisy_labels.numpy()
    # Populate the confusion matrix
    for i in range(labels_np.shape[0]):
        cm[int(labels_np[i])][int(noisy_labels_np[i])] += 1

    # Normalize each row
    row_sums = cm.sum(axis=1, keepdims=True)
    cm_normalized = cm / row_sums

    # Handle cases where row_sums might be zero to avoid division by zero
    cm_normalized = np.nan_to_num(cm_normalized, nan=0.0)

    return cm_normalized

def load_noisy_labels(dataset_name, acc, set_type):
    dir_path = get_data_dir(dataset=dataset_name)
    file_pattern = f"noisy_labels_{set_type}_dataset_{dataset_name}_want_acc_{acc}*.npy" 
    full_pattern = f"{dir_path}/{file_pattern}"
    files = glob.glob(full_pattern)
    if len(files) != 1:
        raise Exception(f"Error reading file with pattern: {full_pattern}")
    noisy_labels = np.load(files[0])
    return torch.from_numpy(noisy_labels)
    

def get_loaders_for_training(dataset_name):
    cur_data = load_data(dataset_name=dataset_name)
    train_tensor_dataset = data.TensorDataset(cur_data.train_images, cur_data.train_labels)
    train_tensor_at_eval = data.TensorDataset(cur_data.train_images, cur_data.train_labels)
    valid_tensor_dataset = data.TensorDataset(cur_data.valid_images, cur_data.valid_labels)
    test_tensor_dataset = data.TensorDataset(cur_data.test_images, cur_data.test_labels)
    
    train_loader = data.DataLoader(dataset=train_tensor_dataset,
                                batch_size=BATCH_SIZE,
                                shuffle=True, num_workers=2)
    train_loader_at_eval = data.DataLoader(dataset=train_tensor_at_eval,
                                batch_size=BATCH_SIZE,
                                shuffle=False, num_workers=2)
    
    val_loader = data.DataLoader(dataset=valid_tensor_dataset,
                                batch_size=BATCH_SIZE,
                                shuffle=False, num_workers=2)

    test_loader = data.DataLoader(dataset=test_tensor_dataset, batch_size=2*BATCH_SIZE, shuffle=False, num_workers=2)
    
    return train_loader, train_loader_at_eval, val_loader, test_loader
    

def get_loaders_noisy_training(dataset_name, acc):
    cur_data = load_data(dataset_name=dataset_name)
    noisy_labels_training = load_noisy_labels(dataset_name=dataset_name, acc=acc, set_type='train')
    noisy_labels_training_eval = load_noisy_labels(dataset_name=dataset_name, acc=acc, set_type='train')
    noisy_labels_valid = load_noisy_labels(dataset_name=dataset_name, acc=acc, set_type='valid')
    noisy_labels_test = load_noisy_labels(dataset_name=dataset_name, acc=acc, set_type='test')
    
    train_tensor_dataset = data.TensorDataset(cur_data.train_images, noisy_labels_training)
    train_tensor_at_eval = data.TensorDataset(cur_data.train_images, noisy_labels_training_eval)
    valid_tensor_dataset = data.TensorDataset(cur_data.valid_images, noisy_labels_valid)
    test_tensor_dataset = data.TensorDataset(cur_data.test_images, noisy_labels_test)
    
    train_loader = data.DataLoader(dataset=train_tensor_dataset,
                                batch_size=BATCH_SIZE,
                                shuffle=True, num_workers=2)
    train_loader_at_eval = data.DataLoader(dataset=train_tensor_at_eval,
                                batch_size=BATCH_SIZE,
                                shuffle=False, num_workers=2)
    
    val_loader = data.DataLoader(dataset=valid_tensor_dataset,
                                batch_size=BATCH_SIZE,
                                shuffle=False, num_workers=2)
    

    test_loader = data.DataLoader(dataset=test_tensor_dataset, batch_size=2*BATCH_SIZE, shuffle=False, num_workers=2)
    true_labels = TrueLabels(train_labels=cur_data.train_labels, valid_labels=cur_data.valid_labels, test_labels=cur_data.test_labels)
    
    n_classes = len(np.unique(cur_data.train_labels.numpy()))
    transition_matrix = TransitionMatrix(train=create_transition_matrix(n_classes=n_classes, labels = cur_data.train_labels, noisy_labels=noisy_labels_training_eval)
                                         , valid= create_transition_matrix(n_classes=n_classes, labels = cur_data.valid_labels, noisy_labels=noisy_labels_valid),
                                         test=create_transition_matrix(n_classes=n_classes, labels = cur_data.test_labels, noisy_labels=noisy_labels_test))
    
    return train_loader, train_loader_at_eval, val_loader, test_loader, true_labels, transition_matrix




def get_num_classes(dataset):
    return dataset_to_n_classes_map[dataset]