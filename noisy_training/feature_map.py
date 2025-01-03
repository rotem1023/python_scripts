import sys
sys.path.append('/home/dsi/rotemnizhar/dev/python_scripts')
import torch
from timm import create_model
from medmnist import PathMNIST
import numpy as np
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
import torch
from torch import nn
import numpy as np
import timm
from torch.utils.data import DataLoader, TensorDataset
from noisy_training.loader import get_loaders_for_training
import argparse
import gc
import os
from noisy_training.file_handler import get_data_dir

class SwinBase(nn.Module):
    def __init__(self, in_22k_1k=False, in_timm_1k=False):
        super(SwinBase, self).__init__()
        self.in_22k_1k = in_22k_1k
        self.in_timm_1k = in_timm_1k

        if in_22k_1k:
            # Load swin_base_patch4_window7_224 pre-trained on 22k dataset
            self.model_swin = timm.create_model('swin_base_patch4_window7_224_in22k', pretrained=True)
        elif in_timm_1k:
            # Load swin_base_patch4_window7_224 pre-trained on ImageNet-1k
            self.model_swin = timm.create_model('swin_base_patch4_window7_224', pretrained=True)
        else:
            # Use timm's built-in Swin transformer and replace the head with Identity
            self.model_swin = timm.create_model('swin_base_patch4_window7_224', pretrained=True)
            self.model_swin.head = torch.nn.Identity()  # Remove final classification layer

    def forward(self, x):
        # Forward pass depending on the model
        if self.in_22k_1k or self.in_timm_1k:
            x = self.model_swin.forward_features(x)  # Use forward_features for models in timm
        else:
            x = self.model_swin(x)  # Use regular forward for the custom model
        x = x.view(x.size(0), -1)  # Flatten the output
        return x


def extract_features(dataloader, model, device):
    """
    Extract features from a dataloader using a given model.

    Args:
        dataloader: Iterable data loader with batches of data.
        model: PyTorch model to extract features.
        device: Device (e.g., 'cuda' or 'cpu') to run computations on.

    Returns:
        features: Numpy array of extracted features.
        labels: Numpy array of corresponding labels.
    """
    features = []
    labels = []

    for i, batch in enumerate(dataloader, start=1):
        # Move batch data to the target device
        batch_images, batch_labels = batch[0].to(device), batch[1]
        labels.extend(batch_labels.tolist())

        # Forward pass to extract features
        with torch.no_grad():
            batch_features = model(batch_images).cpu().numpy()
            features.append(batch_features)

        # Clean up unused tensors
        del batch_images, batch_labels, batch_features
        torch.cuda.empty_cache()
        gc.collect()

        print(f"Batch {i} completed")

    # Concatenate features into a single array
    features = np.vstack(features)  # Stacking is more efficient than concatenation
    return features, np.array(labels)




if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Create feature map ')

    parser.add_argument('--dataset',
                        default='organsmnist',
                        type=str)
    parser.add_argument('--output_root',
                        default='./output',
                        help='output root, where to save models and results',
                        type=str)
    parser.add_argument('--swin22',
                        default=True,
                        type=bool)


    args = parser.parse_args()
    dataset_name = args.dataset
    swin22 = args.swin22
    
    
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    if swin22:
        model = SwinBase(in_22k_1k=True)
        model_type = '22k'
    else:
        model = SwinBase(in_timm_1k=True)
        model_type = '1k'
    model = model.to(device)
    model.eval()
    
    train_loader,train_loader_at_eval, val_loader, test_loader = get_loaders_for_training(dataset_name=dataset_name)
    
    valid_fatures, valid_labels = extract_features(val_loader, model, device)
    test_fetaures, test_labels = extract_features(test_loader, model, device)
        
    
    outputdir = get_data_dir(dataset=dataset_name)
    os.makedirs(outputdir, exist_ok=True) 
    np.save(f"{outputdir}/valid_features_map_{model_type}.npy", valid_fatures)
    np.save(f"{outputdir}/test_features_map_{model_type}.npy", test_fetaures)
    
    # np.save(f"{outputdir}/valid_labels_check.npy", valid_labels)
    # np.save(f"{outputdir}/test_labels_check.npy", test_labels)
    
    print("save features")