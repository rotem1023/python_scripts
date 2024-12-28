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
    features = np.array([])
    i = 0
    labels = []
    for batch in dataloader:
        i+=1
        # Extract the tensor from the batch (batch is a tuple)
        batch_images = batch[0].to(device)
        labels.extend(batch[1].tolist())

        # Apply resizing to all images in the batch
        x_test = torch.stack([img for img in batch_images])
        x_test = x_test.float().to(device)
        cur_features = model.forward(x_test).detach().cpu().numpy()
        if i ==1:
            features = cur_features
        else:
            features = np.concatenate((features, cur_features), axis=0)
        del cur_features
        del batch_images
        del x_test
        torch.cuda.empty_cache()
        gc.collect()
        print(f"batch {i} completed")
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
        
    current_dir = os.path.dirname(os.path.abspath(__file__))
    outputdir = f'./{dataset_name}/data'
    os.makedirs(outputdir, exist_ok=True) 
    np.save(f"{outputdir}/valid_features_map_{model_type}.npy", valid_fatures)
    np.save(f"{outputdir}/test_features_map_{model_type}.npy", test_fetaures)
    
    np.save(f"{outputdir}/valid_labels_check.npy", valid_labels)
    np.save(f"{outputdir}/test_labels_check.npy", test_labels)
    
    print("save features")