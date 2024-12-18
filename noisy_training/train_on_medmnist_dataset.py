import os, sys, inspect
import torch
import torch.nn as nn
import torch.utils.data as data
import torchvision.transforms as transforms
import torchvision
import numpy as np
import pickle
from tqdm import tqdm

import medmnist
from medmnist import INFO
import PIL
import argparse
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import MultiStepLR
import time 
import noise

from torch import nn
from torchvision import models,transforms


device = 'cuda' if (torch.cuda.is_available()) else 'cpu'

def error(T, T_true):
    error = np.sum(np.abs(T-T_true)) / np.sum(np.abs(T_true))
    return error

def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False

def get_model(model_name, num_classes = 9, feature_extract = False, use_pretrained=True):
    model_ft = None

    if model_name == "resnet18":
        """ Resnet18, resnet34, resnet50, resnet101
        """
        model_ft = models.resnet18(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_classes)
        input_size = 128

    elif model_name == "resnet50":
        model_ft = models.resnet50(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_classes)
        input_size = 128
    
    elif model_name == "vgg":
        """ VGG11_bn
        """
        model_ft = models.vgg16_bn(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Linear(num_ftrs,num_classes)
        input_size = 128

    elif model_name == "densenet121":
        """ Densenet121
        """
        model_ft = models.densenet121(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier.in_features
        model_ft.classifier = nn.Linear(num_ftrs, num_classes)
        input_size = 128

    else: 
        print("Invalid model name, exiting...")
        exit()

    return model_ft, input_size

class sig_t(nn.Module):
    def __init__(self, device, num_classes, init=2):
        super(sig_t, self).__init__()

        self.register_parameter(name='w', param=nn.parameter.Parameter(-init*torch.ones(num_classes, num_classes)))

        self.w.to(device)

        co = torch.ones(num_classes, num_classes)
        ind = np.diag_indices(co.shape[0])
        co[ind[0], ind[1]] = torch.zeros(co.shape[0])
        self.co = co.to(device)
        self.identity = torch.eye(num_classes).to(device)


    def forward(self):
        sig = torch.sigmoid(self.w)
        T = self.identity.detach() + sig*self.co.detach()
        T = F.normalize(T, p=1, dim=1)

        return T

class sig_t_uniform(nn.Module):
    def __init__(self, device, num_classes, init=-2):
        super(sig_t_uniform, self).__init__()

        self.register_parameter(name='epsilon', param=nn.parameter.Parameter(torch.Tensor([init])))

        self.epsilon.to(device)
        self.num_classes = num_classes
        self.ones = torch.ones(self.num_classes, self.num_classes).to(device)
        self.diag_idx = torch.arange(num_classes)


    def forward(self):
        val = torch.sigmoid(self.epsilon)
        off_diag_value =  val / (self.num_classes - 1)
    
        matrix = self.ones * off_diag_value
        matrix[self.diag_idx, self.diag_idx] = (1 - val)

        return matrix


class EmbedResnet(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def __call__(self, x):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)

        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)

        x = self.model.avgpool(x)
        x = torch.flatten(x, 1)
        return x

    def fc(self, x):
        return self.model.fc(x)


def save_pickle(data, file_path):
    with open(file_path, 'wb') as file:
        pickle.dump(data, file)


val_loss_list = []
val_acc_list = []

def train_model(model,
                trans,
                train_data_loader,
                valid_data_loader,
                optimizer_es,
                optimizer_trans,
                scheduler1,
                scheduler2,
                n_epochs,
                loss_func_ce,
                batch_size,
                lr,
                epsilon,
                model_name,
                train_transition_matrix,
                lam):


    for epoch in range(n_epochs):

        print('epoch {}'.format(epoch + 1))

        epoch_time_start = time.time()

        model.train()
        trans.train()

        train_loss = 0.
        train_vol_loss =0.
        train_acc = 0.
        val_loss = 0.
        val_acc = 0.

        train_items = 0
        model = model.to(device)
        for batch_x, batch_y in tqdm(train_data_loader):
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            train_items += len(batch_x)
            optimizer_es.zero_grad()
            optimizer_trans.zero_grad()


            clean = F.softmax(model(batch_x), 1)
            t = trans()

            out = torch.mm(clean, t)

            vol_loss = t.slogdet().logabsdet
            ce_loss = loss_func_ce(out.log(), batch_y.long())
            loss = ce_loss + lam * vol_loss

            train_loss += loss.item()
            train_vol_loss += vol_loss.item()

            pred = torch.max(out, 1)[1]
            train_correct = (pred == batch_y).sum()
            train_acc += train_correct.item()


            loss.backward()
            optimizer_es.step()
            optimizer_trans.step()
        print('Train Loss: {:.6f}, Vol_loss: {:.6f}  Acc: {:.6f}'.format(train_loss / train_items, train_vol_loss / train_items, train_acc / train_items))

        # --------- Estimation error ---------- #
        est_T = t.detach().cpu().numpy()
        estimate_error = error(est_T, train_transition_matrix)

        matrix_path = f'./{dataset_name}/ckpts/trans_matrix_est/eps_{epsilon}/{model_name}_lam_{lam}_epochs_{n_epochs}_lr_{lr}_bs_{batch_size}_seed_{seed}_matrix_epoch_{epoch+1}.npy'
        np.save(matrix_path, est_T)

        print('Estimation Error: {:.2f}'.format(estimate_error))
        print(est_T)

        # ----------------------------------- #


        scheduler1.step()
        scheduler2.step()

        valid_items = 0
        with torch.no_grad():
            model.eval()
            trans.eval()
            for batch_x, batch_y in tqdm(valid_data_loader):
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)
                valid_items += len(batch_x)
                clean =  F.softmax(model(batch_x))
                t = trans()

                out = torch.mm(clean, t)
                loss = loss_func_ce(out.log(), batch_y.long())
                val_loss += loss.item()
                pred = torch.max(out, 1)[1]
                val_correct = (pred == batch_y).sum()
                val_acc += val_correct.item()

                
        print('Val Loss: {:.6f}, Acc: {:.6f}'.format(val_loss / valid_items, val_acc / valid_items))

        epoch_time_end = time.time()
        # Log 
        print({'epoch': epoch,
                   'epoch_time': epoch_time_end-epoch_time_start,
                   'test_estimation_error': estimate_error,
                   'valid_acc': val_acc / valid_items,
                   'valid_loss': val_loss / valid_items,
                   'train_acc': train_acc / train_items,
                   'train_loss': train_loss / train_items})

        if (epoch >= 5) and (epoch % 5 == 0):
            best_ckpt_epoch = epoch + 1
            best_model_path = f'./{dataset_name}/ckpts/trans_matrix_est/eps_{epsilon}/new_model_{model_name}_{n_epochs}_lr_{lr}_bs_{batch_size}_lam_{lam}_epoch_{epoch}_seed_{seed}.pth'
            torch.save(model, best_model_path)
        
        val_loss_list.append(val_loss / valid_items)
        val_acc_list.append(val_acc / valid_items)

    val_loss_array = np.array(val_loss_list)
    val_acc_array = np.array(val_acc_list)
    model_index = np.argmin(val_loss_array)
    model_index_acc = np.argmax(val_acc_array)

    matrix_path = f'./{dataset_name}/trans_matrix_est/eps_{epsilon}/{model_name}_lam_{lam}_epochs_{n_epochs}_lr_{lr}_bs_{batch_size}_seed_{seed}_' 'matrix_epoch_%d.npy' % (model_index+1)
    final_est_T = np.load(matrix_path)
    final_estimate_error = error(final_est_T, train_transition_matrix)

    matrix_path_acc = f'./{dataset_name}/trans_matrix_est/eps_{epsilon}/{model_name}_lam_{lam}_epochs_{n_epochs}_lr_{lr}_bs_{batch_size}_seed_{seed}_' + 'matrix_epoch_%d.npy' % (model_index_acc+1)
    final_est_T_acc = np.load(matrix_path_acc)
    final_estimate_error_acc = error(final_est_T_acc, train_transition_matrix)

    print("Final estimation error loss: %f" % final_estimate_error)
    print("Final estimation error loss acc: %f" % final_estimate_error_acc)
    print("Best epoch: %d" % model_index)
    print(final_est_T)

def prep_model_and_data(dataset_name = 'pathmnist'):
    BATCH_SIZE = 32
    PIXELS = 224

    
    download = True
    info = INFO[dataset_name]
    task = info['task']
    n_channels = info['n_channels']
    n_classes = len(info['label'])
    
    DataClass = getattr(medmnist, info['python_class'])

    data_transform = transforms.Compose(
        [transforms.Resize((PIXELS, PIXELS), interpolation=PIL.Image.NEAREST), 
        transforms.ToTensor(),
        transforms.Normalize(mean=[.5], std=[.5])])

    batch_size = BATCH_SIZE
    download = True
    as_rgb = True
    train_epsilon = 0.2
    valid_epsilon = 0

    info = INFO[dataset_name]
    DataClass = getattr(medmnist, info['python_class'])
    
    print ("----- get/download data... -----")
    train_dataset = DataClass(split='train', transform=data_transform, download=download, as_rgb=as_rgb)
    val_dataset = DataClass(split='val', transform=data_transform, download=download, as_rgb=as_rgb)
    test_dataset = DataClass(split='test', transform=data_transform, download=download, as_rgb=as_rgb)

    print ("----- Dataset to TensorDataset -----")
    train_images = torch.cat([item[0] for item in train_dataset], axis = 0).reshape(-1, 3, PIXELS, PIXELS)
    train_labels = torch.tensor([item[1].item() for item in train_dataset])

    valid_images = torch.cat([item[0] for item in val_dataset], axis = 0).reshape(-1, 3, PIXELS, PIXELS)
    valid_labels = torch.tensor([item[1].item() for item in val_dataset])

    test_images = torch.cat([item[0] for item in test_dataset], axis = 0).reshape(-1, 3, PIXELS, PIXELS)
    test_labels = torch.tensor([item[1].item() for item in test_dataset])


    print ("----- Noisifiy -----")
    train_transition_matrix = np.eye(num_classes)
    if train_epsilon > 0:
        train_labels__noisy, real_noise_rate, train_transition_matrix = noise.noisify_multiclass_symmetric_diag_and_uniform(y_train=np.array(train_labels),
                                                    noise=train_epsilon, random_state= seed, nb_classes=num_classes)
        print ('Real train noise:', real_noise_rate)

    valid_transition_matrix = np.eye(num_classes)
    if valid_epsilon > 0:
        valid_labels__noisy, real_noise_rate, valid_transition_matrix = noise.noisify_multiclass_symmetric_diag_and_uniform(y_train=np.array(valid_labels),
                                                    noise=valid_epsilon, random_state= seed, nb_classes=num_classes)
        print ('Real valid noise:', real_noise_rate)

    if train_epsilon > 0:
        train_tensor_dataset = data.TensorDataset(train_images, torch.from_numpy(train_labels__noisy))
    else:
        train_tensor_dataset = data.TensorDataset(train_images, train_labels)

    if valid_epsilon > 0:
        valid_tensor_dataset = data.TensorDataset(valid_images, torch.from_numpy(valid_labels__noisy))
    else:
        valid_tensor_dataset = data.TensorDataset(valid_images, valid_labels)

    test_tensor_dataset = data.TensorDataset(test_images, test_labels)


    train_loader = data.DataLoader(dataset=train_tensor_dataset,
                                batch_size=batch_size,
                                shuffle=True)
    val_loader = data.DataLoader(dataset=valid_tensor_dataset,
                                batch_size=batch_size,
                                shuffle=False)

    test_loader = data.DataLoader(dataset=test_dataset, batch_size=2*BATCH_SIZE, shuffle=False)

    model, _ = get_model(model_name='resnet18', num_classes=num_classes)
    
    return train_loader, val_loader, test_loader, train_transition_matrix, train_labels, model

def get_embeds_logits(model, loader, device):
    model = model.to(device)
    model.eval()
        
    all_embeds = []
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for batch in tqdm(loader):
            x = batch[0].to(device)
            embeds = model(x)
            preds = model.fc(embeds)

            embeds = embeds.detach().cpu().numpy()
            preds = preds.detach().cpu().numpy()

            all_embeds.append(embeds)
            all_preds.append(preds)
            all_labels.append(batch[1].numpy())

    all_embeds = np.concatenate(all_embeds, 0)
    all_preds = np.concatenate(all_preds, 0)
    all_labels = np.concatenate(all_labels, 0)
    return all_embeds, all_preds, all_labels


def calc(model, loader, device, fold='Test'):
    all_embeds, all_preds, all_labels = get_embeds_logits(model, loader, device)
    all_labels = np.squeeze(all_labels)
    print(f'Embeds shape : {all_embeds.shape}')
    print(f'Preds shape : {all_preds.shape}')
    print(f'Labels shape : {all_labels.shape}')

    counts = np.bincount(all_labels)

    print(fold)
    print('Labels count: mean: {:.3f} max: {:.3f} min: {:.3f}'.format(counts.mean(),
                                                                      counts.max(),
                                                                      counts.min()))

    print(f'Acc: {(all_preds.argmax(1) == all_labels).mean()}')
    return all_embeds, all_preds, all_labels


def train(model, train_dataloader, valid_dataloader):
    noise_type = 'symmetric' # 'asymmetric'
    weight_decay = 1e-4
    lam = 0
    iam_lr = 0
    trans = sig_t_uniform(device, num_classes)

    trans = trans.to(device)

    #optimizer and StepLR
    milestones = [30,60]
    optimizer_trans = optim.SGD(trans.parameters(), lr=iam_lr, weight_decay=0, momentum=0.9)
    scheduler2 = MultiStepLR(optimizer_trans, milestones=milestones, gamma=0.1)
    
    
    loss_func_ce = torch.nn.NLLLoss()
    # done with 
    for lr in [0.00001]:
        if opt == 'Adam':
            optimizer_es = optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=1e-4)
        elif opt == 'SGD':
            optimizer_es = optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay, momentum=0.9)
        scheduler1 = MultiStepLR(optimizer_es, milestones=milestones, gamma=0.1)


        print ('=== Start Training ===')
        best_model_path = train_model(model = model, 
                                trans = trans,
                                train_data_loader = train_dataloader, 
                                valid_data_loader = valid_dataloader,
                                optimizer_es = optimizer_es,
                                optimizer_trans = optimizer_trans,
                                scheduler1 = scheduler1,
                                scheduler2 = scheduler2,
                                n_epochs = n_epochs, 
                                loss_func_ce = loss_func_ce, 
                                batch_size = batch_size,
                                lr=lr,
                                lam=lam,
                                epsilon=epsilon,
                                model_name= model_name, 
                                train_transition_matrix = train_transition_matrix)

def save(model, test_loader, val_loader, out_dir):
    # on test
    all_embeds, all_preds, all_labels = calc(model, test_loader, device, fold='Test')
    print(f'Embeds shape : {all_embeds.shape}')
    print(f'Preds shape : {all_preds.shape}')
    print(f'Labels shape : {all_labels.shape}')
    save_pickle({'embeds': all_embeds,
                 'preds': all_preds,
                 'labels': all_labels}, os.path.join(out_dir, 'noisy_training__test.pickle'))

    # on val
    all_embeds, all_preds, all_labels = calc(model, val_loader, device, fold='Val')
    print(f'Embeds shape : {all_embeds.shape}')
    print(f'Preds shape : {all_preds.shape}')
    print(f'Labels shape : {all_labels.shape}')
    save_pickle({'embeds': all_embeds,
                 'preds': all_preds,
                 'labels': all_labels}, os.path.join(out_dir, 'noisy_training__valid.pickle'))


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
    'dermamnist': 7
}

if __name__ == "__main__":
    print ("starting")
    # main_imnet1k()
    seed = 42
    opt = 'Adam'
    n_epochs = 50
    batch_size = 64
    epsilon = 0.2
    model_name = 'resnet18'
    dataset_name = 'organsmnist'
    num_classes = dataset_to_n_classes_map[dataset_name]
    out_dir = f'./{dataset_name}/data'
    print ("prepare data and model")
    train_loader, val_loader, test_loader, train_transition_matrix, train_labels, model = prep_model_and_data(dataset_name=dataset_name)
    model = train(model = model, train_dataloader=train_loader, valid_dataloader=val_loader)
    save(model, test_loader, val_loader, out_dir)
