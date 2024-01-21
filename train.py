import torch.optim as optim
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import itertools
from config import setup
import json
import pickle
from utils import get_logger
import torchvision.transforms as transforms
from itertools import cycle
import matplotlib.pyplot as plt
import numpy as np
import torch.nn.functional as F
from sklearn.metrics import classification_report


def moe_train(model, dataset):
    pos_weight = calculate_pos_weights(dataset['labeled_loader'].dataset.tensors[1])

    logger = get_logger(setup['experiment_name'])
    for key, value in setup.items():
        to_log = str(key) + ': ' + str(value)
        logger.info(to_log)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    path = './models/' + setup['experiment_name']
    if device == 'cuda':
        model = torch.nn.DataParallel(model)
        cudnn.benchmark = True
    model = model.to(device)
    logger.info(f'training with device: {device}')
    router_params = model.router.parameters()
    experts_params = get_experts_params_list(model)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer_experts = optim.SGD(itertools.chain(*experts_params), lr=setup['lr'],
                          momentum=0.9, weight_decay=5e-4)
    scheduler_experts = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_experts, T_max=setup['n_epochs'])
    optimizer_router = optim.SGD(router_params, lr=setup['router_lr'],
                          momentum=0.9, weight_decay=5e-4)
    scheduler_router = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_router, T_max=setup['n_epochs'])
    model.test_acc = []
    for epoch in range(setup['n_epochs']):
        labeled_iter = iter(cycle(dataset['labeled_loader']))
        unlabeled_iter = iter(dataset['unlabeled_loader'])
        model.train()
        batch_idx = 0
        running_loss = 0
        correct = 0
        total = 0
        for labeled_data, unlabeled_data in zip(labeled_iter, unlabeled_iter):
            batch_idx += 1
            optimizer_experts.zero_grad()
            optimizer_router.zero_grad()

            # labeled data
            labeled_inputs, targets = labeled_data[0].to(device), labeled_data[1].to(device)
            labeled_weak_augmented_tensors = add_gaussian_noise(labeled_inputs, std=0.01)

            labeled_strong_augmented_tensors = add_gaussian_noise(labeled_inputs, std=0.1)

            outputs, att_weights = model(labeled_weak_augmented_tensors)
            net_loss_supervised = criterion(outputs, targets)
            experts_loss_supervised = experts_loss(targets, att_weights.squeeze(2), model, pos_weight)
            kl_loss_balance = kl_divergence(att_weights.sum(0))
            supervised_loss = setup['net_loss_supervised_coeff'] * net_loss_supervised + setup['experts_loss_supervised_coeff'] * experts_loss_supervised + setup['kl_coeff'] * kl_loss_balance

            # unlabeled data
            unlabeled_inputs = unlabeled_data[0].to(device)
            unlabeled_weak_augmented_tensors = add_gaussian_noise(labeled_inputs, std=0.01)
            unlabeled_strong_augmented_tensors = add_gaussian_noise(labeled_inputs, std=0.1)

            unsupervied_loss_experts = 0
            for exp in range(1, setup['n_experts'] + 1): # the moe experts returns logits
                expert_name = f"expert{exp}"
                expert = getattr(model, expert_name)
                weak_unlabeled_z, weak_unlabeled_classification = expert(unlabeled_weak_augmented_tensors)

                strong_unlabeled_z, strong_unlabeled_classification = expert(unlabeled_strong_augmented_tensors)
                weak_unlabeled_classification_pseudo = (weak_unlabeled_classification > 0.5).float()

                confidence_mask = weak_unlabeled_classification.max(1)[0] > setup['confidence_th']
                weak_unlabeled_classification_pseudo = weak_unlabeled_classification_pseudo[confidence_mask]
                strong_unlabeled_classification = strong_unlabeled_classification[confidence_mask]
                if weak_unlabeled_classification_pseudo.shape[0] > 0:
                    unsupervied_loss = criterion(strong_unlabeled_classification, weak_unlabeled_classification_pseudo)
                else:
                    unsupervied_loss = 0
                unsupervied_loss_experts += unsupervied_loss

                # fixmatch for labeled data
                strong_unlabeled_z, strong_labeled_classification = expert(labeled_strong_augmented_tensors)
                weak_labeled_z, weak_labeled_classification = expert(labeled_weak_augmented_tensors)
                weak_labeled_classification_pseudo = (weak_labeled_classification > 0.5).float()

                confidence_mask = weak_labeled_classification.max(1)[0] > setup['confidence_th']
                weak_labeled_classification_pseudo = weak_labeled_classification_pseudo[confidence_mask]
                strong_labeled_classification = strong_labeled_classification[confidence_mask]
                if weak_labeled_classification_pseudo.shape[0] > 0:
                    unsupervied_loss_labeled = criterion(strong_labeled_classification, weak_labeled_classification_pseudo)
                else:
                    unsupervied_loss_labeled = 0
                unsupervied_loss_experts += unsupervied_loss_labeled




            loss = setup['supervised_loss_coeff'] * supervised_loss + setup['unsupervised_loss_coeff'] * unsupervied_loss_experts

            loss.backward()
            optimizer_experts.step()
            optimizer_router.step()

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            if batch_idx % 5 == 0:
                logger.info(f'batch_idx: {batch_idx}, experts ratio: {att_weights.sum(0).data.T}')
                logger.info(f'batch_idx: {batch_idx}, loss: {round(running_loss/50, 4)}')
                logger.info(f'batch_idx: {batch_idx}, supervised_loss: {round(supervised_loss.item(), 4)}')
                logger.info(f'batch_idx: {batch_idx}, experts_loss_supervised: {round(experts_loss_supervised.item(), 4)}')
                logger.info(f'batch_idx: {batch_idx}, kl_loss_balance: {round(kl_loss_balance.item(), 4)}')
                logger.info(f'batch_idx: {batch_idx}, mean confidence: {round(weak_unlabeled_classification.max(1)[0].mean().item(), 4)}')
                try:
                    logger.info(f'batch_idx: {batch_idx}, unsupervied_loss: {round(unsupervied_loss_experts.item(), 4)}')
                except:
                    logger.info(f'batch_idx: {batch_idx}, unsupervied_loss: 0.0')


                running_loss = 0

        scheduler_experts.step()
        scheduler_router.step()

        test_report = moe_test(dataset['test_loader'], model)
        f1_class_1 = test_report['1.0']['f1-score']
        model.test_acc.append(f1_class_1)
        logger.info(f'epoch: {epoch}, test f1_class_1: {round(f1_class_1, 3)}')
        if f1_class_1 == max(model.test_acc):
            logger.info('--------------------------------------------saving model--------------------------------------------')
            torch.save(model.state_dict(), f'{path}/model.pkl')
            with open(f'{path}/report.json', 'w') as file:
                json.dump(test_report, file, indent=4)

    with open(f'{path}/acc_test.pkl', 'wb') as f:
        pickle.dump(model.test_acc, f)

def moe_test(test_loader, model):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.eval()
    predicted_list = []
    target_list = []
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs, att_weights = model(inputs)
            total += targets.size(0)
            predicted_list += outputs.tolist()
            target_list += targets.tolist()
    best_report_metric = 0
    best_report = None
    for th in np.arange(0.3, 0.9, 0.01):
        prediction_arr = np.array(predicted_list).flatten()
        prediction_arr_label = (prediction_arr > th).astype(int)
        report = classification_report(target_list, prediction_arr_label, output_dict=True)
        if report['1.0']['f1-score'] > best_report_metric:
            best_report_metric = report['1.0']['f1-score']
            report['th'] = th
            best_report = report

    return best_report

def experts_loss(labels, att_weights, model, pos_weight):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    labels = labels.to(device)
    criterion = nn.BCEWithLogitsLoss(reduction='none', pos_weight=pos_weight)
    if model.n_experts == 2:
        experts_loss_ = torch.stack(
            (
            criterion(model.expert1.out, labels),
            criterion(model.expert2.out, labels)
            )
            , dim=1)

    if model.n_experts == 4:
        experts_loss_ = torch.stack(
            (
            criterion(model.expert1.out, labels),
            criterion(model.expert2.out, labels),
            criterion(model.expert3.out, labels),
            criterion(model.expert4.out, labels)
            )
            , dim=1)

    if model.n_experts == 8:
        experts_loss_ = torch.stack(
            (
            criterion(model.expert1.out, labels),
            criterion(model.expert2.out, labels),
            criterion(model.expert3.out, labels),
            criterion(model.expert4.out, labels),
            criterion(model.expert5.out, labels),
            criterion(model.expert6.out, labels),
            criterion(model.expert7.out, labels),
            criterion(model.expert8.out, labels)
            )
            , dim=1)

    if model.n_experts == 16:
        experts_loss_ = torch.stack(
            (
            criterion(model.expert1.out, labels),
            criterion(model.expert2.out, labels),
            criterion(model.expert3.out, labels),
            criterion(model.expert4.out, labels),
            criterion(model.expert5.out, labels),
            criterion(model.expert6.out, labels),
            criterion(model.expert7.out, labels),
            criterion(model.expert8.out, labels),
            criterion(model.expert9.out, labels),
            criterion(model.expert10.out, labels),
            criterion(model.expert11.out, labels),
            criterion(model.expert12.out, labels),
            criterion(model.expert13.out, labels),
            criterion(model.expert14.out, labels),
            criterion(model.expert15.out, labels),
            criterion(model.expert16.out, labels)
            )
            , dim=1)

    att_weights_flattened = torch.flatten(att_weights)
    experts_loss_flattend = torch.flatten(experts_loss_)
    weighted_experts_loss = torch.dot(att_weights_flattened, experts_loss_flattend)
    return weighted_experts_loss / labels.shape[0]

def early_stop(acc_test_list):
    curr_epoch = len(acc_test_list)
    best_epoch = acc_test_list.index(max(acc_test_list))
    if curr_epoch - best_epoch > 300:
        return True
    else:
        return False

def get_experts_params_list(model):
    if model.n_experts == 2:
        experts_params = [model.expert1.parameters(), model.expert2.parameters()]
        return experts_params
    if model.n_experts == 4:
        experts_params = [model.expert1.parameters(), model.expert2.parameters(),
                          model.expert3.parameters(), model.expert4.parameters()]
        return experts_params
    if model.n_experts == 8:
        experts_params = [model.expert1.parameters(), model.expert2.parameters(),
                          model.expert3.parameters(), model.expert4.parameters(),
                          model.expert5.parameters(), model.expert6.parameters(),
                          model.expert7.parameters(), model.expert8.parameters()]
        return experts_params
    if model.n_experts == 16:
        experts_params = [model.expert1.parameters(), model.expert2.parameters(),
                          model.expert3.parameters(), model.expert4.parameters(),
                          model.expert5.parameters(), model.expert6.parameters(),
                          model.expert7.parameters(), model.expert8.parameters(),
                          model.expert9.parameters(), model.expert10.parameters(),
                          model.expert11.parameters(), model.expert12.parameters(),
                          model.expert13.parameters(), model.expert14.parameters(),
                          model.expert15.parameters(), model.expert16.parameters()]
        return experts_params


import torch


def calculate_pos_weights(Y):
    class_counts = Y.unique(return_counts=True)[1]
    pos_weight = class_counts[0] / class_counts[1]

    return torch.tensor([pos_weight], dtype=torch.float32)

def kl_divergence(vector):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    n = vector.size(0)
    uniform = (torch.ones(n) / n).to(device)
    p = vector / vector.sum()
    return (p * torch.log(p / uniform)).sum()

def add_gaussian_noise(batch_tensors, std):
    """
    Add Gaussian noise with mean 0 and a given standard deviation to each tensor in a batch.

    Parameters:
    batch_tensors (Tensor): A batch of tensors.
    std (float): Standard deviation of the Gaussian noise.

    Returns:
    Tensor: A batch of tensors with Gaussian noise added.
    """
    # Ensure batch_tensors is a batch of tensors
    if batch_tensors.ndim <= 1:
        print(f'check batch tensors dims: {batch_tensors.ndim}')
        return batch_tensors

    # Generate Gaussian noise with mean 0 and the specified standard deviation
    noise = torch.normal(0, std, size=batch_tensors.size())

    # Add noise to the batch of tensors
    noisy_batch = batch_tensors + noise

    return noisy_batch





