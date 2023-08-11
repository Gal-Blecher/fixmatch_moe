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

def tensors_to_pil_images(tensor_batch):
    pil_images = [transforms.functional.to_pil_image(tensor) for tensor in tensor_batch]
    return pil_images

def train_vib(model, dataset):
    weak_transforms = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    strong_transforms = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandAugment(num_ops=2, magnitude=setup['randaugment_magnitude']),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    logger = get_logger(setup['experiment_name'])
    for key, value in setup.items():
        to_log = str(key) + ': ' + str(value)
        logger.info(to_log)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    path = './models/' + setup['experiment_name']
    model = model.to(device)
    logger.info(f'training with device: {device}')
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=setup['lr'],
                          momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=setup['n_epochs'])
    model.test_acc = []
    for epoch in range(setup['n_epochs']):
        labeled_iter = iter(cycle(dataset['labeled_loader']))
        unlabeled_iter = iter(dataset['unlabeled_loader'])
        model.train()
        running_loss = 0
        correct = 0
        total = 0
        batch_idx = 0
        for labeled_data, unlabeled_data in zip(labeled_iter, unlabeled_iter):
            # labeled batch
            labeled_inputs, targets = labeled_data[0].to(device), labeled_data[1].to(device)
            pil_images = tensors_to_pil_images(tensor_batch=labeled_inputs)
            labeled_weak_augmented_images = [weak_transforms(image) for image in pil_images]
            # labeled_strong_augmented_images = [strong_transforms(image) for image in labeled_inputs]
            # plot_augmented_images(labeled_weak_augmented_images, labeled_strong_augmented_images)
            labeled_weak_augmented_tensors = torch.stack(labeled_weak_augmented_images).to(device)
            # labeled_strong_augmented_tensors = torch.stack(labeled_strong_augmented_images).to(device)
            weak_labeled_z, weak_labeled_classification = model(labeled_weak_augmented_tensors)
            # strong_labeled_z, strong_labeled_classification = model(labeled_strong_augmented_tensors)
            supervised_loss = criterion(weak_labeled_classification, targets) + setup['kl_vib_coeff'] * model.kl_loss.mean()

            # unlabeled batch
            unlabeled_inputs = unlabeled_data[0].to(device)
            pil_images = tensors_to_pil_images(tensor_batch=unlabeled_inputs)
            unlabeled_weak_augmented_images = [weak_transforms(image) for image in pil_images]
            unlabeled_strong_augmented_images = [strong_transforms(image) for image in unlabeled_inputs]
            # plot_augmented_images(unlabeled_weak_augmented_images, unlabeled_strong_augmented_images)
            unlabeled_weak_augmented_tensors = torch.stack(unlabeled_weak_augmented_images).to(device)
            unlabeled_strong_augmented_tensors = torch.stack(unlabeled_strong_augmented_images).to(device)
            weak_unlabeled_z, weak_unlabeled_classification = model(unlabeled_weak_augmented_tensors)
            _, _ = model(unlabeled_inputs)
            reconstruction_loss = model.reconstruction_loss

            strong_unlabeled_z, strong_unlabeled_classification = model(unlabeled_strong_augmented_tensors)
            _, weak_unlabeled_classification_pseudo = weak_unlabeled_classification.max(1)
            weak_unlabeled_classification_probs = F.softmax(weak_unlabeled_classification, dim=1)

            confidence_mask = weak_unlabeled_classification_probs.max(1)[0] > setup['confidence_th']
            weak_unlabeled_classification_pseudo = weak_unlabeled_classification_pseudo[confidence_mask]
            strong_unlabeled_classification = strong_unlabeled_classification[confidence_mask]
            if weak_unlabeled_classification_pseudo.shape[0] > 0:
                unsupervied_loss = criterion(strong_unlabeled_classification, weak_unlabeled_classification_pseudo)
            else:
                unsupervied_loss = 0

            loss = supervised_loss + setup['unsupervised_loss_coeff'] * unsupervied_loss + setup['reconstruction_coeff'] * reconstruction_loss

            optimizer.zero_grad()

            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = weak_labeled_classification.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            if batch_idx % 50 == 0:
                logger.info(f'batch_idx: {batch_idx}, loss: {round(running_loss/50, 4)}')
                logger.info(f'supervised loss: {round(supervised_loss.item(), 4)}')
                logger.info(f'reconstruction loss: {round(reconstruction_loss.item(), 4)}')
                running_loss = 0
            batch_idx += 1
        acc_train = round((correct/(total+0.00001))*100, 2)
        logger.info(f'epoch: {epoch}, train accuracy: {acc_train}')

        scheduler.step()

        acc_test = vib_test(dataset['test_loader'], model)
        model.test_acc.append(acc_test)
        logger.info(f'epoch: {epoch}, test accuracy: {round(acc_test, 2)}')
        if acc_test == max(model.test_acc):
            logger.info('--------------------------------------------saving model--------------------------------------------')
            torch.save(model.state_dict(), f'{path}/model.pkl')


def moe_train_vib(model, dataset):
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
    criterion = nn.CrossEntropyLoss()
    optimizer_experts = optim.SGD(itertools.chain(*experts_params), lr=setup['lr'],
                          momentum=0.9, weight_decay=5e-4)
    scheduler_experts = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_experts, T_max=setup['n_epochs'])
    optimizer_router = optim.SGD(router_params, lr=setup['router_lr'],
                          momentum=0.9, weight_decay=5e-4)
    scheduler_router = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_router, T_max=setup['n_epochs'])
    model.test_acc = []
    for epoch in range(setup['n_epochs']):
        model.train()
        running_loss = 0
        correct = 0
        total = 0
        if epoch % 2 == 0:
            loader = dataset['labeled_trainloader']
        else:
            loader = dataset['unlabeled_trainloader']
        for batch_idx, (inputs, targets) in enumerate(loader):
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer_experts.zero_grad()
            optimizer_router.zero_grad()
            outputs, att_weights = model(inputs)
            net_loss = criterion(outputs, targets)
            experts_loss_ = experts_loss(targets, att_weights.squeeze(2), model)
            kl_loss = kl_divergence(att_weights.sum(0))
            loss = net_loss + setup['experts_coeff'] * experts_loss_ + setup['kl_coeff'] * kl_loss

            loss.backward()
            optimizer_experts.step()
            optimizer_router.step()

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            if batch_idx % 50 == 0:
                logger.info(f'batch_idx: {batch_idx}, experts ratio: {att_weights.sum(0).data.T}')
        acc_train = round((correct/total)*100, 2)
        logger.info(f'epoch: {epoch}, train accuracy: {acc_train}')

        scheduler_experts.step()
        scheduler_router.step()
        if epoch % 1 == 0 or epoch > 150:
            acc_test = moe_test(dataset['test_loader'], model)
            model.test_acc.append(acc_test)
            logger.info(f'epoch: {epoch}, test accuracy: {round(acc_test, 2)}')
            with open(f"{path}/current_epoch.txt", "w") as file:
                file.write(f'{epoch}')
            if acc_test == max(model.test_acc):
                logger.info('--------------------------------------------saving model--------------------------------------------')
                torch.save(model, f'{path}/model.pkl')
                with open(f"{path}/config.txt", "w") as file:
                    file.write(json.dumps(setup))
                    file.write(json.dumps(train_config))
                with open(f"{path}/accuracy.txt", "w") as file:
                    file.write(f'{epoch}: {acc_test}')
            if early_stop(model.test_acc):
                with open(f'{path}/acc_test.pkl', 'wb') as f:
                    pickle.dump(model.test_acc, f)
                return

    with open(f'{path}/acc_test.pkl', 'wb') as f:
        pickle.dump(model.test_acc, f)

def moe_test(test_loader, model):
    device = train_config['device']
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs, att_weights = model(inputs)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
        acc = round((correct / total)*100, 2)
        return acc

def vib_test(test_loader, model):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            z, classification_output = model(inputs)
            _, predicted = classification_output.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
        acc = round((correct / total)*100, 2)
        return acc

def experts_loss(labels, att_weights, model):
    device = train_config['device']
    labels = labels.to(device)
    criterion = nn.CrossEntropyLoss(reduction='none')
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

def kl_divergence(vector):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    n = vector.size(0)
    uniform = (torch.ones(n) / n).to(device)
    p = vector / vector.sum()
    return (p * torch.log(p / uniform)).sum()

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

def plot_augmented_images(weak_augmented_images, strong_augmented_images):
    num_images = 5
    fig, axes = plt.subplots(num_images, 2, figsize=(8, 12))

    for i in range(num_images):
        weak_image = weak_augmented_images[i]
        strong_image = strong_augmented_images[i]

        # Transpose the images if necessary (from shape (3, 32, 32) to (32, 32, 3))
        if weak_image.shape[0] == 3:
            weak_image = np.transpose(weak_image, (1, 2, 0))
        if strong_image.shape[0] == 3:
            strong_image = np.transpose(strong_image, (1, 2, 0))

        axes[i, 0].imshow(weak_image)
        axes[i, 0].set_title('Weak Augmentation')
        axes[i, 0].axis('off')

        axes[i, 1].imshow(strong_image)
        axes[i, 1].set_title('Strong Augmentation')
        axes[i, 1].axis('off')

    plt.tight_layout()
    plt.show()