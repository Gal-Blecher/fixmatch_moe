import torch.optim as optim
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import itertools
from config import setup
import json
import pickle
from utils import get_logger

def train_vib(model, dataset):
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
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=setup['lr'],
                          momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=setup['n_epochs'])
    model.test_acc = []
    for epoch in range(setup['n_epochs']):
        model.train()
        running_loss = 0
        correct = 0
        total = 0
        for batch_idx, (inputs, targets) in enumerate(dataset['labeled_loader']):
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            z, classification_output = model(inputs)
            net_loss = criterion(classification_output, targets)
            loss = net_loss + setup['kl_vib_coeff'] * model.kl_loss.mean()

            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = classification_output.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            if batch_idx % 50 == 0:
                logger.info(f'batch_idx: {batch_idx}, loss: {loss.item()}')
            break
        acc_train = round((correct/total)*100, 2)
        logger.info(f'epoch: {epoch}, train accuracy: {acc_train}')

        scheduler.step()

        acc_test = vib_test(dataset['test_loader'], model)
        model.test_acc.append(acc_test)
        logger.info(f'epoch: {epoch}, test accuracy: {round(acc_test, 2)}')
        with open(f"{path}/current_epoch.txt", "w") as file:
            file.write(f'{epoch}')
        if acc_test == max(model.test_acc):
            logger.info('--------------------------------------------saving model--------------------------------------------')
            torch.save(model, f'{path}/model.pkl')
        if early_stop(model.test_acc):
            with open(f'{path}/acc_test.pkl', 'wb') as f:
                pickle.dump(model.test_acc, f)
            return

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