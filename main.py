import train
from config import setup
import torch
import datasets
import build
import os
import nets
# from torchsummary import summary
import subprocess

torch.manual_seed(42)

if __name__ == '__main__':

    path = './models/' + setup['experiment_name']
    if not os.path.exists(path):
        os.makedirs(path)

    dataset = datasets.get_dataset()
    model = build.build_model()

    if setup['n_experts'] == 1:
        model = nets.Ifeel_fc(42)
        train.train_vib(model, dataset)
    else:
        train.moe_train(model, dataset)