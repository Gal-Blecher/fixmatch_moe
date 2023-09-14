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
    def get_git_info():
        try:
            branch = subprocess.check_output(['git', 'rev-parse', '--abbrev-ref', 'HEAD'], stderr=subprocess.STDOUT,
                                             text=True).strip()
            commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], stderr=subprocess.STDOUT, text=True).strip()
            return branch, commit
        except subprocess.CalledProcessError as e:
            # Handle any errors that occur when running Git commands
            return None, None

    branch, commit = get_git_info()
    setup['commit'] = commit
    setup['branch'] = branch
    path = './models/' + setup['experiment_name']
    if not os.path.exists(path):
        os.makedirs(path)

    dataset = datasets.get_dataset()
    model = build.build_model()

    if setup['n_experts'] == 1:
        model = nets.VIBNet(42)
        train.train_vib(model, dataset)
    else:
        train.moe_train_vib(model, dataset)