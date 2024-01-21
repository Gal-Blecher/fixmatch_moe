import torch
import torch.nn as nn
import torch.nn.functional as F
from config import setup
import matplotlib.pyplot as plt
import torchvision
from config import setup




# class VIBNet(nn.Module):
#     def __init__(self, seed, latent_dim=setup['latent_dim'], num_classes=10):
#         super(VIBNet, self).__init__()
#         torch.manual_seed(seed)
#         self.encoder = ResNet18(seed)
#         # self.decoder = VIB_Decoder(setup['latent_dim'])
#         self.decoder = ResNet18Dec(setup['latent_dim'])
#
#         self.fc_combined = nn.Linear(512, latent_dim * 2)
#         self.fc_classifier = nn.Linear(latent_dim, num_classes)
#
#     def reparameterize(self, mean, logvar):
#         std = torch.exp(0.5 * logvar)
#         eps = torch.randn_like(std)
#         z = mean + eps * std
#         return z
#
#     def forward(self, x):
#         x_input = x
#         x = self.encoder(x)
#         combined = self.fc_combined(x)
#         mean, logvar = torch.chunk(combined, 2, dim=1)
#         z = self.reparameterize(mean, logvar)
#         classification_output = self.fc_classifier(z)
#
#         self.kl_loss = -0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp(), dim=1)
#         self.out = classification_output
#         # self.x_hat = self.decoder(z)
#         # self.reconstruction_loss = ((x_input - self.x_hat) ** 2).mean()
#         self.z = z
#         self.classification_output = classification_output
#         logits = F.softmax(classification_output, dim=1)
#
#         return z, logits


class Ifeel_fc(nn.Module):
    def __init__(self, seed, num_classes=2):
        super(Ifeel_fc, self).__init__()
        torch.manual_seed(seed)
        input_dim = 16

        self.fc1 = nn.Linear(input_dim, setup['latent_dim'])
        self.fc2 = nn.Linear(setup['latent_dim'], setup['latent_dim'])
        self.fc_classifier = nn.Linear(setup['latent_dim'], 1)

    def forward(self, x):
        x = self.fc1(x)
        x = torch.relu(x)
        z = self.fc2(x)
        classification_output = self.fc_classifier(z)
        self.out = classification_output
        self.z = z
        self.classification_output = classification_output
        logits = torch.sigmoid(classification_output)

        return z, logits










