


setup = {'n_epochs': 200,
         'batch_size': 128,
         'lr': 0.05,
         'router_lr': 0.001,
         'kl_coeff': 0.05,
         'experts_coeff': 1.0,
         'expert_type': 'VIBNet',
         'kl_vib_coeff': 1e-7,
         'latent_dim': 32,
         'dataset_name': 'cifar10',
         'n_experts': 4,
         'experiment_name': 'moe_vib_ssl_28',
         'model_eval_path': None,
         'ssl_labels': 1000,
         'label_all': False,
         'labeled_only': False,
         'unlabeled_only': True,
         'confidence_th': 0.99,
         'unsupervised_loss_coeff': 1.0,
         'randaugment_magnitude': 20,
         'reconstruction_coeff': 0.0001,
         'supervised_loss_coeff': 1.0,
         'net_loss_supervised_coeff': 1.0,
         'experts_loss_supervised_coeff': 1.0
         }