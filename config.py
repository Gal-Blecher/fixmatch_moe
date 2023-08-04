


setup = {'n_epochs': 2,
         'batch_size': 128,
         'lr': 0.1,
         'router_lr': 0.1,
         'kl_coeff': 0,
         'experts_coeff': 1.0,
         'expert_type': 'VIBNet',
         'kl_vib_coeff': 1e-5,
         'latent_dim': 32,
         'dataset_name': 'cifar10',
         'n_experts': 1,
         'experiment_name': 'vib_2',
         'model_eval_path': None,
         'ssl_labels': 49_900,
         'label_all': False,
         'labeled_only': False,
         'unlabeled_only': True
         }