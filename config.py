


setup = {'n_epochs': 200,
         'batch_size': 128,
         'lr': 0.001,
         'router_lr': 0.01,
         'kl_coeff': 0,
         'experts_coeff': 1.0,
         'expert_type': 'VIBNet',
         'kl_vib_coeff': 1e-7,
         'latent_dim': 256,
         'dataset_name': 'cifar10',
         'n_experts': 1,
         'experiment_name': 'reconstruction_only_2_ld_256',
         'model_eval_path': None,
         'ssl_labels': 1000,
         'label_all': False,
         'labeled_only': False,
         'unlabeled_only': True,
         'confidence_th': 0.95,
         'unsupervised_loss_coeff': 0,
         'randaugment_magnitude': 20,
         'reconstruction_coeff': 1.0,
         'supervised_coeff': 0
         }