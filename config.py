


setup = {'n_epochs': 200,
         'batch_size': 8,
         'lr': 0.01,
         'router_lr': 0.001,
         'kl_coeff': 0.05,
         'experts_coeff': 1.0,
         'expert_type': 'ifeel_fc',
         'kl_vib_coeff': 1e-7,
         'latent_dim': 4,
         'dataset_name': 'cifar10',
         'n_experts': 2,
         'experiment_name': 'ifeel_tests',
         'model_eval_path': None,
         'ssl_labels': 1000,
         'label_all': False,
         'labeled_only': False,
         'unlabeled_only': True,
         'confidence_th': 0.55,
         'unsupervised_loss_coeff': 1.0,
         'randaugment_magnitude': 20,
         'reconstruction_coeff': 0.0001,
         'supervised_loss_coeff': 1.0,
         'net_loss_supervised_coeff': 1.0,
         'experts_loss_supervised_coeff': 1.0
         }