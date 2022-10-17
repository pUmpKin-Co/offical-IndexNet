import ml_collections


def get_config():
    config = ml_collections.ConfigDict()
    config.data = ml_collections.ConfigDict()
    config.network = ml_collections.ConfigDict()
    config.schedule = ml_collections.ConfigDict()
    config.deeplab = ml_collections.ConfigDict()
    config.transfer_schedule = ml_collections.ConfigDict()
    config.others = ml_collections.ConfigDict()

    config.data.size = (224, 224)
    config.data.num_classes = 6

    config.schedule.schedule_name = "cosine"
    config.schedule.base_weight_lr = 3e-2
    config.schedule.base_bias_lr = 3e-2
    config.schedule.epochs = 400
    config.schedule.weight_decay = 1.5e-6
    config.schedule.optimizer = "sgd"
    config.schedule.warmup_epochs = 10
    config.schedule.warmup_lr = 5e-7
    config.schedule.accumulate_iter = 1
    config.schedule.decay_epochs = 1
    config.schedule.min_lr = 1e-4
    config.schedule.decay_rate = 0.1  # for step schedule

    config.network.output_size = 7
    config.network.backbone = "resnet50"
    config.network.head_hidden_dim = [2048 * 2]
    config.network.spatial_dimension = [7]
    config.network.head_out_dim = 256
    config.network.base_momentum = 0.996
    config.network.final_momentum = 1
    config.network.loss_weight_alpha = [1.0]
    config.network.no_pyd = False
    config.network.no_index = False

    config.deeplab.backbone = "resnet50"
    config.deeplab.aspp = {
        'in_channels': 2048,
        'out_channels': 512,
        'dilations': [1, 6, 12, 18],
    }
    config.deeplab.shortcut = {
        'in_channels': 256,
        'out_channels': 48,
    }
    config.deeplab.decoder = {
        'in_channels': 560,
        'out_channels': 512,
        'dropout': 0.1,
    }
    config.deeplab.outstride = 16

    config.transfer_schedule.backbone_lr = 0.001
    config.transfer_schedule.lr = config.transfer_schedule.backbone_lr * 10
    config.transfer_schedule.epochs = 100
    config.transfer_schedule.weight = None
    config.transfer_schedule.momentum = 0.9
    config.transfer_schedule.weight_decay = 1.e-4
    config.transfer_schedule.validate_interval = 1
    config.transfer_schedule.save_val_results = False

    config.others.vis_num_samples = 0  # Samples to visualize for validation and test
    config.others.enable_vis = False  # whether to visualize result

    return config