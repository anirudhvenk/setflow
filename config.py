import ml_collections

def create_config():
    config = ml_collections.ConfigDict()

    encoder = config.encoder = ml_collections.ConfigDict()
    encoder.input_dim = 2
    encoder.hidden_dim = 64
    encoder.num_heads = 4
    encoder.depth = 32
    # encoder.z_scales = [1, 1, 2, 4, 8, 16, 32]
    encoder.z_scales = [2, 4, 8, 16, 32]
    encoder.z_dim = 16

    decoder = config.decoder = ml_collections.ConfigDict()
    decoder.hidden_dim = 64
    decoder.num_heads = 4
    decoder.conditioning_dim = 16
    decoder.depth = 7
    decoder.dropout = 0.0
    decoder.num_prototypes = 5

    training = config.training = ml_collections.ConfigDict()
    training.lr = 1e-3
    training.epochs = 200
    training.weight_decay = 1e-2
    training.beta1 = 0.9
    training.beta2 = 0.999
    training.type = "mnist"

    return config