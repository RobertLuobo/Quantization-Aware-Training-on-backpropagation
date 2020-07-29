import torch

class cfg:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    batch_size = 128
    test_batch_size = 128

    input_size = 224
    epoch = 10
    lr = 0.01
    momentum = 0.9
    seed = 1

    # QAT cfg
    start_QAT_epoch = 4
    num_bits = 8

    # log config
    log_interval = 39
    save_model = False
    no_cuda = False

    # file path cfg
    dataset_root = "../dataCifar"
    logger_path = '../logger/log'