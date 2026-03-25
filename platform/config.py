from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
MODELS_DIR = PROJECT_ROOT / "models"
DATA_DIR = PROJECT_ROOT / "data"
UPLOADS_DIR = PROJECT_ROOT / "uploads"
GENERATED_DIR = PROJECT_ROOT / "generated"

DATASETS = {
    "cifar10": {
        "name": "CIFAR-10",
        "description": "60,000 32x32 color images in 10 classes",
        "input_shape": [3, 32, 32],
        "num_classes": 10,
        "classes": ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
    },
    "mnist": {
        "name": "MNIST",
        "description": "70,000 28x28 grayscale handwritten digits",
        "input_shape": [1, 28, 28],
        "num_classes": 10,
        "classes": ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]
    }
}

CIFAR10_MEAN, CIFAR10_STD = [0.4914, 0.4822, 0.4465], [0.2470, 0.2435, 0.2616]
MNIST_MEAN, MNIST_STD = [0.1307], [0.3081]

LAYER_TYPES = {
    "conv2d": {"name": "Conv2d", "params": ["in_channels", "out_channels", "kernel_size", "stride", "padding"]},
    "batchnorm2d": {"name": "BatchNorm2d", "params": ["num_features"]},
    "relu": {"name": "ReLU", "params": []},
    "leakyrelu": {"name": "LeakyReLU", "params": ["negative_slope"]},
    "maxpool2d": {"name": "MaxPool2d", "params": ["kernel_size"]},
    "avgpool2d": {"name": "AvgPool2d", "params": ["kernel_size"]},
    "dropout": {"name": "Dropout", "params": ["p"]},
    "flatten": {"name": "Flatten", "params": []},
    "linear": {"name": "Linear", "params": ["in_features", "out_features"]}
}

OPTIMIZERS = {
    "sgd": {"name": "SGD", "params": {"learning_rate": 0.01, "momentum": 0.9, "weight_decay": 0.0}},
    "adam": {"name": "Adam", "params": {"learning_rate": 0.001, "beta1": 0.9, "beta2": 0.999, "weight_decay": 0.0}}
}

SCHEDULERS = {
    "none": {"name": "None", "params": {}},
    "step": {"name": "StepLR", "params": {"step_size": 10, "gamma": 0.1}},
    "cosine": {"name": "CosineAnnealing", "params": {"eta_min": 0.0}},
    "exponential": {"name": "ExponentialLR", "params": {"gamma": 0.95}}
}

def ensure_dirs():
    MODELS_DIR.mkdir(exist_ok=True)
    UPLOADS_DIR.mkdir(exist_ok=True)
    GENERATED_DIR.mkdir(exist_ok=True)


AUGMENTATIONS = {
    "horizontal_flip": {"name": "Horizontal Flip", "params": {"p": 0.5}},
    "random_crop": {"name": "Random Crop", "params": {"padding": 4}},
    "color_jitter": {"name": "Color Jitter", "params": {"brightness": 0.2, "contrast": 0.2}},
    "normalize": {"name": "Normalize", "params": {}}
}

PRESET_ARCHITECTURES = {
    "simple_cnn_cifar10": {
        "name": "Simple CNN (CIFAR-10)", "dataset": "cifar10",
        "layers": [
            {"type": "conv2d", "params": {"in_channels": 3, "out_channels": 32, "kernel_size": 3, "padding": 1}},
            {"type": "batchnorm2d", "params": {"num_features": 32}}, {"type": "relu", "params": {}},
            {"type": "maxpool2d", "params": {"kernel_size": 2}},
            {"type": "conv2d", "params": {"in_channels": 32, "out_channels": 64, "kernel_size": 3, "padding": 1}},
            {"type": "batchnorm2d", "params": {"num_features": 64}}, {"type": "relu", "params": {}},
            {"type": "maxpool2d", "params": {"kernel_size": 2}}, {"type": "flatten", "params": {}},
            {"type": "linear", "params": {"in_features": 4096, "out_features": 256}}, {"type": "relu", "params": {}},
            {"type": "dropout", "params": {"p": 0.5}},
            {"type": "linear", "params": {"in_features": 256, "out_features": 10}}
        ]
    },
    "vgg_cifar10": {
        "name": "VGG-style CNN (CIFAR-10)", "dataset": "cifar10",
        "layers": [
            {"type": "conv2d", "params": {"in_channels": 3, "out_channels": 64, "kernel_size": 3, "padding": 1}},
            {"type": "batchnorm2d", "params": {"num_features": 64}}, {"type": "relu", "params": {}},
            {"type": "conv2d", "params": {"in_channels": 64, "out_channels": 64, "kernel_size": 3, "padding": 1}},
            {"type": "batchnorm2d", "params": {"num_features": 64}}, {"type": "relu", "params": {}},
            {"type": "maxpool2d", "params": {"kernel_size": 2}},
            {"type": "conv2d", "params": {"in_channels": 64, "out_channels": 128, "kernel_size": 3, "padding": 1}},
            {"type": "batchnorm2d", "params": {"num_features": 128}}, {"type": "relu", "params": {}},
            {"type": "conv2d", "params": {"in_channels": 128, "out_channels": 128, "kernel_size": 3, "padding": 1}},
            {"type": "batchnorm2d", "params": {"num_features": 128}}, {"type": "relu", "params": {}},
            {"type": "maxpool2d", "params": {"kernel_size": 2}},
            {"type": "conv2d", "params": {"in_channels": 128, "out_channels": 256, "kernel_size": 3, "padding": 1}},
            {"type": "batchnorm2d", "params": {"num_features": 256}}, {"type": "relu", "params": {}},
            {"type": "conv2d", "params": {"in_channels": 256, "out_channels": 256, "kernel_size": 3, "padding": 1}},
            {"type": "batchnorm2d", "params": {"num_features": 256}}, {"type": "relu", "params": {}},
            {"type": "maxpool2d", "params": {"kernel_size": 2}}, {"type": "flatten", "params": {}},
            {"type": "linear", "params": {"in_features": 4096, "out_features": 512}}, {"type": "relu", "params": {}},
            {"type": "dropout", "params": {"p": 0.5}},
            {"type": "linear", "params": {"in_features": 512, "out_features": 10}}
        ]
    },
    "simple_cnn_mnist": {
        "name": "Simple CNN (MNIST)", "dataset": "mnist",
        "layers": [
            {"type": "conv2d", "params": {"in_channels": 1, "out_channels": 16, "kernel_size": 3, "padding": 1}},
            {"type": "batchnorm2d", "params": {"num_features": 16}}, {"type": "relu", "params": {}},
            {"type": "maxpool2d", "params": {"kernel_size": 2}},
            {"type": "conv2d", "params": {"in_channels": 16, "out_channels": 32, "kernel_size": 3, "padding": 1}},
            {"type": "batchnorm2d", "params": {"num_features": 32}}, {"type": "relu", "params": {}},
            {"type": "maxpool2d", "params": {"kernel_size": 2}}, {"type": "flatten", "params": {}},
            {"type": "linear", "params": {"in_features": 1568, "out_features": 128}}, {"type": "relu", "params": {}},
            {"type": "linear", "params": {"in_features": 128, "out_features": 10}}
        ]
    }
}
