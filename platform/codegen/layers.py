from typing import Dict, Any, List


LAYER_TEMPLATES = {
    "conv2d": "new Conv2d({in_channels}, {out_channels}, {kernel_size}, {stride}, {padding})",
    "batchnorm2d": "new BatchNorm2d({num_features})",
    "layernorm": "new LayerNorm({{{normalized_shape}}})",
    "linear": "new Linear({in_features}, {out_features})",
    "relu": "new ReLU()",
    "sigmoid": "new Sigmoid()",
    "tanh": "new Tanh()",
    "softmax": "new Softmax({dim})",
    "leakyrelu": "new LeakyReLU({negative_slope}f)",
    "maxpool2d": "new MaxPool2d({kernel_size})",
    "avgpool2d": "new AvgPool2d({kernel_size})",
    "dropout": "new Dropout({p}f)",
    "flatten": "new Flatten()",
    "embedding": "new Embedding({num_embeddings}, {embedding_dim})",
    "lstm": "new LSTM({input_size}, {hidden_size}, true)",
    "gru": "new GRU({input_size}, {hidden_size}, true)",
}


def fill_template(template: str, params: dict, layer_type: str) -> str:
    defaults = {
        "stride": 1,
        "padding": 0,
        "momentum": 0.9,
        "beta1": 0.9,
        "beta2": 0.999,
        "negative_slope": 0.01,
        "p": 0.5,
        "dim": -1,
    }

    aliases = {
        "vocab_size": "num_embeddings",
        "embed_dim": "embedding_dim",
    }

    aliased_params = {}
    for key, value in params.items():
        if key in aliases:
            aliased_params[aliases[key]] = value
        aliased_params[key] = value

    filled_params = {**defaults, **aliased_params}
    try:
        return template.format(**filled_params)
    except KeyError as e:
        raise ValueError(f"Missing required parameter {e} for layer {layer_type}")


def generate_layers(layers: List[Dict[str, Any]]) -> str:
    lines = []
    for layer in layers:
        layer_type = layer["type"].lower()
        params = layer.get("params", {})

        # "transformer" is a composite; text models use TransformerLM template, not Sequential
        if layer_type == "transformer":
            continue

        if layer_type not in LAYER_TEMPLATES:
            raise ValueError(f"Unknown layer type: {layer_type}")

        template = LAYER_TEMPLATES[layer_type]
        filled = fill_template(template, params, layer_type)
        lines.append(f"        {filled}")

    return ",\n".join(lines)
