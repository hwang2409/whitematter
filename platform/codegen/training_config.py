OPTIMIZER_TEMPLATES = {
    "sgd": "SGD optimizer(model.parameters(), {learning_rate}f, {momentum}f);",
    "adam": "Adam optimizer(model.parameters(), {learning_rate}f, {beta1}f, {beta2}f);",
}

SCHEDULER_TEMPLATES = {
    "none": "",
    "step": "StepLR scheduler(&optimizer, {step_size}, {gamma}f);",
    "cosine": "CosineAnnealingLR scheduler(&optimizer, {T_max});",
    "exponential": "ExponentialLR scheduler(&optimizer, {gamma}f);",
}


def generate_optimizer(optimizer_config: dict) -> str:
    opt_type = optimizer_config.get("type", "sgd").lower()
    params = optimizer_config.get("params", {})

    if opt_type not in OPTIMIZER_TEMPLATES:
        raise ValueError(f"Unknown optimizer: {opt_type}")

    defaults = {
        "learning_rate": 0.01,
        "momentum": 0.9,
        "beta1": 0.9,
        "beta2": 0.999,
    }

    filled_params = {**defaults, **params}
    return OPTIMIZER_TEMPLATES[opt_type].format(**filled_params)


def generate_scheduler(scheduler_config: dict) -> str:
    sched_type = scheduler_config.get("type", "none").lower()
    params = scheduler_config.get("params", {})

    if sched_type not in SCHEDULER_TEMPLATES:
        return ""

    if sched_type == "none":
        return ""

    defaults = {
        "step_size": 10,
        "gamma": 0.1,
        "T_max": 50,
    }

    filled_params = {**defaults, **params}
    return SCHEDULER_TEMPLATES[sched_type].format(**filled_params)
