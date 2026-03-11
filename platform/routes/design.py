"""
Architecture design endpoints (LLM-powered suggestions, validation, refinement).
"""

import logging
from typing import Dict, Any, Optional

from fastapi import APIRouter, HTTPException

from schemas import DesignRequest, RefineRequest, DesignHelpRequest
from dependencies import dataset_manager, dataset_service, code_generator, llm_service

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post("/design/suggest")
async def suggest_architecture(request: DesignRequest):
    """Get LLM-suggested architecture for a dataset."""
    metadata = dataset_manager.get_metadata(request.dataset_id)
    if metadata:
        dataset_info = metadata.to_dict()
    else:
        dataset_info = dataset_service.get_dataset(request.dataset_id)
        if not dataset_info:
            raise HTTPException(status_code=404, detail="Dataset not found")

    if dataset_info.get('data_type') == 'text':
        preprocessing_config = dataset_info.get('preprocessing_config', {})
        dataset_info['vocab_size'] = preprocessing_config.get('vocab_size', dataset_info.get('num_classes', 100))
        dataset_info['seq_length'] = preprocessing_config.get('seq_length', 128)
        dataset_info['tokenizer_type'] = preprocessing_config.get('tokenizer_type', 'character')

    try:
        result = llm_service.suggest_architecture(
            dataset_info=dataset_info,
            user_prompt=request.prompt
        )

        if dataset_info.get('data_type') == 'text':
            result["architecture"]["vocab_size"] = dataset_info.get('vocab_size')
            result["architecture"]["seq_length"] = dataset_info.get('seq_length')
            result["architecture"]["data_type"] = "text"

        validation = code_generator.validate_architecture(result["architecture"])

        logger.info("Suggested architecture for dataset %s", request.dataset_id)
        return {
            "architecture": result["architecture"],
            "explanation": result["explanation"],
            "validation": validation
        }
    except (ValueError, RuntimeError, ConnectionError) as e:
        logger.exception("LLM architecture suggestion failed for dataset %s", request.dataset_id)
        raise HTTPException(status_code=500, detail=f"LLM error: {str(e)}") from e


@router.post("/design/validate")
async def validate_architecture(architecture: Dict[str, Any]):
    """Validate an architecture specification."""
    validation = code_generator.validate_architecture(architecture)
    return validation


@router.post("/design/refine")
async def refine_architecture(request: RefineRequest):
    """Refine an architecture based on feedback."""
    try:
        result = llm_service.refine_architecture(
            current_architecture=request.architecture,
            feedback=request.feedback
        )

        validation = code_generator.validate_architecture(result["architecture"])

        logger.info("Refined architecture based on feedback")
        return {
            "architecture": result["architecture"],
            "explanation": result["explanation"],
            "validation": validation
        }
    except (ValueError, RuntimeError, ConnectionError) as e:
        logger.exception("LLM architecture refinement failed")
        raise HTTPException(status_code=500, detail=f"LLM error: {str(e)}") from e


@router.post("/design/help")
async def design_help(request: DesignHelpRequest):
    """LLM-powered design assistant for neural network architecture questions."""
    message = request.message.lower()
    context = request.context or {}
    dataset_type = context.get("dataset_type", "image")

    responses = {
        "conv": "**Convolutional layers (Conv2d)** extract spatial features from images using learnable filters.\n\n- `in_channels`: Number of input channels (1 for grayscale, 3 for RGB)\n- `out_channels`: Number of filters (more = more features, but slower)\n- `kernel_size`: Filter size (3x3 is most common)\n- `stride`: Step size (1 = dense, 2 = downsamples)\n- `padding`: Border padding (1 for 3x3 kernel preserves size)\n\n**Typical progression**: 32 → 64 → 128 filters",
        "lstm": "**LSTM layers** are great for sequential data like text.\n\n- `input_size`: Size of input features (embedding dim for text)\n- `hidden_size`: Size of hidden state (128-512 typical)\n- `num_layers`: Stacked LSTMs (2-3 is common)\n\n**For language models**: Use Embedding → LSTM → Linear\n\n**Tip**: Larger hidden_size = more capacity but slower training",
        "dropout": "**Dropout** randomly zeros neurons during training to prevent overfitting.\n\n- `p`: Dropout probability (0.1-0.5 typical)\n- Use **0.2-0.3** for most cases\n- Use **0.5** for very large models or small datasets\n- Place after Dense/Linear layers, not after BatchNorm",
        "batch": "**BatchNorm** normalizes activations, allowing faster training and higher learning rates.\n\n- Place **after Conv2d/Linear**, **before activation**\n- Helps with gradient flow in deep networks\n- Allows learning rates 10x higher\n- Acts as mild regularization",
        "embed": "**Embedding layers** convert token IDs to dense vectors.\n\n- `num_embeddings`: Vocabulary size\n- `embedding_dim`: Vector size (64-256 typical)\n\n**For character-level**: 64-128 dim\n**For word-level**: 128-300 dim",
        "linear": "**Linear (Dense) layers** perform `output = input @ weights + bias`.\n\n- `in_features`: Input size\n- `out_features`: Output size\n- Final layer should match number of classes\n\n**Tip**: Add activation (ReLU) between Linear layers",
        "learning": "**Learning Rate** controls step size during optimization.\n\n- **Too high**: Training diverges, loss explodes\n- **Too low**: Training is slow, may get stuck\n\n**Recommended starting points**:\n- Adam: 0.001 - 0.0001\n- SGD: 0.01 - 0.1\n\n**For fine-tuning**: Use 10x smaller\n**If loss plateaus**: Reduce by 2-10x",
        "epoch": "**Epochs** = full passes through training data.\n\n**Guidelines**:\n- Small dataset (<1K): 50-100 epochs\n- Medium dataset (1K-10K): 20-50 epochs\n- Large dataset (>10K): 10-30 epochs\n\n**For language models**: Character-level needs more epochs (30-100)\n\n**Signs to stop**: Validation loss stops improving",
        "batch_size": "**Batch Size** affects training speed and generalization.\n\n- **Larger** (64-256): Faster, more stable, may generalize worse\n- **Smaller** (16-32): Slower, noisier, often generalizes better\n\n**Memory limited?** Use smaller batch\n**Recommended**: 32-64 for most cases",
        "optimizer": "**Optimizers** update weights based on gradients.\n\n**Adam** (recommended for most cases):\n- Adaptive learning rates\n- Works well out-of-the-box\n- lr=0.001 typical\n\n**SGD with momentum**:\n- Often better final accuracy\n- Needs more tuning\n- lr=0.01-0.1, momentum=0.9",
        "cnn": "**CNN Architecture Pattern** for images:\n\n```\nConv2d → BatchNorm → ReLU → MaxPool\n↓ (repeat 2-4 times, increasing filters)\nFlatten → Linear → ReLU → Dropout → Linear\n```\n\n**Filter progression**: 32 → 64 → 128\n**Always use**: BatchNorm after Conv, Dropout before final Linear",
        "language": "**Language Model Architecture**:\n\n```\nEmbedding(vocab_size, 128-256)\n↓\nLSTM(embed_dim, 256-512, layers=2)\n↓\nDropout(0.3)\n↓\nLinear(hidden_size, vocab_size)\n```\n\n**Tips**:\n- More epochs = better (30-100)\n- Lower learning rate (0.001-0.002)\n- Character-level is easier to train than word-level",
        "overfit": "**Signs of Overfitting**:\n- Training loss decreases but validation loss increases\n- Training accuracy >> validation accuracy\n\n**Solutions**:\n1. Add **Dropout** (0.2-0.5)\n2. Add **data augmentation**\n3. Use **smaller model**\n4. **Early stopping**\n5. Reduce epochs",
        "underfit": "**Signs of Underfitting**:\n- Both training and validation loss are high\n- Model isn't learning patterns\n\n**Solutions**:\n1. **Larger model** (more layers/filters)\n2. **Train longer** (more epochs)\n3. **Higher learning rate**\n4. **Remove regularization** (less dropout)\n5. Check data quality",
        "accuracy": "**Improving Accuracy**:\n\n1. **Train longer**: More epochs (but watch for overfitting)\n2. **Bigger model**: More layers, more filters/units\n3. **Better hyperparameters**: Lower LR, tune batch size\n4. **Data augmentation**: For images, use flips/rotations\n5. **Learning rate schedule**: Reduce LR when plateauing\n\n**For language models**: Accuracy = 100/perplexity, so 50% = perplexity 2.0 (good!)",
    }

    response = None
    for key, value in responses.items():
        if key in message:
            response = value
            break

    if not response:
        if "help" in message or "what" in message or "how" in message:
            response = """**I can help you with:**

- **Layers**: conv, lstm, embedding, linear, dropout, batchnorm
- **Training**: learning rate, epochs, batch size, optimizer
- **Patterns**: CNN architecture, language models
- **Problems**: overfitting, underfitting, improving accuracy

**Just ask about any of these topics!**

Examples:

- "What learning rate should I use?"
- "How do I fix overfitting?"
- "Explain dropout"
- "How to improve accuracy?" """
        elif dataset_type == "text":
            response = """**For text/language models**, I recommend:

- **Architecture**: Embedding → LSTM (2 layers) → Dropout → Linear
- **Embedding dim**: 128-256
- **Hidden size**: 256-512
- **Dropout**: 0.3
- **Epochs**: 30-50 for good results
- **Learning rate**: 0.001-0.002

Ask me about specific layers or hyperparameters!"""
        else:
            response = """**For image classification**, I recommend:

- **Architecture**: Conv → BatchNorm → ReLU → Pool (repeat) → Flatten → Linear
- **Filters**: Start with 32, double each block (32→64→128)
- **Dropout**: 0.2-0.3 before final layer
- **Epochs**: 20-30
- **Learning rate**: 0.001 with Adam

Ask me about specific layers or hyperparameters!"""

    return {"response": response}
