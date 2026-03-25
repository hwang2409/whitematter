"""Chat tool definitions and executors for Claude tool use."""
import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor
from typing import Any

logger = logging.getLogger(__name__)

CHAT_TOOLS = [
    {
        "name": "search_datasets",
        "description": "Search HuggingFace Hub for ML datasets. Use when user asks to find/search for datasets.",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search query"},
                "limit": {
                    "type": "integer",
                    "description": "Max results (1-10, default 5)",
                },
            },
            "required": ["query"],
        },
    },
    {
        "name": "import_dataset",
        "description": "Import a HuggingFace dataset into the system for training. Use after user picks a dataset.",
        "input_schema": {
            "type": "object",
            "properties": {
                "dataset_id": {
                    "type": "string",
                    "description": "HuggingFace dataset ID (e.g. 'mnist')",
                },
                "config": {
                    "type": "string",
                    "description": "Dataset config/subset if applicable",
                },
            },
            "required": ["dataset_id"],
        },
    },
]


def execute_search_datasets(query: str, limit: int = 5) -> dict[str, Any]:
    """Search HuggingFace Hub for datasets matching query."""
    limit = max(1, min(10, limit))
    try:
        from huggingface_hub import HfApi

        api = HfApi()
        results = list(
            api.list_datasets(search=query, sort="downloads", direction=-1, limit=limit)
        )
        datasets = []
        for ds in results:
            datasets.append(
                {
                    "id": ds.id,
                    "description": (getattr(ds, "description", None) or "")[:150],
                    "downloads": getattr(ds, "downloads", 0),
                    "tags": (getattr(ds, "tags", None) or [])[:5],
                }
            )
        return {"found": len(datasets), "datasets": datasets}
    except Exception as e:
        logger.exception("HuggingFace search failed")
        return {"found": 0, "datasets": [], "error": str(e)}


def execute_import_dataset(
    dataset_id: str, config: str | None = None
) -> dict[str, Any]:
    """Import a HuggingFace dataset using the existing import pipeline."""
    try:
        from services.huggingface_import import import_from_huggingface

        result = import_from_huggingface(
            dataset_id=dataset_id,
            config=config,
        )
        return {
            "success": True,
            "dataset_id": result.get("id"),
            "name": result.get("name"),
            "data_type": result.get("data_type"),
            "total_samples": result.get("total_samples"),
            "num_classes": result.get("num_classes"),
            "input_shape": result.get("input_shape"),
        }
    except Exception as e:
        logger.exception("HuggingFace import failed")
        return {"success": False, "error": str(e)}


def execute_tool(name: str, input_data: dict[str, Any]) -> dict[str, Any]:
    """Dispatch a tool call to the appropriate executor."""
    if name == "search_datasets":
        return execute_search_datasets(
            query=input_data["query"],
            limit=input_data.get("limit", 5),
        )
    elif name == "import_dataset":
        return execute_import_dataset(
            dataset_id=input_data["dataset_id"],
            config=input_data.get("config"),
        )
    else:
        return {"error": f"Unknown tool: {name}"}


_tool_executor = ThreadPoolExecutor(max_workers=2)


async def execute_tool_async(name: str, input_data: dict[str, Any]) -> dict[str, Any]:
    """Run execute_tool in a thread so it doesn't block the async event loop."""
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(
        _tool_executor, execute_tool, name, input_data
    )


def summarize_tool_result(name: str, result: dict[str, Any]) -> str:
    """Return a short human-readable summary of a tool result."""
    if name == "search_datasets":
        return f"Found {result.get('found', 0)} datasets"
    elif name == "import_dataset":
        if result.get("success"):
            return f"Imported {result.get('name', 'dataset')} ({result.get('total_samples', '?')} samples)"
        return f"Import failed: {result.get('error', 'unknown error')}"
    return "Tool executed"
