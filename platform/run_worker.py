#!/usr/bin/env python3
"""
Run a training worker.

Usage:
    python run_worker.py                    # Run single worker
    python run_worker.py --id worker-1      # Run with specific ID
    python run_worker.py --count 4          # Run 4 workers in parallel

Workers will poll the job queue and process training jobs.
Multiple workers can run concurrently for parallel training.
"""

import argparse
import logging
import multiprocessing
import signal
import sys
from typing import List

logger = logging.getLogger(__name__)

from db import init_db
from workers import TrainingWorker


def run_single_worker(worker_id: str = None):
    """Run a single worker."""
    init_db()
    worker = TrainingWorker(worker_id=worker_id)
    worker.start()


def run_multiple_workers(count: int):
    """Run multiple workers in parallel processes."""
    init_db()

    processes: List[multiprocessing.Process] = []

    def signal_handler(signum, frame):
        logger.info("Shutting down workers...")
        for p in processes:
            p.terminate()
        for p in processes:
            p.join(timeout=5)
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    logger.info("Starting %d workers...", count)

    for i in range(count):
        worker_id = f"worker-{i+1}"
        p = multiprocessing.Process(
            target=run_single_worker,
            args=(worker_id,),
            name=worker_id
        )
        p.start()
        processes.append(p)
        logger.info("  Started %s (PID: %d)", worker_id, p.pid)

    logger.info("All %d workers running. Press Ctrl+C to stop.", count)

    # Wait for all processes
    for p in processes:
        p.join()


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    )

    parser = argparse.ArgumentParser(description="Run training workers")
    parser.add_argument(
        "--id",
        help="Worker ID (for single worker mode)"
    )
    parser.add_argument(
        "--count",
        type=int,
        default=1,
        help="Number of workers to run (default: 1)"
    )
    args = parser.parse_args()

    if args.count > 1:
        run_multiple_workers(args.count)
    else:
        run_single_worker(args.id)


if __name__ == "__main__":
    main()
