import gc
import multiprocessing as mp
import threading
import time
from collections import OrderedDict
from collections.abc import Generator
from concurrent.futures import Future, ProcessPoolExecutor
from typing import Any

import numpy as np
from hydra.utils import get_class
from omegaconf import DictConfig, OmegaConf


def _load_dataset_worker(datasets_cfg_dict: dict, data_name: str) -> Any:
    """Worker function for loading dataset in separate process"""
    try:
        datasets_cfg = OmegaConf.create(datasets_cfg_dict)

        dataset_cls = get_class(datasets_cfg._target_)
        data = dataset_cls(**datasets_cfg.cfgs, data_name=data_name)

        return data
    except Exception as e:
        print(f"Error loading dataset {data_name} in worker process: {e}")
        return None


class GraphDatasetLoader:
    """
    On-demand data loader for multiple datasets with LRU caching and async loading.
    """

    def __init__(
        self,
        datasets_cfg: DictConfig,
        data_names: list[str],
        shuffle: bool = True,
        max_datasets_in_memory: int = 1,
        data_loading_workers: int = 2,
    ) -> None:
        """
        Initialize the data loader.
        Args:
            datasets_cfg (DictConfig): Configuration for datasets.
            data_names (list[str]): List of dataset names to load.
            shuffle (bool): Whether to shuffle the datasets.
            max_datasets_in_memory (int): Maximum number of datasets to keep in memory.
            data_loading_workers (int): Number of workers for async loading.
        """
        self.datasets_cfg = datasets_cfg
        self.data_names = data_names
        self.shuffle = shuffle
        self.max_datasets_in_memory = max_datasets_in_memory
        self.data_loading_workers = data_loading_workers

        # Use OrderedDict to maintain LRU cache
        self.loaded_datasets: OrderedDict[str, Any] = OrderedDict()

        # Multiprocessing components
        self.executor: None | ProcessPoolExecutor = None
        self.loading_futures: dict[str, Future] = {}  # Track ongoing loading tasks
        self.loading_lock = (
            threading.Lock()
        )  # Protect concurrent access to loading_futures

        # Initialize process pool only if max_workers > 0
        if self.data_loading_workers > 0:
            self._init_process_pool()
        else:
            self.executor = None

    def _init_process_pool(self) -> None:
        """Initialize the process pool executor"""
        if self.data_loading_workers <= 0:
            self.executor = None
            return

        # Use spawn method to avoid issues with CUDA contexts
        mp_context = mp.get_context("spawn")
        self.executor = ProcessPoolExecutor(
            max_workers=self.data_loading_workers, mp_context=mp_context
        )

    def __del__(self) -> None:
        """Cleanup when object is destroyed"""
        self.shutdown()

    def shutdown(self) -> None:
        """Shutdown the process pool executor"""
        if hasattr(self, "executor") and self.executor:
            # Cancel all pending futures
            with self.loading_lock:
                for future in self.loading_futures.values():
                    future.cancel()
                self.loading_futures.clear()

            self.executor.shutdown(wait=False)
            self.executor = None

    def set_epoch(self, epoch: int) -> None:
        np.random.seed(epoch)

    def _manage_memory(self) -> None:
        """Manage memory to ensure not exceeding maximum dataset count"""
        while len(self.loaded_datasets) >= self.max_datasets_in_memory:
            # Remove the oldest dataset
            oldest_name, oldest_dataset = self.loaded_datasets.popitem(last=False)

            del oldest_dataset
            gc.collect()

    def _start_async_loading(self, data_names: list[str]) -> None:
        """Start async loading for multiple datasets (up to max_workers)"""
        # Skip async loading if max_workers is 0
        if self.data_loading_workers <= 0 or not self.executor:
            return

        with self.loading_lock:
            # Calculate how many we can start loading
            available_workers = self.data_loading_workers - len(self.loading_futures)
            datasets_to_load = []

            for data_name in data_names[:available_workers]:
                # Skip if already loaded or currently being loaded
                if (
                    data_name in self.loaded_datasets
                    or data_name in self.loading_futures
                ):
                    continue
                datasets_to_load.append(data_name)

            # Start async loading for each dataset
            for data_name in datasets_to_load:
                try:
                    # Convert DictConfig to dict for serialization
                    datasets_cfg_dict = OmegaConf.to_container(
                        self.datasets_cfg, resolve=True
                    )

                    future = self.executor.submit(
                        _load_dataset_worker,
                        datasets_cfg_dict,
                        data_name,
                    )
                    self.loading_futures[data_name] = future
                except Exception as e:
                    print(f"Failed to start async loading for {data_name}: {e}")

    def _get_next_datasets_to_prefetch(
        self, current_index: int, data_name_list: list[str]
    ) -> list[str]:
        """Get the next max_workers datasets to prefetch"""
        start_idx = current_index + 1
        end_idx = min(start_idx + self.data_loading_workers, len(data_name_list))
        return data_name_list[start_idx:end_idx]

    def _wait_for_dataset(self, data_name: str, timeout: float = 30.0) -> dict | None:
        """Wait for async loading to complete and return dataset"""
        with self.loading_lock:
            if data_name not in self.loading_futures:
                return None

            future = self.loading_futures.pop(data_name)

        try:
            # Wait for the loading to complete
            dataset = future.result(timeout=timeout)
            return dataset
        except Exception as e:
            print(f"Error loading dataset {data_name}: {e}")
            return None

    def _cleanup_completed_futures(self) -> None:
        """Clean up completed futures and store results"""
        if not self.executor:
            return

        with self.loading_lock:
            completed_names = []
            for name, future in self.loading_futures.items():
                if future.done():
                    try:
                        # Try to get the result to handle any exceptions
                        dataset = future.result()
                        if dataset is not None:
                            # Only store if we have space and don't already have it
                            if name not in self.loaded_datasets:
                                self._manage_memory()
                                self.loaded_datasets[name] = dataset
                        completed_names.append(name)
                    except Exception as e:
                        print(f"Error in background loading of {name}: {e}")
                        completed_names.append(name)

            # Remove completed futures
            for name in completed_names:
                self.loading_futures.pop(name, None)

    def _preload_datasets(self, data_name_list: list[str]) -> None:
        """Preload datasets up to memory limit"""
        # Preload first few datasets synchronously
        sync_preload_count = min(self.max_datasets_in_memory, len(data_name_list))

        for i in range(sync_preload_count):
            data_name = data_name_list[i]
            if data_name not in self.loaded_datasets:
                self._manage_memory()
                dataset = _load_dataset_worker(
                    OmegaConf.to_container(self.datasets_cfg, resolve=True), data_name
                )
                self.loaded_datasets[data_name] = dataset

        # Start async loading for next max_workers datasets only if max_workers > 0
        if self.data_loading_workers > 0 and sync_preload_count < len(data_name_list):
            async_start_idx = sync_preload_count
            async_datasets = self._get_next_datasets_to_prefetch(
                async_start_idx - 1, data_name_list
            )
            if async_datasets:
                self._start_async_loading(async_datasets)

    def _get_dataset(self, data_name: str) -> Any:
        """Get dataset, load if not in memory"""
        # Clean up any completed background loading first
        self._cleanup_completed_futures()

        if data_name in self.loaded_datasets:
            # Move to most recently used position (LRU update)
            dataset = self.loaded_datasets.pop(data_name)
            self.loaded_datasets[data_name] = dataset
            return dataset
        else:
            # Check if it's being loaded asynchronously
            if data_name in self.loading_futures:
                dataset = self._wait_for_dataset(data_name)
                if dataset is not None:
                    self._manage_memory()
                    self.loaded_datasets[data_name] = dataset
                    return dataset

            # Dataset not in memory and not being loaded, load synchronously
            self._manage_memory()
            dataset = _load_dataset_worker(
                OmegaConf.to_container(self.datasets_cfg, resolve=True),
                data_name,
            )
            self.loaded_datasets[data_name] = dataset
            return dataset

    def __iter__(self) -> Generator[dict[str, Any], None, None]:
        data_name_list = self.data_names.copy()
        if self.shuffle:
            np.random.shuffle(data_name_list)

        # Preload datasets
        self._preload_datasets(data_name_list)

        for i, data_name in enumerate(data_name_list):
            # Start async loading for next max_workers datasets only if max_workers > 0
            if self.data_loading_workers > 0:
                next_datasets = self._get_next_datasets_to_prefetch(i, data_name_list)
                if next_datasets:
                    self._start_async_loading(next_datasets)

            # Get current dataset
            dataset = self._get_dataset(data_name)

            yield {"data_name": data_name, "data": dataset}

    def clear_cache(self) -> None:
        """Clear all cached datasets and cancel pending loads"""
        # Cancel all pending async loads
        with self.loading_lock:
            for future in self.loading_futures.values():
                future.cancel()
            self.loading_futures.clear()

        self.loaded_datasets.clear()
        gc.collect()

    def get_memory_info(self) -> dict:
        """Get current memory usage information"""
        with self.loading_lock:
            loading_count = len(self.loading_futures)
            loading_names = list(self.loading_futures.keys())

        return {
            "loaded_datasets_count": len(self.loaded_datasets),
            "max_datasets_in_memory": self.max_datasets_in_memory,
            "loaded_dataset_names": list(self.loaded_datasets.keys()),
            "async_loading_count": loading_count,
            "async_loading_names": loading_names,
            "max_workers": self.data_loading_workers,
        }

    def wait_for_all_loading(self, timeout: float = 60.0) -> None:
        """Wait for all async loading to complete"""
        if not self.executor:
            return

        start_time = time.time()
        while True:
            with self.loading_lock:
                if not self.loading_futures:
                    break

                # Check for completed futures
                self._cleanup_completed_futures()

            # Check timeout
            if time.time() - start_time > timeout:
                print("Warning: Timeout waiting for async loading to complete")
                break

            time.sleep(0.1)
