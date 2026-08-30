# Copyright (c) 2025 Valentin Boussot
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0


"""Runtime helpers: the workflow environment, logging, and the distributed runtime."""

from konfai.utils import State as State
from konfai.utils.runtime.distributed import _T as _T
from konfai.utils.runtime.distributed import DistributedObject as DistributedObject
from konfai.utils.runtime.distributed import _cpu_budget_applied as _cpu_budget_applied
from konfai.utils.runtime.distributed import _forget_rank_pool as _forget_rank_pool
from konfai.utils.runtime.distributed import _rank_pool as _rank_pool
from konfai.utils.runtime.distributed import _rank_pool_lock as _rank_pool_lock
from konfai.utils.runtime.distributed import _rank_pool_share as _rank_pool_share
from konfai.utils.runtime.distributed import _runs_inline as _runs_inline
from konfai.utils.runtime.distributed import apply_cpu_thread_budget as apply_cpu_thread_budget
from konfai.utils.runtime.distributed import cleanup as cleanup
from konfai.utils.runtime.distributed import execute_distributed_object as execute_distributed_object
from konfai.utils.runtime.distributed import find_free_port as find_free_port
from konfai.utils.runtime.distributed import map_over_rank_pool as map_over_rank_pool
from konfai.utils.runtime.distributed import pin_gloo_to_loopback as pin_gloo_to_loopback
from konfai.utils.runtime.distributed import preserved_rng as preserved_rng
from konfai.utils.runtime.distributed import rank_cpu_share as rank_cpu_share
from konfai.utils.runtime.distributed import rank_pool as rank_pool
from konfai.utils.runtime.distributed import run_distributed_app as run_distributed_app
from konfai.utils.runtime.distributed import seed_all as seed_all
from konfai.utils.runtime.distributed import setup_gpu as setup_gpu
from konfai.utils.runtime.distributed import synchronize_data as synchronize_data
from konfai.utils.runtime.environment import ClusterKwargs as ClusterKwargs
from konfai.utils.runtime.environment import NeedDevice as NeedDevice
from konfai.utils.runtime.environment import _materialized_config as _materialized_config
from konfai.utils.runtime.environment import clear_directory_except_logs as clear_directory_except_logs
from konfai.utils.runtime.environment import configure_workflow_environment as configure_workflow_environment
from konfai.utils.runtime.environment import confirm_overwrite_or_raise as confirm_overwrite_or_raise
from konfai.utils.runtime.environment import description as description
from konfai.utils.runtime.environment import get_cpu_info as get_cpu_info
from konfai.utils.runtime.environment import get_device as get_device
from konfai.utils.runtime.environment import get_gpu_memory as get_gpu_memory
from konfai.utils.runtime.environment import get_memory as get_memory
from konfai.utils.runtime.environment import get_memory_info as get_memory_info
from konfai.utils.runtime.environment import gpu_info as gpu_info
from konfai.utils.runtime.environment import is_interactive_session as is_interactive_session
from konfai.utils.runtime.environment import memory_forecast as memory_forecast
from konfai.utils.runtime.environment import safe_torch_load as safe_torch_load
from konfai.utils.runtime.logging import ANSI_ESCAPE_RE as ANSI_ESCAPE_RE
from konfai.utils.runtime.logging import DataLog as DataLog
from konfai.utils.runtime.logging import Log as Log
from konfai.utils.runtime.logging import MinimalLog as MinimalLog
from konfai.utils.runtime.logging import TensorBoard as TensorBoard
from konfai.utils.runtime.logging import _bar_key as _bar_key
from konfai.utils.runtime.logging import _log_image_format as _log_image_format
from konfai.utils.runtime.logging import _log_images_format as _log_images_format
from konfai.utils.runtime.logging import _log_signal_format as _log_signal_format
from konfai.utils.runtime.logging import _log_video_format as _log_video_format
from konfai.utils.runtime.logging import record as record

__all__ = [
    "ANSI_ESCAPE_RE",
    "ClusterKwargs",
    "DataLog",
    "DistributedObject",
    "Log",
    "MinimalLog",
    "NeedDevice",
    "State",
    "TensorBoard",
    "apply_cpu_thread_budget",
    "cleanup",
    "clear_directory_except_logs",
    "configure_workflow_environment",
    "confirm_overwrite_or_raise",
    "description",
    "execute_distributed_object",
    "find_free_port",
    "get_cpu_info",
    "get_device",
    "get_gpu_memory",
    "get_memory",
    "get_memory_info",
    "gpu_info",
    "is_interactive_session",
    "map_over_rank_pool",
    "memory_forecast",
    "pin_gloo_to_loopback",
    "preserved_rng",
    "rank_cpu_share",
    "rank_pool",
    "record",
    "run_distributed_app",
    "safe_torch_load",
    "seed_all",
    "setup_gpu",
    "synchronize_data",
]
