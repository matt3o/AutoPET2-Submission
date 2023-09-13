# Copyright (c) MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Code extension and modification by M.Sc. Zdravko Marinov, Karlsuhe Institute of Techonology #
# zdravko.marinov@kit.edu #
# Further code extension and modification by B.Sc. Matthias Hadlich, Karlsuhe Institute of Techonology #
# matthiashadlich@posteo.de #

from __future__ import annotations

import os
import pathlib
import resource
import sys
import time

import torch
from ignite.engine import Events
from monai.engines.utils import IterationEvents
from monai.utils.profiling import ProfileHandler, WorkflowProfiler

from sw_interactive_segmentation.api import (
    get_trainer,
    oom_observer,
    get_save_dict,
    get_cross_validation_trainers_generator,
)
from sw_interactive_segmentation.utils.argparser import parse_args, setup_environment_and_adapt_args
from sw_interactive_segmentation.utils.tensorboard_logger import init_tensorboard_logger
from sw_interactive_segmentation.utils.helper import GPU_Thread, TerminationHandler, get_gpu_usage, handle_exception
from monai.handlers import (
    CheckpointLoader,
)

logger = None


def run(args):
    for arg in vars(args):
        logger.info("USING:: {} = {}".format(arg, getattr(args, arg)))
    print("")
    device = torch.device(f"cuda:{args.gpu}")

    gpu_thread = GPU_Thread(1, "Track_GPU_Usage", os.path.join(args.output_dir, "usage.csv"), device)
    logger.info(f"Logging GPU usage to {args.output_dir}/usage.csv")

    try:
        gpu_thread.start()
        terminator = TerminationHandler(args, None, None, gpu_thread)

        try:
            start_time = time.time()
            if not args.eval_only:
                for trainer in get_cross_validation_trainers_generator(args, nfolds=5):
                    trainer.run()
            else:
                raise NotImplementedError()

        except torch.cuda.OutOfMemoryError:
            oom_observer(device, None, None, None)
            logger.critical(get_gpu_usage(device, used_memory_only=False, context="ERROR"))
        except RuntimeError as e:
            oom_observer(device, None, None, None)
            logger.critical(get_gpu_usage(device, used_memory_only=False, context="ERROR"))
        finally:
            logger.info("Total Training Time {}".format(time.time() - start_time))
    finally:
        terminator.cleanup()
        terminator.join_threads()


def main():
    global logger

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.deterministic = True

    # Slurm only: Speed up the creation of temporary files
    if os.environ.get("SLURM_JOB_ID") is not None:
        tmpdir = "/local/work/mhadlich/tmp"
        os.environ["TMPDIR"] = tmpdir
        if not os.path.exists(tmpdir):
            pathlib.Path(tmpdir).mkdir(parents=True)

    rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
    resource.setrlimit(resource.RLIMIT_NOFILE, (8 * 8192, rlimit[1]))

    sys.excepthook = handle_exception

    torch.set_num_threads(int(os.cpu_count() / 3))  # Limit number of threads to 1/3 of resources

    args = parse_args()
    args, logger = setup_environment_and_adapt_args(args)

    run(args)


if __name__ == "__main__":
    raise UserWarning("Code has not been finished! Before running please fix it!")

    main()
