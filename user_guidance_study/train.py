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

from __future__ import annotations

import os
import pathlib
import resource
import sys
import time
from parser import parse_args
from pickle import dump

import pandas as pd
import torch
from ignite.engine import Events

from monai.engines.utils import IterationEvents

from monai.utils.profiling import ProfileHandler, WorkflowProfiler
from tensorboard_logger import init_tensorboard_logger
from utils.helper import (
    GPU_Thread,
    TerminationHandler,
    get_gpu_usage,
    handle_exception,
)
from api import get_trainer

# Various settings

logger = None
output_dir = None

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

tmpdir = "/local/work/mhadlich/tmp"
if os.environ.get("SLURM_JOB_ID") is not None:
    os.environ["TMPDIR"] = tmpdir
    if not os.path.exists(tmpdir):
        pathlib.Path(tmpdir).mkdir(parents=True)

rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (8 * 8192, rlimit[1]))


def oom_observer(device, alloc, device_alloc, device_free):
    if device is not None and logger is not None:
        logger.critical(torch.cuda.memory_summary(device))
    # snapshot right after an OOM happened
    print("saving allocated state during OOM")
    print("Tips: \nReduce sw_batch_size if there is an OOM (maybe even roi_size)")
    snapshot = torch.cuda.memory._snapshot()
    dump(snapshot, open(f"{output_dir}/oom_snapshot.pickle", "wb"))
    # logger.critical(snapshot)
    torch.cuda.memory._save_memory_usage(
        filename=f"{output_dir}/memory.svg", snapshot=snapshot
    )
    torch.cuda.memory._save_segment_usage(
        filename=f"{output_dir}/segments.svg", snapshot=snapshot
    )


sys.excepthook = handle_exception


def run(args):
    for arg in vars(args):
        logger.info("USING:: {} = {}".format(arg, getattr(args, arg)))
    print("")
    device = torch.device(f"cuda:{args.gpu}")

    gpu_thread = GPU_Thread(1, "Track_GPU_Usage", f"{args.output}/usage.csv", device)
    logger.info(f"Logging GPU usage to {args.output}/usage.csv")

    try:
        wp = WorkflowProfiler()
        trainer, evaluator, all_train_metrics, all_val_metrics = get_trainer(args)

        tb_logger = init_tensorboard_logger(
            trainer,
            evaluator,
            trainer.optimizer,
            all_train_metrics,
            all_val_metrics,
            network=trainer.network,
            output_dir=args.output,
        )

        gpu_thread.start()
        terminator = TerminationHandler(args, tb_logger, wp, gpu_thread)

        with tb_logger:
            with wp:
                start_time = time.time()
                for t, name in [(trainer, "trainer"), (evaluator, "evaluator")]:
                    for event in [
                        ["Epoch", wp, Events.EPOCH_STARTED, Events.EPOCH_COMPLETED],
                        [
                            "Iteration",
                            wp,
                            Events.ITERATION_STARTED,
                            Events.ITERATION_COMPLETED,
                        ],
                        [
                            "Batch generation",
                            wp,
                            Events.GET_BATCH_STARTED,
                            Events.GET_BATCH_COMPLETED,
                        ],
                        [
                            "Inner Iteration",
                            wp,
                            IterationEvents.INNER_ITERATION_STARTED,
                            IterationEvents.INNER_ITERATION_COMPLETED,
                        ],
                        ["Whole run", wp, Events.STARTED, Events.COMPLETED],
                    ]:
                        event[0] = f"{name}: {event[0]}"
                        # print(*event)
                        ProfileHandler(*event).attach(t)

                try:
                    if not args.eval_only:
                        trainer.run()
                    else:
                        evaluator.run()
                except torch.cuda.OutOfMemoryError:
                    oom_observer(device, None, None, None)
                    logger.critical(
                        get_gpu_usage(device, used_memory_only=False, context="ERROR")
                    )

                except RuntimeError as e:
                    if "cuDNN" in str(e):
                        # Got a cuDNN error
                        pass
                    oom_observer(device, None, None, None)
                    logger.critical(
                        get_gpu_usage(device, used_memory_only=False, context="ERROR")
                    )

                finally:
                    logger.info(
                        get_gpu_usage(device, used_memory_only=False, context="ERROR")
                    )
                    logger.info(
                        "Total Training Time {}".format(time.time() - start_time)
                    )

        if not args.eval_only:
            logger.info("{}:: Saving Final PT Model".format(args.gpu))

            torch.save(
                trainer.network.state_dict(),
                os.path.join(
                    args.output, "pretrained_deepedit_" + args.network + "-final.pt"
                ),
            )

            logger.info("{}:: Saving TorchScript Model".format(args.gpu))
            model_ts = torch.jit.script(trainer.network)
            torch.jit.save(
                model_ts,
                os.path.join(
                    args.output, "pretrained_deepedit_" + args.network + "-final.ts"
                ),
            )
    finally:
        terminator.cleanup()
        terminator.join_threads()
        pd.set_option("display.max_columns", None)
        pd.set_option("display.max_rows", None)
        logger.info(f"\n{wp.get_times_summary_pd()}")


def main():
    global logger
    global output_dir

    torch.set_num_threads(
        int(os.cpu_count() / 3)
    )  # Limit number of threads to 1/3 of resources

    args, logger = parse_args()

    # for OOM debugging
    output_dir = args.output

    run(args)


if __name__ == "__main__":
    main()
