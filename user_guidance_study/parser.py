import argparse
import sys
import os
import logging
import pathlib
import uuid
import torch

from utils.logger import setup_loggers, get_logger
from utils.helper import print_gpu_usage, get_gpu_usage, get_actual_cuda_index_of_device, get_git_information, gpu_usage

from utils.transforms import (
    ClickGenerationStrategy,
    StoppingCriterion,
)


def parse_args():
    parser = argparse.ArgumentParser()

    # Data
    parser.add_argument("-i", "--input", default="/cvhci/data/AutoPET/AutoPET/")
    parser.add_argument("-o", "--output", default="/cvhci/temp/mhadlich/output")
    parser.add_argument("-d", "--data", default="None")
    # a subdirectory is created below cache_dir for every run
    parser.add_argument("-c", "--cache_dir", type=str, default='/cvhci/temp/mhadlich/cache')
    parser.add_argument("-ta", "--throw_away_cache", default=False, action='store_true')
    parser.add_argument("-x", "--split", type=float, default=0.8)
    parser.add_argument("-t", "--limit", type=int, default=0, help='Limit the amount of training/validation samples')
    parser.add_argument("--dataset", default="AutoPET") #MSD_Spleen

    # Configuration
    parser.add_argument("-s", "--seed", type=int, default=36)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--no_log", default=False, action='store_true')
    parser.add_argument("--dont_check_output_dir", default=False, action='store_true')
    parser.add_argument("--debug", default=False, action='store_true')


    # Model
    parser.add_argument("-n", "--network", default="dynunet", choices=["dynunet", "smalldynunet"])
    parser.add_argument("-in", "--inferer", default="SimpleInferer", choices=["SimpleInferer", "SlidingWindowInferer"])
    parser.add_argument("--sw_roi_size", default="(128,128,128)", action='store')
    # crop_size multiples of sliding window size (128,128,128) with overlap 0.25 (default): 128, 224, 320, 416, 512
    parser.add_argument("--train_crop_size", default="(224,224,224)", action='store')
    parser.add_argument("--val_crop_size", default="None", action='store')
    # 1 on 24 Gb, 8 on 50 Gb,
    parser.add_argument("--train_sw_batch_size", type=int, default=8)
    parser.add_argument("--val_sw_batch_size", type=int, default=1)
    

    # Training
    parser.add_argument("-a", "--amp", default=False, action='store_true')
    parser.add_argument("--num_workers", type=int, default=1)
    parser.add_argument("-e", "--epochs", type=int, default=100)
    # If learning rate is set to 0.001, the DiceCELoss will produce Nans very quickly
    parser.add_argument("-lr", "--learning_rate", type=float, default=0.0001)
    parser.add_argument("--optimizer", default="Adam", choices=["Adam", "Novograd"])
    parser.add_argument("--scheduler", default="MultiStepLR", choices=["MultiStepLR", "PolynomialLR", "CosineAnnealingLR"])
    parser.add_argument("--resume_from", type=str, default='None')

    # Logging
    parser.add_argument("-f", "--val_freq", type=int, default=1) # Epoch Level
    parser.add_argument("--save_interval", type=int, default=3)
    parser.add_argument("--export", default=False, action='store_true')
    parser.add_argument("--eval_only", default=False, action='store_true')
    parser.add_argument("--save_nifti", default=False, action='store_true')

    # Interactions
    parser.add_argument("-it", "--max_train_interactions", type=int, default=10)
    parser.add_argument("-iv", "--max_val_interactions", type=int, default=10)
    parser.add_argument("-dpt", "--deepgrow_probability_train", type=float, default=1.0)
    parser.add_argument("-dpv", "--deepgrow_probability_val", type=float, default=1.0)


    # Guidance Signal Hyperparameters
    parser.add_argument("--sigma", type=int, default=1)
    parser.add_argument("--disks", default=False, action='store_true')
    parser.add_argument("--edt", default=False, action='store_true')
    parser.add_argument("--gdt", default=False, action='store_true')
    parser.add_argument("--gdt_th", type=float, default=10)
    parser.add_argument("--exp_geos", default=False, action='store_true')
    parser.add_argument("--conv1d", default=False, action='store_true')
    parser.add_argument("--conv1s", default=False, action='store_true')
    parser.add_argument("--adaptive_sigma", default=False, action='store_true')

    # Guidance Signal Click Generation - for details see the mappings below
    parser.add_argument("-tcg", "--train_click_generation", type=int, default=2, choices=[1,2])
    parser.add_argument("-vcg", "--val_click_generation", type=int, default=1, choices=[1,2])
    parser.add_argument("-tcgsc", "--train_click_generation_stopping_criterion", type=int, default=1, choices=[1,2,3,4,5])
    # Usually this setting should be at 1, so max_iter
    parser.add_argument("-vcgsc", "--val_click_generation_stopping_criterion", type=int, default=1, choices=[1,2,3,4,5])

    # Set up additional information concerning the environment and the way the script was called
    args = parser.parse_args()
    args.caller_args = sys.argv
    args.env = os.environ
    args.git = get_git_information()

    device = torch.device(f"cuda:{args.gpu}")

    # For single label using one of the Medical Segmentation Decathlon
    args.labels = {'spleen': 1,
                   'background': 0
                   }    

    if not args.dont_check_output_dir and os.path.isdir(args.output):
        raise UserWarning(f"output path {args.output} already exists. Please choose another path..")
    if not os.path.exists(args.output):
        pathlib.Path(args.output).mkdir(parents=True)
    
    if args.debug:
        loglevel = logging.DEBUG
    else:
        loglevel = logging.INFO
    if args.no_log:
        log_folder_path = None
    else:
        log_folder_path = args.output
    setup_loggers(loglevel, log_folder_path)
    logger = get_logger()

    if args.throw_away_cache:
        args.cache_dir = f"{args.cache_dir}/{uuid.uuid4()}"
    else:
        args.cache_dir = f"{args.cache_dir}"

    if not os.path.exists(args.cache_dir):
        pathlib.Path(args.cache_dir).mkdir(parents=True)
    
    if args.data == "None":
        args.data = f"{args.output}/data"
        logger.info(f"--data was None, so that {args.data} was selected instead")
        
    if not os.path.exists(args.data):
        pathlib.Path(args.data).mkdir(parents=True)

    # Training only, so done on the patch of size train_crop_size
    train_click_generation_mapping = {
        1: ClickGenerationStrategy.GLOBAL_NON_CORRECTIVE, #"non-corrective",
        2: ClickGenerationStrategy.GLOBAL_CORRECTIVE, #"corrective",
    }
    args.train_click_generation = train_click_generation_mapping[args.train_click_generation]
    # Validation, so everything is done on the full volume
    val_click_generation_mapping = {
        
        1: ClickGenerationStrategy.GLOBAL_CORRECTIVE, #"patch-based corrective",
        # Sample directly from the global error
        2: ClickGenerationStrategy.PATCH_BASED_CORRECTIVE, # "global corrective",
    }
    args.val_click_generation = val_click_generation_mapping[args.val_click_generation]
    
    args.train_click_generation_stopping_criterion = StoppingCriterion(args.train_click_generation_stopping_criterion)
    args.val_click_generation_stopping_criterion = StoppingCriterion(args.val_click_generation_stopping_criterion)

    # NOTE Added for backwards compatibility with DeepGrow. Manual override of some settings, thus need to accept it
    if args.deepgrow_probability_val != 1 or args.deepgrow_probability_val != 1:
        # raise UserWarning("For DeepGrow to work you have to set args.val_click_generation_stopping_criterion to 5!")
        logger.warning("############## DeepGrow mode activated ###################")
        logger.warning("args.train_click_generation, args.val_click_generation, args.train_click_generation_stopping_criterion and args.val_click_generation_stopping_criterion will be overwritten by this option")
        logger.warning("##########################################################")
        # logger.info("To reproduce ")
        # accept = input("please type y to agree: ")
        # if not accept.startswith("y"):
        #     logger.warning("Not accepted. Now leaving the program")
        #     exit(0)
        # else:
        args.train_click_generation_stopping_criterion = StoppingCriterion.DEEPGROW_PROBABILITY
        args.val_click_generation_stopping_criterion = StoppingCriterion.DEEPGROW_PROBABILITY
        args.train_click_generation = ClickGenerationStrategy.DEEPGROW_GLOBAL_CORRECTIVE
        args.val_click_generation = ClickGenerationStrategy.DEEPGROW_GLOBAL_CORRECTIVE


    args.real_cuda_device = get_actual_cuda_index_of_device(torch.device(f"cuda:{args.gpu}"))

    logger.info(f"CPU Count: {os.cpu_count()}")
    logger.info(f"Num threads: {torch.get_num_threads()}")

    args.cwd = os.getcwd()

    nv_total = gpu_usage(device, used_memory_only=False)[3]
    if nv_total < 25000:
        args.gpu_size = "small"
    elif nv_total < 55000:
        args.gpu_size = "medium"
    else:
        args.gpu_size = "large"
    logger.info(f"Selected GPU size: {args.gpu_size}, since GPU Memory: {nv_total} MB")

    # Init the Inferer
    args.sw_roi_size = eval(args.sw_roi_size)
    assert len(args.sw_roi_size) == 3

    if args.val_crop_size == "None":
        args.val_crop_size = None
    else:
        args.val_crop_size = eval(args.val_crop_size)
        assert len(args.val_crop_size) == 3

    if args.train_crop_size == "None":
        args.train_crop_size = None
    else:
        args.train_crop_size = eval(args.train_crop_size)
        assert len(args.train_crop_size) == 3

    # verify both have a valid size (for Unet with seven layers)
    if args.network == "dynunet" and args.inferer == "SimpleInferer":
        for size in args.train_crop_size:
            assert (size % 64) == 0
        for size in args.val_crop_size:
            assert (size % 64) == 0

    return args, logger
