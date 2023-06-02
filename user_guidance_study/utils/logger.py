import logging

logger = None

def setup_loggers(args=None):
    global logger
    logger = logging.getLogger("interactive_segmentation")
    if (logger.hasHandlers()):
        logger.handlers.clear()
    logger.propagate = False
    logger.setLevel(logging.DEBUG)
    # Add the stream handler
    streamHandler = logging.StreamHandler()
    # (%(name)s)
    formatter = logging.Formatter(fmt="[%(asctime)s.%(msecs)03d][%(levelname)s] %(funcName)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    streamHandler.setFormatter(formatter)
    if args.debug:
        streamHandler.setLevel(logging.DEBUG)
    else:
        streamHandler.setLevel(logging.INFO)
    logger.addHandler(streamHandler)
    
    if args is not None and not args.no_log:
    # Add the file handler
        log_file_path = f"{args.output}/log.txt"
        fileHandler = logging.FileHandler(log_file_path)
        fileHandler.setFormatter(formatter)
        if args.debug:
            fileHandler.setLevel(logging.DEBUG)
        else:
            fileHandler.setLevel(logging.INFO)
        logger.addHandler(fileHandler)
        logger.info(f"Logging all the data to '{log_file_path}'")
    else:
        logger.info("Logging only to the console")

    # Set logging level for external libraries
    for _ in ("ignite.engine.engine.SupervisedTrainer", "ignite.engine.engine.SupervisedEvaluator"):
        l = logging.getLogger(_)
        l.handlers.clear()
        if args.debug:
            l.setLevel(logging.DEBUG)
        else:
            l.setLevel(logging.INFO)
        l.addHandler(streamHandler)
        if args is not None and not args.no_log:
            l.addHandler(fileHandler)
            

def get_logger():
    global logger
    if logger == None:
        raise UserWarning("Logger not initialized")
    else:
        return logger

