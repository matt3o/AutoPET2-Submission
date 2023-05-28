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
    streamHandler.setLevel(logging.INFO)
    logger.addHandler(streamHandler)
    
    if args is not None and args.log_to_file:
    # Add the file handler
        log_file_path = "{}/log.txt".format(args.output)
        fileHandler = logging.FileHandler(log_file_path)
        fileHandler.setFormatter(formatter)
        logger.addHandler(fileHandler)
        logger.info("Logging all the data to '{}'".format(log_file_path))
    else:
        logger.info("Logging only to the console")

    # Set logging level for external libraries
    for _ in ("ignite.engine.engine.SupervisedTrainer", "ignite.engine.engine.SupervisedEvaluator"):
        l = logging.getLogger(_)
        l.handlers.clear()
        l.setLevel(logging.INFO)
        l.addHandler(streamHandler)
        if args is not None and args.log_to_file:
            l.addHandler(fileHandler)
            

def get_logger():
    global logger
    if logger == None:
        raise UserWarning("Logger not initialized")
    else:
        return logger

