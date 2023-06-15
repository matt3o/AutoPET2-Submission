import logging

logger = None

def setup_loggers(loglevel=logging.INFO, log_file_folder=None,):
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
    streamHandler.setLevel(loglevel)
    logger.addHandler(streamHandler)
    fileHandler = None

    if log_file_folder is not None:
    # Add the file handler
        log_file_path = f"{log_file_folder}/log.txt"
        fileHandler = logging.FileHandler(log_file_path)
        fileHandler.setFormatter(formatter)
        fileHandler.setLevel(loglevel)
        logger.addHandler(fileHandler)
        logger.info(f"Logging all the data to '{log_file_path}'")
    else:
        logger.info("Logging only to the console")

    # Set logging level for external libraries
    for _ in ("ignite.engine.engine.SupervisedTrainer", "ignite.engine.engine.SupervisedEvaluator"):
        l = logging.getLogger(_)
        if l.hasHandlers():
            l.handlers.clear()
        l.propagate = False
        l.addHandler(streamHandler)
        if fileHandler is not None:
            l.addHandler(fileHandler)
            

def get_logger():
    global logger
    if logger == None:
        raise UserWarning("Logger not initialized")
    else:
        return logger

