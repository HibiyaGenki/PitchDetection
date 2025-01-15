from logging import (
    getLogger,
    StreamHandler,
    Formatter,
    DEBUG,
    INFO,
    FileHandler,
    Logger,
)


def get_logger(name: str) -> Logger:
    logger = getLogger(name)
    handler = StreamHandler()
    handler.setLevel(DEBUG)
    handler.setFormatter(
        Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    )
    logger.setLevel(DEBUG)
    logger.addHandler(handler)
    return logger


def set_logger(logger: Logger, log_path: str) -> None:
    file_handler = FileHandler(log_path)
    file_handler.setLevel(DEBUG)
    file_handler.setFormatter(
        Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    )
    logger.addHandler(file_handler)
