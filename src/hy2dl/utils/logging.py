import logging
from pathlib import Path

from tqdm import tqdm


class TqdmHandler(logging.StreamHandler):
    """Stream handler for tqdm.

    This class is a subclass of a `StreamHandler` in the `logging` module
    specifically designed for tqdm.
    """

    def emit(self, record: str) -> None:
        """Emit the log record.

        Uses tqdm.write() to write to the console.

        Parameters
        ----------
        record : str
            The log record to emit.

        """
        msg = self.format(record)
        tqdm.write(msg)


def get_logger(path_save_folder: Path) -> logging.Logger:
    """Get logger instance.

    Parameters
    ----------
    path_save_folder : Path
        Path to the folder where the log file will be saved.

    """
    # path to save the log file
    path_to_log = path_save_folder / "run.log"

    # Format
    msg_format = logging.Formatter("%(asctime)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S")

    # File handler
    fh = logging.FileHandler(path_to_log)
    fh.setLevel(logging.INFO)
    fh.setFormatter(msg_format)

    # Standard output
    ch = TqdmHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(msg_format)

    # Logger instance
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.addHandler(fh)
    logger.addHandler(ch)

    return logger
