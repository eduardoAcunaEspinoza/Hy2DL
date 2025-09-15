import logging
from pathlib import Path

from tqdm import tqdm


class TqdmHandler(logging.StreamHandler):
    """A logging handler that plays nicely with tqdm progress bars.

    This handler is a subclass of ``logging.StreamHandler`` that overrides
    the ``emit`` method to use ``tqdm.write()`` instead of writing directly
    to stdout or stderr. This ensures that log messages do not interfere
    with tqdm progress bars, and are displayed cleanly above them.
    """

    def emit(self, record: logging.LogRecord) -> None:
        """Emit a log record using tqdm.write().

        Parameters
        ----------
        record : logging.LogRecord
            The log record to be emitted. The record is first formatted
            using the handler's formatter before being passed to tqdm.write().
        """
        msg = self.format(record)
        tqdm.write(msg)


def get_logger(path_save_folder: Path, name: str = "my_logger") -> logging.Logger:
    """Create (or retrieve) a configured logger that logs to both file and console.

    This function returns a named ``logging.Logger`` instance that writes
    logs to a file (``run.log`` in the given folder) and to the console
    using a ``TqdmHandler`` so that output plays nicely with tqdm progress bars.

    If a logger with the given name already exists and has handlers
    configured, it is returned as-is to prevent duplicate log messages.

    Parameters
    ----------
    path_save_folder : Path
        Directory where the log file (``run.log``) will be created.
    name : str, optional
        The name of the logger. Using a unique name allows multiple
        independent loggers to coexist without interfering with each other.
        Defaults to "my_logger".

    Returns
    -------
    logging.Logger
        A configured logger instance with file and console handlers attached.

    """
    logger = logging.getLogger(name)

    if logger.handlers:  # prevent adding duplicate handlers
        return logger

    logger.propagate = False  # prevent double logging via root logger

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

    # Attach handlers
    logger.setLevel(logging.INFO)
    logger.addHandler(fh)
    logger.addHandler(ch)

    return logger
