"""Shared helpers for SCUD data loading: fsspec utilities, logging."""

import logging

import fsspec
import lightning
import torch


def fsspec_exists(filename: str) -> bool:
    """Check if a file exists using fsspec."""
    fs, _ = fsspec.core.url_to_fs(filename)
    result: bool = fs.exists(filename)
    return result


def fsspec_listdir(dirname: str) -> list[str]:
    """Listdir in manner compatible with fsspec."""
    fs, _ = fsspec.core.url_to_fs(dirname)
    result: list[str] = fs.ls(dirname)
    return result


def fsspec_mkdirs(dirname: str, exist_ok: bool = True) -> None:
    """Mkdirs in manner compatible with fsspec."""
    fs, _ = fsspec.core.url_to_fs(dirname)
    fs.makedirs(dirname, exist_ok=exist_ok)


def print_nans(tensor: torch.Tensor, name: str) -> None:
    if torch.isnan(tensor).any():
        print(name, tensor)


class LoggingContext:
    """Context manager for selective logging."""

    def __init__(
        self,
        logger: logging.Logger,
        level: int | None = None,
        handler: logging.Handler | None = None,
        close: bool = True,
    ):
        self.logger = logger
        self.level = level
        self.handler = handler
        self.close = close

    def __enter__(self) -> None:
        if self.level is not None:
            self.old_level = self.logger.level
            self.logger.setLevel(self.level)
        if self.handler:
            self.logger.addHandler(self.handler)

    def __exit__(self, et, ev, tb) -> None:  # noqa: ANN001
        if self.level is not None:
            self.logger.setLevel(self.old_level)
        if self.handler:
            self.logger.removeHandler(self.handler)
        if self.handler and self.close:
            self.handler.close()


def get_logger(name: str = __name__, level: int = logging.INFO) -> logging.Logger:
    """Initializes multi-GPU-friendly python logger."""
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # this ensures all logging levels get marked with the rank zero decorator
    # otherwise logs would get multiplied for each GPU process in multi-GPU setup
    for lvl in (
        "debug",
        "info",
        "warning",
        "error",
        "exception",
        "fatal",
        "critical",
    ):
        setattr(
            logger,
            lvl,
            lightning.pytorch.utilities.rank_zero_only(getattr(logger, lvl)),
        )

    return logger
