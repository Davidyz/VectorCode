import logging
import os
import sys
from typing import Callable, Iterable, Self, Sequence

logger = logging.getLogger(name=__name__)

FileFilter = Callable[[str], bool]


class FilterManager:
    def __init__(self, from_filters: Sequence[FileFilter] | None = None) -> None:
        self._filters: list[FileFilter] = []
        if from_filters:
            self._filters.extend(from_filters)

    def add_filter(self, f: FileFilter = lambda x: bool(x)) -> Self:
        self._filters.append(f)
        return self

    def _has_debugging(self):  # pragma: nocover
        """
        Iterators are difficult to debug.
        Use this function to decide whether we should convert iterators to tuples
        to make debugging easier.
        """
        return (
            sys.gettrace() is not None
            or os.environ.get("VECTORCODE_LOG_LEVEL") is not None
        )

    def __call__(self, files: Iterable[str]) -> Iterable[str]:
        if self._has_debugging():  # pragma: nocover
            files = tuple(files)
            logger.debug(
                f"Applying the following filters: {list(i.__name__ for i in self._filters)} to the following files ({len(files)}): {files}"
            )

        for f in self._filters:
            files = filter(f, files)

            if self._has_debugging():  # pragma: nocover
                files = tuple(files)
                logger.debug(f"{f.__name__} remaining items ({len(files)}): {files}")

        return files
