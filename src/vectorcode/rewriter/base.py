from abc import ABC, abstractmethod

from vectorcode.cli_utils import Config


class RewriterBase(ABC):  # pragma: nocover
    def __init__(self, config: Config) -> None:
        super().__init__()
        self.config = config

    @abstractmethod
    async def rewrite(self, original_query: list[str]) -> list[str]:
        raise NotImplementedError
