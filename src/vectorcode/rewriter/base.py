from abc import ABC, abstractmethod

from vectorcode.cli_utils import Config


class RewriterBase(ABC):  # pragma: nocover
    def __init__(self, config: Config) -> None:
        super().__init__()
        self.config = config

    @classmethod
    def create(cls, configs: Config):
        try:
            return cls(configs)
        except Exception as e:
            e.add_note(
                "\n"
                + (
                    cls.__doc__
                    or f"There was an issue initialising {cls}. Please doublecheck your configuration."
                )
            )
            raise

    @abstractmethod
    async def rewrite(self, original_query: list[str]) -> list[str]:
        raise NotImplementedError
