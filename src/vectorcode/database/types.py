import json
from dataclasses import dataclass, field, fields
from enum import StrEnum
from typing import Any

import tabulate

from vectorcode.chunking import Chunk

CollectionID = str


class ResultType(StrEnum):
    document = "document"
    chunk = "chunk"


@dataclass
class QueryOpts:
    keywords: list[str]
    count: int | None = None
    return_type: ResultType = ResultType.chunk


@dataclass
class VectoriseStats:
    add: int = 0
    update: int = 0
    removed: int = 0
    skipped: int = 0
    failed: int = 0

    def to_json(self) -> str:
        return json.dumps(self.to_dict())

    def to_dict(self) -> dict[str, int]:
        return {i.name: getattr(self, i.name) for i in fields(self)}

    def to_table(self) -> str:
        _fields = fields(self)
        return tabulate.tabulate(
            [
                [i.name.capitalize() for i in _fields],
                [getattr(self, i.name) for i in _fields],
            ],
            headers="firstrow",
        )


@dataclass
class CollectionInfo:
    id: CollectionID
    path: str  # absolute path to the directory
    embedding_function: str
    database_backend: str
    file_count: int = 0
    chunk_count: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class FileInCollection:
    path: str
    sha256: str

    def __hash__(self):
        return hash(self.sha256)


@dataclass
class CollectionContent:
    files: list[FileInCollection] = field(default_factory=list)
    chunks: list[Chunk] = field(default_factory=list)
