import concurrent.futures as futures
import hashlib
import json
import os
import sys
import uuid
from threading import Lock

import pathspec
import tabulate
import tqdm
from chromadb.api.types import IncludeEnum

from vectorcode.chunking import FileChunker
from vectorcode.cli_utils import Config, expand_globs, expand_path
from vectorcode.common import get_client, make_or_get_collection, verify_ef


def hash_str(string: str) -> str:
    """Return the sha-256 hash of a string."""
    return hashlib.sha256(string.encode()).hexdigest()


def get_uuid() -> str:
    return uuid.uuid4().hex


def vectorise(configs: Config) -> int:
    client = get_client(configs)
    collection = make_or_get_collection(client, configs)
    if not verify_ef(collection, configs):
        return 1
    files = expand_globs(configs.files or [], recursive=configs.recursive)

    gitignore_path = os.path.join(configs.project_root, ".gitignore")
    if os.path.isfile(gitignore_path):
        with open(gitignore_path) as fin:
            gitignore_spec = pathspec.GitIgnoreSpec.from_lines(fin.readlines())
    else:
        gitignore_spec = None

    stats = {"add": 0, "update": 0, "removed": 0}
    collection_lock = Lock()
    stats_lock = Lock()

    def chunked_add(file_path):
        if (
            (not configs.force)
            and gitignore_spec is not None
            and gitignore_spec.match_file(file_path)
        ):
            # handles gitignore.
            return

        full_path_str = str(expand_path(str(file_path), True))
        with collection_lock:
            num_existing_chunks = len(
                collection.get(
                    where={"path": full_path_str}, include=[IncludeEnum.metadatas]
                )["ids"]
            )
        if num_existing_chunks:
            with collection_lock:
                collection.delete(where={"path": full_path_str})
            with stats_lock:
                stats["update"] += 1
        else:
            with stats_lock:
                stats["add"] += 1
        with open(full_path_str) as fin:
            for chunk in FileChunker(configs.chunk_size, configs.overlap_ratio).chunk(
                fin
            ):
                with collection_lock:
                    collection.add(
                        ids=[get_uuid()],
                        documents=[chunk],
                        metadatas=[{"path": full_path_str}],
                    )

    with tqdm.tqdm(
        total=len(files), desc="Vectorising files...", disable=configs.pipe
    ) as bar:
        try:
            with futures.ThreadPoolExecutor(
                max_workers=max((os.cpu_count() or 1) - 1, 1)
            ) as executor:
                jobs = (executor.submit(chunked_add, file) for file in files)
                for future in futures.as_completed(jobs):
                    bar.update(1)
        except KeyboardInterrupt:
            print("Abort.", file=sys.stderr)
            return 1

    all_results = collection.get(include=[IncludeEnum.metadatas])
    if all_results is not None and all_results.get("metadatas"):
        for idx in range(len(all_results["ids"])):
            path_in_meta = str(all_results["metadatas"][idx].get("path"))
            if path_in_meta is not None and not os.path.isfile(path_in_meta):
                collection.delete(where={"path": path_in_meta})
                stats["removed"] += 1

    if configs.pipe:
        print(json.dumps(stats))
    else:
        print(
            tabulate.tabulate(
                [
                    ["Added", "Updated", "Removed"],
                    [stats["add"], stats["update"], stats["removed"]],
                ],
                headers="firstrow",
            )
        )
    return 0
