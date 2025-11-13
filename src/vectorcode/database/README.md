# Database Configuration

This document provides the `db_params` configuration for the supported database connectors in VectorCode.

For instructions on how to add a new database connector, please refer to [DEVELOPERS.md](./DEVELOPERS.md).


<!-- mtoc-start -->

* [ChromaDB (v0.6.3)](#chromadb-v063)

<!-- mtoc-end -->

## ChromaDB (v0.6.3)

The `ChromaDB0Connector` is used for ChromaDB versions 0.6.3.

- **`db_type`**: `"ChromaDB0"`. The `Connector` suffix is optional and will be added automatically.

- **`db_params`**:
  An example of the `db_params` for `ChromaDB0Connector` in your `config.json5`:
  ```json5
  {
    "db_params": {
      "db_url": "http://127.0.0.1:8000",
      "db_path": "~/.local/share/vectorcode/chromadb/",
      "db_log_path": "~/.local/share/vectorcode/",
      "db_settings": {},
      "hnsw": {
        "hnsw:M": 64,
      },
    },
  }
  ```

  - `db_url`: The URL of the ChromaDB server. Defaults to `"http://127.0.0.1:8000"`.
  - `db_path`: Path to the directory where ChromaDB stores its data. Defaults to `"~/.local/share/vectorcode/chromadb/"`.
  - `db_log_path`: Path to the directory for ChromaDB log files. Defaults to `"~/.local/share/vectorcode/"`.
  - `db_settings`: Additional ChromaDB settings. You usually don't need to touch this, but in case you do, you can refer to [ChromaDB source](https://github.com/chroma-core/chroma/blob/a3b86a0302a385350a8f092a5f89a2dcdebcf6be/chromadb/config.py#L101) for details. Defaults to `{}`.
  - `hnsw`: HNSW index parameters. Defaults to `{"hnsw:M": 64}`.
