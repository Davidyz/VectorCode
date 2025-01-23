# VectorCode Command-line Tool 


<!-- mtoc-start -->

* [Installation](#installation)
  * [Chromadb](#chromadb)
* [Getting Started](#getting-started)
  * [Initialising a Project](#initialising-a-project)
  * [Configuring VectorCode](#configuring-vectorcode)
* [Vectorising Your Code](#vectorising-your-code)
* [Making a Query](#making-a-query)
* [Listing All Collections](#listing-all-collections)
* [Removing a Collection](#removing-a-collection)
* [Checking Project Setup](#checking-project-setup)
* [Shell Completion](#shell-completion)
* [For Developers](#for-developers)
  * [`vectorcode query`](#vectorcode-query)
  * [`vectorcode vectorise`](#vectorcode-vectorise)
  * [`vectorcode ls`](#vectorcode-ls)

<!-- mtoc-end -->

## Installation

After 0.1.3, the CLI supports Python 3.11~3.13. The recommended way of
installation is through [`pipx`](https://pipx.pypa.io/stable/), which will
create a virtual environment for the package itself that doesn't mess up with
your system Python or project-local virtual environments.

After installing `pipx`, run `pipx install vectorcode` in your shell. To
specify a particular version of Python, use the `--python` flag. For 
example, `pipx install vectorcode --python python3.11`.

> [!NOTE] 
> The command only install VectorCode and `SentenceTransformer`, the default
> embedding engine. To use a different embedding function supported by Chromadb,
> you may need to use `pipx inject` to install extra dependences to the virtual
> environment that `pipx` creates for VectorCode.

### Chromadb
[Chromadb](https://www.trychroma.com/) is the vector database used by VectorCode
to store and retrieve the code embeddings. It is generally not necessary to 
install Chromadb itself as a stand-alone application because it's already 
bundled with VectorCode as a dependency, but having a Chromadb host running in
the background may help with the performance because it reduces the IO overhead.
You may consult Chromadb documentation to find out more about how to set up a
local Chromadb server.

## Getting Started
### Initialising a Project
For each project, VectorCode creates a collection (similar to tables in
traditional databases) and puts the code embeddings in the corresponsing
collection. In the root directory of a project, you may run `vectorcode init`.
This will initialise the repository with a subdirectory
`project_root/.vectorcode/`. This will mark this directory a _project root_, a
concept that will later be used to construct the collection. You may put a
`config.json` file in `project_root/.vectorcode`. This file may be used to store
project-specific settings such as embedding functions and database entry point
(more on this later). If you already have a global configuration file at
`~/.config/vectorcode/config.json`, it will be copied to
`project_root/.vectorcode/config.json` when you run `vectorcode init`.

If you skip this step, VectorCode may still work but you have to make sure that
you always stay at the directory where you vectorised your code, because
VectorCode defaults to the current working directory as the path to the
embedding collection.

### Configuring VectorCode
The JSON configuration file may hold the following values:
- `embedding_function`: string, one of the embedding functions supported by [Chromadb](https://www.trychroma.com/) 
  (find more [here](https://docs.trychroma.com/docs/embeddings/embedding-functions) and 
  [here](https://docs.trychroma.com/integrations/chroma-integrations)). For
  example, Chromadb supports Ollama as `chromadb.utils.embedding_functions.OllamaEmbeddingFunction`,
  and the corresponding value for `embedding_function` would be `OllamaEmbeddingFunction`. Default: `SentenceTransformerEmbeddingFunction`;
- `embedding_params`: dictionary, stores whatever initialisation parameters your embedding function
  takes. For `OllamaEmbeddingFunction`, if you set `embedding_params` to:
  ```json
  {
    "url": "http://127.0.0.1:11434/api/embeddings",
    "model_name": "nomic-embed-text"
  }
  ```
  Then the embedding function object will be initialised as
  `OllamaEmbeddingFunction(url="http://127.0.0.1:11434/api/embeddings",
  model_name="nomic-embed-text")`. Default: `{}`;
- `host` and `port`: string and integer, Chromadb server host and port. VectorCode will start a
  local HTTP server at `host:port` if it's not already accessible. This allows
  the use of `AsyncHttpClient`. Default: `127.0.0.1:8000`;
- `db_path`: string, Path to local persistent database. This is where the files for 
  your database will be stored. Default: `~/.local/share/vectorcode/chromadb/`;
- `chunk_size`: integer, the maximum number of characters per chunk. A larger
  value reduces the number of items in the database, and hence accelerates the
  search, but at the cost of potentially truncated data and lost information.
  Default: `-1` (no chunking), but it's **highly** recommended to set it to a
  positive integer that works for your model when working with large documents;
- `overlap_ratio`: float between 0 and 1, the ratio of overlapping/shared content 
  between 2 adjacent chunks. A larger ratio improves the coherences of chunks,
  but at the cost of increasing number of entries in the database and hence
  slowing down the search. Default: `0.2`;
- `query_multplier`: integer, when you use the `query` command to retrieve `n` documents,
  VectorCode will check `n * query_multplier` chunks and return at most `n` 
  documents. A larger value of `query_multplier`
  guarantees the return of `n` documents, but with the risk of including too
  many less-relevant chunks that may affect the document selection. Default: 
  `-1` (any negative value means selecting documents based on all indexed chunks);
- `reranker`: string, experimental reranking support using 
  [`FlagEmbedding`](https://github.com/FlagOpen/FlagEmbedding) (_This is still
  experimental and slows down the query even if it works. Use at your own risk,
  and open an issue if things go wrong. Requires <Python3.13)_;
- `db_settings`: dictionary, works in a similar way to `embedding_params`, but 
  for Chromadb client settings so that you can configure 
  [authentication for remote Chromadb](https://docs.trychroma.com/production/administration/auth).

## Vectorising Your Code

Run `vectorcode vectorise <path_to_your_code>` or `vectorcode vectorise
<directory> -r`. There are a few extra tweaks you may use:

- chunk size: embedding APIs may truncate long documents so that the documents 
  can be handled by the embedding models. To solve this, VectorCode implemented
  basic chunking features that chunks the documents into smaller segments so
  that the embeddings are more representative of the code content. To adjust the
  chunk size when vectorising, you may either set the `chunk_size` option in the
  JSON configuration file, or use `--chunk_size`/`-c` parameter of the
  `vectorise` command to specify the maximum number of characters per chunk;
- overlapping ratio: when the chunk size is set to $c$ and overlapping ratio set
  to $o$, the maximum number of repeated content between 2 adjacent chunks will
  be $co$. This prevents loss of information due to the key characters being
  cut into 2 chunks. To configure this, you may either set `overlap_ratio` in 
  JSON configuration file or use `--overlap`/`-o` parameter.

Note that, the documents being vectorised is not limited to source code. You can
even try documentation/README, or files that are in the filesystem but not in the
project directory (yes I'm talking about neovim lua runtimes).

This command also respects `.gitignore`. It by default skips files in
`.gitignore`. To override this, run the `vectorise` command with `-f`/`--force`
flag.

## Making a Query

To retrieve a list of documents from the database, you can use the following command:
```bash 
vectorcode query "your query message"
```
The command can take an arbitrary number of query words, but make sure that
full-sentences are enclosed in quotation marks. Otherwise, they may be
interpreted as separated words and the embeddings may be inaccurate. The 
returned results are sorted by their similarity to the query message.

You may also specify how many documents should be retrieved with `-n`/`--number`
parameter (default is 5). This is the maximum number of documents that may be
returned. Depending on a number of factors, the actual returned documents may be
less than this number.

You may also set a multiplier for the queries. When VectorCode sends queries to
the database, it recieves chunks, not document. It then uses some scoring
algorithms to determine which documents are the best fit. The multiplier, set by
command-line flag `--multiplier` or `-m`, defines how many chunks VectorCode
will request from the database. The default is `-1`, which means to retrieve all 
chunks. A larger multiplier guarantees the return of `n` documents, but with the risk
of including too many less-relevant chunks that may affect the document selection.

The `query` subcommand also supports customising chunk size and overlapping
ratio because when the query message is too long it might be necessary to chunk
it. The parameters follow the same syntax as in `vectorise` command.


## Listing All Collections

You can use `vectorcode ls` command to list all collections in your Chromadb.
This is useful if you want to check whether the collection has been created for
the current project or not. The output will be a table with 4 columns:
- Project Root: path to the directory where VectorCode vectorised;
- Collection Size: number of chunks in the database;
- Number of Files: number of files that have been indexed;
- Embedding Function: name of embedding function used for this collection.

## Removing a Collection

You can use `vectorcode drop` command to remove a collection from Chromadb. This
is useful if you want to clean up your Chromadb database, or if the project has 
been deleted, and you don't need its embeddings any more. 

## Checking Project Setup
You may run `vectorcode check` command to check whether VectorCode is properly 
installed and configured for your project. This currently supports only 1 check:

- `config`: checks whether a project-local configuration directory exists.

Running `vectorcode check config` is faster than running `vectorcode query
some_message` and then getting an empty results.

## Shell Completion

VectorCode supports shell completion for bash/zsh/tcsh. You can use `vectorcode -s {bash,zsh,tcsh}`
or `vectorcode --print-completion {bash,zsh,tcsh}` to print the completion script
for your shell of choice.

## For Developers
To develop a tool that makes use of VectorCode, you may find the `--pipe`/`-p`
flag helpful. It formats the output into JSON and suppress other outputs so that 
you can grab whatever's in the `STDOUT` and parse it as a JSON document. In
fact, this is exactly what I did when I wrote the neovim plugin.

### `vectorcode query`
For the query command, here's the format printed in the `pipe` mode:
```json 
[
    {
        "path": "path_to_your_code.py", 
        "document":"import something"
    },
    {
        "path": "path_to_another_file.py",
        "document": "print('hello world')"
    }
]
```
Basically an array of dictionaries with 2 keys: `"path"` for the path to the
document, and `"document"` for the content of the document.

### `vectorcode vectorise`
The output is in JSON format. It contains a dictionary with the following fields:
- `"add"`: number of added documents;
- `"update"`: number of updated documents;
- `"removed"`: number of removed documents;

### `vectorcode ls`
A JSON array of collection information of the following format will be printed:
```json 
{
    "project_root": str,
    "user": str,
    "hostname": str,
    "collection_name": str,
    "size": int,
    "num_files": int,
    "embedding_function": str
}
```
- `"project_root"`: the path to the `project-root`;
- `"user"`: your *nix username, which are automatically added when vectorising to
  avoid collision;
- `"hostname"`: your *nix hostname. The purpose of this field is the same as the
  `"user"`;
- `"collection_name"`: the unique identifier for the project used in the database;
- `"size"`: number of chunks stored in the database;
- `"num_files"`: number of files that have been vectorised in the project.
