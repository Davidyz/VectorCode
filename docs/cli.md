# VectorCode Command-line Tool 


<!-- mtoc-start -->

* [Installation](#installation)
  * [Install from Source](#install-from-source)
  * [Chromadb](#chromadb)
    * [For Windows Users](#for-windows-users)
  * [Legacy Environments](#legacy-environments)
* [Getting Started](#getting-started)
  * [Refreshing Embeddings](#refreshing-embeddings)
  * [If Anything Goes Wrong...](#if-anything-goes-wrong)
* [Advanced Usage](#advanced-usage)
  * [Initialising a Project](#initialising-a-project)
  * [Configuring VectorCode](#configuring-vectorcode)
  * [Vectorising Your Code](#vectorising-your-code)
  * [Making a Query](#making-a-query)
  * [Listing All Collections](#listing-all-collections)
  * [Removing a Collection](#removing-a-collection)
  * [Checking Project Setup](#checking-project-setup)
  * [Cleaning up](#cleaning-up)
* [Shell Completion](#shell-completion)
* [Hardware Acceleration](#hardware-acceleration)
* [For Developers](#for-developers)
  * [Working with Results](#working-with-results)
    * [`vectorcode query`](#vectorcode-query)
    * [`vectorcode vectorise`](#vectorcode-vectorise)
    * [`vectorcode ls`](#vectorcode-ls)
  * [LSP Mode](#lsp-mode)
  * [MCP Server](#mcp-server)
  * [Writing Prompts](#writing-prompts)

<!-- mtoc-end -->

## Installation

After 0.1.3, the CLI supports Python 3.11~3.13. The recommended way of
installation is through [`pipx`](https://pipx.pypa.io/stable/), which will
create a virtual environment for the package itself that doesn't mess up with
your system Python or project-local virtual environments.

After installing `pipx`, run:
```bash
pipx install vectorcode
```
in your shell. To specify a particular version of Python, use the `--python` 
flag. For example, `pipx install vectorcode --python python3.11`. For hardware
accelerated embedding, refer to [the relevant section](#hardware-acceleration).
If you need to install multiple dependency group (for [LSP](#lsp-mode) or
[MCP](#mcp-server)), you can use the following syntax:
```bash
pipx install vectorcode[lsp,mcp]
```
> [!NOTE] 
> The command only install VectorCode and `SentenceTransformer`, the default
> embedding engine. To use a different embedding function supported by Chromadb,
> you may need to use `pipx inject` to install extra dependences to the virtual
> environment that `pipx` creates for VectorCode. This may include OpenAI,
> Ollama and other self/cloud-hosted embedding model providers.

### Install from Source
To install from source, either `git clone` this reposority and run `pipx install
<path_to_vectorcode_repo>`, or use the git URL:
```bash
pipx install git+https://github.com/Davidyz/VectorCode
```

### Chromadb
[Chromadb](https://www.trychroma.com/) is the vector database used by VectorCode
to store and retrieve the code embeddings. Although it is already bundled with
VectorCode and you can absolutely use VectorCode just fine, it is **recommended** to
set up a standalone local server (they provides detailed instructions through
[docker](https://docs.trychroma.com/production/containers/docker) and
[systemd](https://cookbook.chromadb.dev/running/systemd-service/)), because this
will significantly reduce the IO overhead and avoid potential race condition.

> If you're setting up a standalone ChromaDB server, I recommend sticking to
> v0.6.3.
> ChromaDB recently released v1.0.0, which may not work with VectorCode. I'm
> testing with v1.0.0 and will publish a new release when it's ready.

#### For Windows Users

Windows support is not officially tested at this moment. [This PR](https://github.com/Davidyz/VectorCode/pull/40)
tracks my progress trying to provide better experiences for windows users.

### Legacy Environments

If your environment doesn't support `numpy` version 2.0+, the default,
unconstrained numpy picked by `pipx` may not work for you. In this case, you can
try installing the package by `pipx install vectorcode[legacy]`, which enforces 
numpy `v1.x`. If this doesn't help, please open an issue with your OS, CPU
architecture, python version and the vectorcode virtual environment 
(`pipx runpip vectorcode freeze`).

## Getting Started

`cd` into your project root repo, and run:
```bash
vectorcode init
```
This will initialise the project for VectorCode and create a `.vectorcode`
directory in your project root. This is where you keep your configuration file
for VectorCode, if any.

After that, you can start vectorising files for the project.
```bash
vectorcode vectorise src/**/*.py
```
> VectorCode doesn't track file changes, so you need to re-vectorise edited
> files. You may automate this by a git pre-commit hook, etc. See the 
> [wiki](https://github.com/Davidyz/VectorCode/wiki/Tips-and-Tricks#git-hooks)
> for examples to set them up.

Ideally, you should try to vectorise all source code in the repo, but for large 
repos you may experience slow queries. If that happens, try to `vectorcode drop` 
the project and only vectorise files that are important or informative.

And now, you're ready to make queries that will retrieve the relevant documents:
```bash
vectorcode query reranker -n 3
```
This will try to find the 3 most relevant documents in the embedding database
that are related to the query `reranker`. You can pass multiple query words:
```bash
vectorcode query embedding reranking -n 3
```
or if you want to query a sentence, wrap them in quotation mark:
```bash
vectorcode query "How to configure reranker model"
```
If things are going right, you'll see some paths being printed, followed by
their content. These are the selected documents that are relevant to the query.

If you want to wipe the embedding for the repository (to use a new embedding
function or after an upgrade with breaking changes), use
```bash
vectorcode drop
```

To see a full list of CLI options and tricks to optimise the retrieval, keep 
reading or use the `--help` flag.

### Refreshing Embeddings

To maintain the accuracy of the vector search, it's important to keep your
embeddings up-to-date. You can simply run the `vectorise` subcommand on a file
to refresh the embedding for a particular file, and the CLI provides a 
`vectorcode update` subcommand, which updates the embeddings for all files that 
are currently indexed by VectorCode for the current project. 

If you want something more automagic, check out 
[this secion in the wiki](https://github.com/Davidyz/VectorCode/wiki/Tips-and-Tricks#git-hooks) 
about setting up git hooks to trigger automatic embedding updates when you
commit/checkout to a different tag.

### If Anything Goes Wrong...

Please try the following and see if any of these fix your issue:

- [`drop`](#removing-a-collection) the collection and 
  [re-index it](#vectorising-your-code), because there may be changes in the way
  embeddings are stored in the database;
- upgrade/re-install the CLI (via `pipx` or however you installed VectorCode).

## Advanced Usage
### Initialising a Project
For each project, VectorCode creates a collection (similar to tables in
traditional databases) and puts the code embeddings in the corresponding
collection. In the root directory of a project, you may run `vectorcode init`.
This will initialise the repository with a subdirectory
`project_root/.vectorcode/`. This will mark this directory a _project root_, a
concept that will later be used to construct the collection. You may put a
`config.json` file in `project_root/.vectorcode`. This file may be used to store
project-specific settings such as embedding functions and database entry point
(more on this later). If you already have a global configuration file at
`~/.config/vectorcode/config.json`, it will be copied to
`project_root/.vectorcode/config.json` when you run `vectorcode init`. When a
project-local config file is present, the global configuration file is ignored
to avoid confusion. 

If you skip `vectorcode init`, VectorCode will look for a directory that
contains `.git/` subdirectory and use it as the _project root_. In this case, the
default global configuration will be used. If `.git/` does not exist, VectorCode
falls back to using the current working directory as the _project root_.

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
- `host` and `port`: string and integer, Chromadb server host and port. VectorCode will start an
  HTTP server for Chromadb at a randomly picked free port on `localhost` if your 
  configured `host:port` is not accessible. This allows the use of `AsyncHttpClient`. 
  Default: `127.0.0.1:8000`;
- `db_path`: string, Path to local persistent database. This is where the files for 
  your database will be stored. Default: `~/.local/share/vectorcode/chromadb/`;
- `chunk_size`: integer, the maximum number of characters per chunk. A larger
  value reduces the number of items in the database, and hence accelerates the
  search, but at the cost of potentially truncated data and lost information.
  Default: `2500`. To disable chunking, set it to a negative number;
- `overlap_ratio`: float between 0 and 1, the ratio of overlapping/shared content 
  between 2 adjacent chunks. A larger ratio improves the coherences of chunks,
  but at the cost of increasing number of entries in the database and hence
  slowing down the search. Default: `0.2`. _Starting from 0.4.11, VectorCode
  will use treesitter to parse languages that it can automatically detect. It
  uses [pygments](https://github.com/pygments/pygments) to guess the language
  from filename, and 
  [tree-sitter-language-pack](https://github.com/Goldziher/tree-sitter-language-pack) 
  to fetch the correct parser. `overlap_ratio` has no effects when treesitter
  works. If VectorCode fails to find an appropriate parser, it'll fallback to
  the legacy naive parser, in which case `overlap_ratio` works exactly in the
  same way as before;_
- `query_multiplier`: integer, when you use the `query` command to retrieve `n` documents,
  VectorCode will check `n * query_multiplier` chunks and return at most `n` 
  documents. A larger value of `query_multiplier`
  guarantees the return of `n` documents, but with the risk of including too
  many less-relevant chunks that may affect the document selection. Default: 
  `-1` (any negative value means selecting documents based on all indexed chunks);
- `reranker`: string, a reranking model supported by 
  [`CrossEncoder`](https://sbert.net/docs/package_reference/cross_encoder/index.html). 
  A list of available models is available on their documentation. The default
  model is `"cross-encoder/ms-marco-MiniLM-L-6-v2"`. You can disable the use of
  `CrossEncoder` by setting this option to a falsy value that is not `null`,
  such as `false` or `""` (empty string);
- `reranker_params`: dictionary, similar to `embedding_params`. The options
  passed to `CrossEncoder` class constructor;
- `db_settings`: dictionary, works in a similar way to `embedding_params`, but 
  for Chromadb client settings so that you can configure 
  [authentication for remote Chromadb](https://docs.trychroma.com/production/administration/auth);
- `hnsw`: a dictionary of 
  [hnsw settings](https://cookbook.chromadb.dev/core/configuration/#hnsw-configuration) 
  that may improve the query performances or avoid runtime errors during
  queries. **It's recommended to re-vectorise the collection after modifying these
  options, because some of the options can only be set during collection
  creation.** Example:
  ```json
  "hnsw": {
    "hnsw:M": 64,
    "hnsw:construction_ef": 100
  }
  ```
- `chunk_filters`: `dict[str, list[str]]`, a dictionary where the keys are
  [language name](https://github.com/Goldziher/tree-sitter-language-pack?tab=readme-ov-file#available-languages)
  and values are lists of [Python regex patterns](https://docs.python.org/3/library/re.html) 
  that will match chunks to be excluded from being vectorised. This only applies
  to languages supported by treesitter chunker. By default, no filters will be
  added. Example configuration:
  ```json5
  "chunk_filters": {
    "python": ["^[^a-zA-Z0-9]+$"], // multiple patterns will be merged (unioned)
    // or you can use wildcard to match any languages that has no dedicated filters:
    "*": ["^[^a-zA-Z0-9]+$"],
  }
  ```

### Vectorising Your Code

Run `vectorcode vectorise <path_to_your_file>` or `vectorcode vectorise
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
  be $c \times o$. This prevents loss of information due to the key characters being
  cut into 2 chunks. To configure this, you may either set `overlap_ratio` in 
  JSON configuration file or use `--overlap`/`-o` parameter.

Note that, the documents being vectorised is not limited to source code. You can
even try documentation/README, or files that are in the filesystem but not in the
project directory (yes I'm talking about neovim lua runtimes).

This command also respects `.gitignore`. It by default skips files in
`.gitignore`. To override this, run the `vectorise` command with `-f`/`--force`
flag.

As a shorthand, you can create a file at `project_root/.vectorcode/vectorcode.include`.
This file should follow the same syntax as a 
[`gitignore` file](https://git-scm.com/docs/gitignore). Files matched by this
specs will be vectorised when you run `vectorcode vectorise` without specifying
files. This file has lower priority than `.gitignore`, but you can override this
by the `-f` flag. It also doesn't assume `--recursive`, so if you want to add a
whole directory to this file, you can use `dir/**`, which matches all content
of `dir/` recursively.

Similarly, you can also create a `project_root/.vectorcode/vectorcode.exclude`
file to denote any files that you want to exclude. This is useful when you have
some files that should be tracked by git, but are not necessary to be indexed by
VectorCode.

There's also a `update` subcommand, which updates the embedding for all the indexed 
files and remove the embeddings for files that no longer exist.

### Making a Query

To retrieve a list of documents from the database, you can use the following command:
```bash 
vectorcode query "your query message"
```
The command can take an arbitrary number of query words, but make sure that
full-sentences are enclosed in quotation marks. Otherwise, they may be
interpreted as separated words and the embeddings may be inaccurate. The 
returned results are sorted by their similarity to the query message.

You may also specify how many documents should be retrieved with `-n`/`--number`
parameter (default is 1). This is the maximum number of documents that may be
returned. Depending on a number of factors, the actual returned documents may be
less than this number but at least 1 document will be returned.

You may also set a multiplier for the queries. When VectorCode sends queries to
the database, it receives chunks, not document. It then uses some scoring
algorithms to determine which documents are the best fit. The multiplier, set by
command-line flag `--multiplier` or `-m`, defines how many chunks VectorCode
will request from the database. The default is `-1`, which means to retrieve all 
chunks. A larger multiplier guarantees the return of `n` documents, but with the risk
of including too many less-relevant chunks that may affect the document selection.

The `query` subcommand also supports customising chunk size and overlapping
ratio because when the query message is too long it might be necessary to chunk
it. The parameters follow the same syntax as in `vectorise` command.

The CLI defaults to return the relative path of the documents from the project
root. To use absolute path, add the `--absolute` flag.

If you wish to limit the output to "path only" or "document (content) only", you
can achieve this by using the `--include` flag:
```
vectorcode query foo bar --include path
```
This will only include the `path` in the output. This is effective for both
normal CLI usage and [`--pipe` mode](#for-developers).

For some applications, it may be overkill to use the full document as context
and all you need is the chunks. You can do this by using `--include chunk` or
`--include chunk path` in the command. This will return chunks from the
document, and in `pipe` mode the objects will also include the line numbers of 
the first and last lines in the chunk. Note that `chunk` and `document` cannot be used at
the same time, and the number of query result (the `-n` parameter) will refer to
the number of retrieved chunks when you use `--include chunk`. For the sake of 
completeness, the first and last lines of a chunk will be completed to include
the whole lines if the chunker broke the text from mid-line.

### Listing All Collections

You can use `vectorcode ls` command to list all collections in your ChromaDB.
This is useful if you want to check whether the collection has been created for
the current project or not. The output will be a table with 4 columns:
- Project Root: path to the directory where VectorCode vectorised;
- Collection Size: number of chunks in the database;
- Number of Files: number of files that have been indexed;
- Embedding Function: name of embedding function used for this collection.

This can only discover collections that are stored in the same ChromaDB
instance.

### Removing a Collection

You can use `vectorcode drop` command to remove a collection from Chromadb. This
is useful if you want to clean up your Chromadb database, or if the project has 
been deleted, and you don't need its embeddings any more. 

### Checking Project Setup
You may run `vectorcode check` command to check whether VectorCode is properly 
installed and configured for your project. This currently supports only 1 check:

- `config`: checks whether a project-local configuration directory exists.
  Prints the project-root if successful, otherwise returns a non-zero exit code.

Running `vectorcode check config` is faster than running `vectorcode query
some_message` and then getting an empty results.

### Cleaning up

For empty collections and collections for removed projects, you can use the
`vectorcode clean` command to remove them at once.

## Shell Completion

VectorCode supports shell completion for bash/zsh/tcsh. You can use `vectorcode -s {bash,zsh,tcsh}`
or `vectorcode --print-completion {bash,zsh,tcsh}` to print the completion script
for your shell of choice.

## Hardware Acceleration
> This section covers hardware acceleration when using sentence transformer as
> the embedding backend.

For Nvidia users this should work out of the box. If not, try setting the
following options in the JSON config file:
```json 
{
  "embedding_params": {
    "backend": "torch",
    "device": "cuda"
  },
}
```

For Intel users, [sentence transformer](https://www.sbert.net/index.html)
supports [OpenVINO](https://www.intel.com/content/www/us/en/developer/tools/openvino-toolkit/overview.html) 
backend for supported GPU. Run `pipx install vectorcode[intel]` which will 
bundle the relevant libraries when you install VectorCode. After that, you will
need to configure `SentenceTransformer` to use `openvino` backend. In your
`config.json`, set `backend` key in `embedding_params` to `"openvino"`:
```json 
{
  "embedding_params": {
    "backend": "openvino",
  },
}
```
This will run the embedding model on your GPU. This is supported even for
some integrated GPUs.

When using the default embedding function, any options inside the
`"embedding_params"` will go to the class constructor of `SentenceTransformer`,
so you can always take a look at 
[their documentation](https://www.sbert.net/docs/package_reference/sentence_transformer/SentenceTransformer.html#sentence_transformers.SentenceTransformer)
for detailed information _regardless of your platform_.

## For Developers
To develop a tool that makes use of VectorCode, you may find the `--pipe`/`-p`
flag helpful. It formats the output into JSON and suppress other outputs so that 
you can grab whatever's in the `STDOUT` and parse it as a JSON document. In
fact, this is exactly what I did when I wrote the neovim plugin.

### Working with Results

#### `vectorcode query`
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

If you used `--include chunk path` parameters, the array will look like this:
```json
[
    {
        "path": "path_to_your_code.py",
        "chunk": "foo",
        "start_line": 1,
        "end_line": 1,
    },
    {
        "path": "path_to_another_file.py",
        "chunk": "bar",
        "start_line": 1,
        "end_line": 1,
    }
]
```
Keep in mind that both `start_line` and `end_line` are inclusive.

#### `vectorcode vectorise`
The output is in JSON format. It contains a dictionary with the following fields:
- `"add"`: number of added documents;
- `"update"`: number of updated documents;
- `"removed"`: number of removed documents;

#### `vectorcode ls`
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

### LSP Mode

There's an experimental implementation of VectorCode CLI, which accepts requests
of [`workspace/executeCommand`](https://microsoft.github.io/language-server-protocol/specifications/lsp/3.17/specification/#workspace_executeCommand) 
from `STDIO`. This allows the CLI to keep the embedding model loaded in the
memory/VRAM, and therefore speed up the query by avoiding the IO overhead of
loading the models.

The experimental language server can be installed via the `lsp` dependency
group:
```bash
pipx install vectorcode[lsp]

## or if you have an existing `vectorcode` install:

pipx inject vectorcode vectorcode[lsp] --force
```

The LSP request for the `workspace/executeCommand` is defined as follows: 
```
{
    command: str
    arguments: list[Any]
}
```
For the `vectorcode-server`, the only valid value for the `command` key is 
`"vectorcode"`, and `arguments` is any other remaining components of a valid CLI
command. For example, to execute `vectorcode query -n 10 reranker`, the request
would be: 
```
{
    command: "vectorcode",
    arguments: ["query", "-n", "10", "reranker"]
}
```

The `vectorcode-server` optionally accepts a `--project_root` parameter, which
specifies the default project root for this process. If not specified, it
will:

1. try to find a project root by root anchors (`.vectorcode` or `.git`)
   starting from the current working directory;
2. if 1 fails, but the first request contains a `--project_root` parameter, it
   will use that as the default project root for this process;
3. if 2 fails too, the process throws an error.

Note that:

1. For easier parsing, `--pipe` is assumed to be enabled in LSP mode;
2. At the time this only work with vectorcode setup that uses a **standalone
   ChromaDB server**, which is not difficult to setup using docker;
3. At the time this only work with `query` subcommand. I will consider adding
   support for other subcommand but first I need to figure out how to properly
   manage `project_root` across different requests if they change.

### MCP Server

[Model Context Protocol (MCP)](https://modelcontextprotocol.io/introduction) is 
an open protocol that standardizes how applications provide context to LLMs.
VectorCode provides an experimental implementation that provides the following 
features:

- `ls`: list local collections, similar to the `ls` subcommand in the CLI;
- `query`: query from a given collection, similar to the `query` subcommand in
  the CLI.

To try it out, install the `vectorcode[mcp]` dependency group and the MCP server 
is available in the shell as `vectorcode-mcp-server`, and make sure you're using 
a [standalone chromadb server](#chromadb) configured in the [JSON](#configuring-vectorcode) 
via the `host` and `port` options. 

### Writing Prompts

If you want to integrate VectorCode in your LLM application, you may want to
write some prompt that tells the LLM how to use this tool. Apart from the
function signatures, a list of instructions used by the MCP server is included in
this package. This can be retrieved by running the `vectorcode prompts` command.
