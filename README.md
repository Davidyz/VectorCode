# VectorCode

VectorCode is a code repository indexing tool. It helps you write better prompt
for your coding LLMs by indexing and providing information about the code
repository you're working on. This repository also contains the corresponding
neovim plugin because that's what I used to write this tool.

> [!NOTE]
> This project is in beta quality and only implements very basic retrieval and
> embedding functionalities. There are plenty of rooms for improvements and any
> help is welcomed.

> [!NOTE]
> [Chromadb](https://www.trychroma.com/), the vector database backend behind
> this project, supports multiple embedding engines. I developed this tool using
> SentenceTransformer, but if you encounter any issues with a different embedding 
> function, please open an issue (or even better, a pull request :D).

<!-- mtoc-start -->

* [Why VectorCode?](#why-vectorcode)
* [Prerequisites](#prerequisites)
* [Installation](#installation)
  * [NeoVim users:](#neovim-users)
* [Configuration](#configuration)
  * [CLI tool](#cli-tool)
* [Usage](#usage)
  * [CLI tool](#cli-tool-1)
    * [Initialising Project-Local Configuration](#initialising-project-local-configuration)
    * [Vectorising documents](#vectorising-documents)
    * [Querying from a collection](#querying-from-a-collection)
    * [Listing all collections](#listing-all-collections)
    * [Removing a collection](#removing-a-collection)
    * [Shell Completion](#shell-completion)
  * [NeoVim plugin](#neovim-plugin)
    * [Asynchronous Caching](#asynchronous-caching)
    * [Lualine Integration](#lualine-integration)
    * [The Boring Stuff](#the-boring-stuff)
  * [For Developers](#for-developers)
    * [`vectorise`](#vectorise)
    * [`query`](#query)
    * [`ls`](#ls)
    * [`drop`](#drop)
* [TODOs](#todos)

<!-- mtoc-end -->

## Why VectorCode?
LLMs usually have very limited understanding about close-source and/or infamous 
projects, as well as cutting edge developments that have not made it into the
releases. Their capabilities on these projects are quite limited. Take my little
toy sudoku-solving project as an example: When I wrote the first few lines and
want the LLM to fill in the list of solvers that I implemented in
`solver_candidates`, without project context, the completions are simply random 
guesses that *might* be part of another sudoku project:
![](./images/sudoku_no_rag.png)
But with RAG context provided by VectorCode, my completion LLM was able to
provide completions that I actually implemented:
![](./images/sudoku_with_rag.png)
This makes the completion results far more usable. 
[A similar strategy](https://docs.continue.dev/customize/deep-dives/codebase) 
is implemented in [continue](https://www.continue.dev/), a popular AI completion
and chat plugin available on VSCode and JetBrain products.

## Prerequisites


- ~A working instance of [Chromadb](https://www.trychroma.com/). A local docker
  image will suffice.~
- ~An embedding tool supported by [Chromadb](https://www.trychroma.com/), which 
you can find out more from 
[here](https://docs.trychroma.com/docs/embeddings/embedding-functions) and 
[here](https://docs.trychroma.com/integrations/chroma-integrations)~

As long as you managed to install `VectorCode` itself, you're good to go!

## Installation


I recommend using [`pipx`](https://github.com/pypa/pipx). This will take care of
the dependencies of `vectorcode` and create a dedicated virtual environment
without messing up your system Python.

Run the following command:
```bash 
pipx install vectorcode
```

To install the latest commit from GitHub, clone the repo and run `pipx install
<path_to_repo>`.

### NeoVim users:

This repo doubles as a neovim plugin. Use your favourite plugin manager to
install.

For `lazy.nvim`: 
```lua
{
  "Davidyz/VectorCode",
  dependencies = { "nvim-lua/plenary.nvim" },
  opts = { 
    n_query = 1, -- number of retrieved documents
    notify = true, -- enable notifications
    timeout_ms = 5000, -- timeout in milliseconds for the query operation.
    exclude_this = true, -- exclude the buffer from which the query is called.
                         -- This avoids repetition when you change some code but
                         -- the embedding has not been updated.
  },
  cond = function() return vim.fn.executable('vectorcode') == 1 end,
}
```

It might be helpful to add VectorCode as a dependency of your AI completion
plugin.

## Configuration

### CLI tool
This tool uses a JSON file to store the configuration. The global config is located at
`$HOME/.config/vectorcode/config.json`. You can also set a project-specific
configuration at `<project_root>/.vectorcode/config.json`. Options in the
project configuration will override the global config. The closest parent directory 
of the current working directory that contains a project-specific config will be
used as the project-root, but this can be overridden by the `--project_root`
flag.

```json 
{
    "embedding_function": 'SomeEmbeddingFunction',
    "embedding_params": {
    }
    "host": "localhost",  
    "port": 8000,
    "db_path": "~/.local/share/vectorcode/chromadb/",
}
```
The following are the available options for the JSON configuration file:
- `embedding_function`: One of the embedding functions supported by [Chromadb](https://www.trychroma.com/) 
  (find more [here](https://docs.trychroma.com/docs/embeddings/embedding-functions) and 
  [here](https://docs.trychroma.com/integrations/chroma-integrations)). For
  example, Chromadb supports Ollama as `chromadb.utils.embedding_functions.OllamaEmbeddingFunction`,
  and the corresponding value for `embedding_function` would be `OllamaEmbeddingFunction`. Default: `SentenceTransformerEmbeddingFunction`;
- `embedding_params`: Whatever initialisation parameters your embedding function
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
- `host` and `port`: Chromadb server host and port. Default: not set, in favour
  of local persistent client set by `db_path`. **Please only use with local or
  LAN Chromadb server because ChromaDB authentication is still WIP**;
- `db_path`: Path to local persistent database. **If `host` or `port` is set, this
  will be ignored**. Default: `~/.local/share/vectorcode/chromadb/`;
- `chunk_size`: integer, the maximum number of characters per chunk. A larger
  value reduces the number of items in the database, and hence accelerates the
  search, but at the cost of potentially truncated data and lost information.
  Default: `-1` (no chunking), but it's **highly** recommended to set it to a
  positive integer that works for your model when working with large documents;
- `overlap_ratio`: float between 0 and 1, the ratio of overlapping content in a
  between 2 adjacent chunks. A larger ratio improves the coherences of chunks,
  but at the cost of increasing number of entries in the database and hence
  slowing down the search. Default: `0.2`;
- `query_multplier`: when you use the `query` command to retrieve `n` documents,
  VectorCode will check `n * query_multplier` chunks and return at most `n` 
  documents. A larger value of `query_multplier`
  guarantees the return of `n` documents, but with the risk of including too
  many less-relevant chunks that may affect the document selection. Default: 
  `-1` (any negative value means selecting documents based on all indexed chunks);
- `reranker`: `str`, experimental reranking support using [`FlagEmbedding`](https://github.com/FlagOpen/FlagEmbedding).

For the convenience of deployment, environment variables in the
configuration values will be automatically expanded so that you can override
thing at run time without modifying the JSON.

Also, some of the built-in embedding functions supported by Chromadb requires
external library (such as `openai`) that are not included in the dependency
list. This is what Chromadb did, so I did the same. If you installed
`vectorcode` via `pipx`, you can install extra libraries by running the
following command:
```bash 
pipx inject vectorcode openai
```
And `openai` will be added to the virtual environment of `vectorcode`.

## Usage
### CLI tool
>This is an incomplete list of command-line options. You can always use
`vectorcode -h` to view the full list of arguments.

This tool creates a `collection` (just like tables in traditional databases) for each 
project. The collections are identified by project root, which, by default, is
the current working directory. You can override this by using the `--project_root
<path_to_your_project_root>` argument.

#### Initialising Project-Local Configuration

```bash
vectorcode init 
```
Create a project-local configuration at the current directory (or the directory
specified by the `--project_root` flag). This directory acts like a `.git`
directory. Consider the following file directory:
```
foo/
foo/.vectorcode/
foo/bar/
```
Running `vectorcode init` command in `foo/` creates the `foo/.vectorcode/`
directory, which can contain the project-local `config.json`. When 
`foo/.vectorcode/` is present, `foo/` will be used as the project-root for 
VectorCode when you run `vectorcode` command from any of the subdirectories of 
`foo/` (such as `foo/bar/`), unless overridden by `--project_root`. 

When you run `vectorcode init` and a global configuration file is present, it'll
be copied to your project-local config directory. **If a project-local
configuration is found, the global configuration will be ignored to avoid
confusion.**

#### Vectorising documents
```bash
vectorcode vectorise src/*.py
```
"Orphaned documents" that has been removed in your filesystem but still "exists"
in the database will be automatically cleaned. This will respect `.gitignore`
under project root, unless the `-f`/`--force` flag is set.

Extra options: 

- `--overlap` or `-o`: ratio of overlaps between chunks;
- `--chunk_size` or `-c`: maximum number of characters per chunk;
- `--recursive` or `-r`: recursively vectorise files in a directory;
- `--force` or `-f`: override `.gitignore`.

#### Querying from a collection
```bash 
vectorcode query "some query message"
```

Extra options:
- `--overlap` and `--chunk_size`: same as `vectorcode vectorise`;
- `--number` or `-n`: maximum number of returned documents;
- `--multiplier` or `-m`: query multiplier. See [CLI tool](#cli-tool-1);
- `--exclude`: files from which the query results should be ignored.

#### Listing all collections
```bash 
vectorcode ls 
```

#### Removing a collection

```bash 
vectorcode drop 
```

For `vectorise`, `query` and `ls`, adding `--pipe` or `-p` flag will convert the
output into a structured format. This is explained in detail [here](#for-developers).

#### Shell Completion

```bash 
vectorcode -s {bash,zsh,tcsh}
```
or 
```bash 
vectorcode --print-completion {bash,zsh,tcsh}
```
will print the completion script for the corresponding shell. Please consult the
documentation of your shell for instructions of how to use them.

### NeoVim plugin
> In this document I will be using [qwen2.5-coder](https://github.com/QwenLM/Qwen2.5-Coder) 
> as an example. Adjust your config when needed.

This is **NOT** a completion plugin, but a helper that facilitates prompting. It
provides APIs so that your completion engine (such as 
[`cmp-ai`](https://github.com/tzachar/cmp-ai)) can leverage the repository-level
context.

Using [`cmp-ai`](https://github.com/tzachar/cmp-ai) as an example, the
[configuration](https://github.com/tzachar/cmp-ai?tab=readme-ov-file#setup)
provides a `prompt` option, with which you can customize the prompt sent to the
LLM for each of the completion.

By consulting the [qwen2.5-coder documentation](https://github.com/QwenLM/Qwen2.5-Coder?tab=readme-ov-file#3-file-level-code-completion-fill-in-the-middle), 
we know that a trivial prompt can be constructed as the
following:
```lua 
prompt = function(lines_before, lines_after)
    return '<|fim_prefix|>' 
        .. lines_before 
        .. '<|fim_suffix|>' 
        .. lines_after 
        .. '<|fim_middle|>'
end
```

However, the information from such a context is limited to the document itself.
By utilising VectorCode and this plugin, you'll be able to construct contexts
that contain repository-level information:
```lua
prompt = function(lines_before, lines_after)
    local file_context = ""
    local ok, retrieval = pcall(
        -- safeguard the query call if your embedding function is over the
        -- network and may timeout on large documents.
        require("vectorcode").query,
            lines_before .. " " .. lines_after,
            { n_query = n_query } 
        )
    if ok then
        for _, source in pairs(retrieval) do
            -- This works for qwen2.5-coder.
            file_context = file_context
                .. "<|file_sep|>"
                .. source.path
                .. "\n"
                .. source.document
                .. "\n"
        end
    end
    return file_context
        ..'<|fim_prefix|>' 
        .. lines_before 
        .. '<|fim_suffix|>' 
        .. lines_after 
        .. '<|fim_middle|>'
```
Note that, the use of `<|file_sep|>` is documented in 
[qwen2.5-coder documentation](https://github.com/QwenLM/Qwen2.5-Coder?tab=readme-ov-file#3-file-level-code-completion-fill-in-the-middle) 
and is likely to be model-specific. You may need to figure out the best prompt
structure for your own model.

The number of files returned by the `query` function call can be configured
either by the `setup` function, or passed as an argument to the `query` call
which overrides the `setup` setting for this call:
```lua 
require("vectorcode").query(some_query_message, {n_query=5})
```
The second parameter follows the same structure as the `opts` table for the
`setup` function. Settings in this table will override the options in `setup`
for this `query` call. This allows adjusting the number of retrieved documents
on the fly.

> [!NOTE]
> This API is synchronous and will block your main nvim UI.

#### Asynchronous Caching

For applications that are sensitive to timing, the above process may not be
responsive enough. As you can see from using the CLI, the query itself takes
some noticeable amount of time. This is why I wrote a per-buffer async caching
mechanism that will overcome the issue to some extent.

To use the per-buffer async cache, you need to use the following API:
- `require("vectorcode.cacher").register_buffer(buf_nr?, opts?, query_cb?,
  events?)`: Register a buffer for background query. 
  - `buf_nr` (optional): integer, the buffer number to setup async runner in;
  - `opts` (optional): table, the same structure as what you use for `setup` 
    and the synchronous `query`. This opts will be used for all the async 
    updates managed by this plugin. This defaults to the option configured in 
    `setup`;
  - `query_cb` (optional): `fun(bufnr: integer):string`, a function that will
    be used to construct the query message. You can use this function to
    customise the message sent to the `vectorcode` CLI. This defaults to the
    whole buffer (for now);
  - `events` (optional): `string[]`, an array of `autocmd` events on which 
    the queries will be initialised. This defaults to 
    `{ "BufWritePost", "InsertEnter", "BufReadPost" }`;
  - `debounce` (optional): `int` (default to 10), the minimum amount of time
    between 2 asynchronous queries. Setting this option avoids excessive calls
    to the vector database, hence reduces resource usage.

  Calling this function on a buffer that has been registered will update its
  `opts` and `query_cb`.

- `require("vectorcode.cacher).query_from_cache(bufnr?)`: Returns the retrieval
  results from the most recent async cache for the given buffer. If the buffer
  has not been registered, it will return an empty array. The returned data is
  in the same format as the synchronous `query` API.
  - `bufnr` (optional): integer, the buffer number to retrieve cache from.
    Defaults to the current buffer.

With these async caching mechanism, you'll be able to utilise the retrieval with
minimum latency and without blocking the main UI. All you need to do is to setup
some kind of autocmd that register buffers, for example:
```lua
vim.api.nvim_create_autocmd("LspAttach", {
  callback = function()
    local bufnr = vim.api.nvim_get_current_buf()
    require("vectorcode.cacher").register_buffer(bufnr)
  end,
})
```
And in your completion prompt construction, you can use
`require("vectorcode.cacher").query_from_cache(bufnr)` to get the cached
retrieval results that you use to build your prompt.

#### Lualine Integration

```lua
opts = {
  tabline = {
    lualine_y = { require("vectorcode.cacher").lualine() }
  }
}
```

#### The Boring Stuff
Under the hood, the caching mechanism stores the information in
`vim.b[bufnr].vectorcode_cache`. The variable is a table with the following
definition:
```lua
{
  enabled = true, -- controls whether the async jobs will be run. 
  retrieval = {}, -- the cached retrieval result.
  options = {}, -- options passed from the `opts` argument when registering the
                -- buffer.
}
```
### For Developers

When the `--pipe` flag is set, the output of the CLI tool will be structured
into some sort of JSON string.

#### `vectorise`
The number of added, updated and removed entries will be printed.
```json
{
    "add": int,
    "update": int,
    "removed": int,
}
```
- `add`: number of added documents;
- `update`: number of updated (existing) documents;
- `removed`: number of removed documents due to original documents being
  deleted.

#### `query`
A JSON array of query results of the following format will be printed:
```json
{
    "path": str,
    "document": str,
}
```

- `path`: path to the file;
- `document`: content of the file.

#### `ls`
A JSON array of collection information of the following format will be printed:
```json 
{
    "project-root": str,
    "user": str,
    "hostname": str,
    "collection_name": str,
    "size": int,
    "num_files": int,
    "embedding_function": str
}
```

- `project_root`: path to the project directory (your code repository);
- `user`: your *nix username, which are automatically added when vectorising to
  avoid collision;
- `hostname`: your *nix hostname. The purpose of this field is the same as the
  `user` field;
- `collection_name`: the unique identifier of the collection in the database.
  This is the first 63 characters of the sha256 hash of the absolute path of the
  project root.
- `size`: number of chunks in the collection;
- `num_files`: number of files in the collection;
- `embedding_function`: name of embedding function used for the collection.

#### `drop`
The `drop` command doesn't offer a `--pipe` model output at the moment.

## TODOs
- [x] query by ~file path~ excluded paths;
- [ ] chunking support;
  - [x] add metadata for files;
  - [x] chunk-size configuration;
  - [ ] smarter chunking (semantics/syntax based);
  - [ ] configurable document selection from query results.
- [x] ~NeoVim Lua API with cache to skip the retrieval when a project has not
  been indexed~ Returns empty array instead;
- [x] job pool for async caching;
- [x] [persistent-client](https://docs.trychroma.com/docs/run-chroma/persistent-client);
- [ ] proper [remote Chromadb](https://docs.trychroma.com/production/administration/auth) support (with authentication, etc.);
- [x] respect `.gitignore`;
- [x] implement some sort of project-root anchors (such as `.git` or a custom
  `.vectorcode.json`) that enhances automatic project-root detection.
  **Implemented project-level `.vectorcode/config.json` as root anchor**
