# NeoVim Plugin
> [!NOTE]
> This plugin depends on the CLI tool. Please go through 
> [the CLI documentation](./cli.md) and make sure the VectorCode CLI is working
> before proceeding.

> [!NOTE]
> When the neovim plugin doesn't work properly, please try upgrading the CLI
> tool to the latest version before opening an issue.


<!-- mtoc-start -->

* [Installation](#installation)
  * [Nix](#nix)
* [Integrations](#integrations)
* [User Command](#user-command)
  * [`VectorCode register`](#vectorcode-register)
  * [`VectorCode deregister`](#vectorcode-deregister)
* [Configuration](#configuration)
  * [`setup(opts?)`](#setupopts)
* [API Usage](#api-usage)
  * [Synchronous API](#synchronous-api)
    * [`query(query_message, opts?, callback?)`](#queryquery_message-opts-callback)
    * [`check(check_item?)`](#checkcheck_item)
    * [`update(project_root?)`](#updateproject_root)
  * [Cached Asynchronous API](#cached-asynchronous-api)
    * [`cacher_backend.register_buffer(bufnr?, opts?)`](#cacher_backendregister_bufferbufnr-opts)
    * [`cacher_backend.query_from_cache(bufnr?)`](#cacher_backendquery_from_cachebufnr)
    * [`cacher_backend.async_check(check_item?, on_success?, on_failure?)`](#cacher_backendasync_checkcheck_item-on_success-on_failure)
    * [`cacher_backend.buf_is_registered(bufnr?)`](#cacher_backendbuf_is_registeredbufnr)
    * [`cacher_backend.buf_is_enabled(bufnr?)`](#cacher_backendbuf_is_enabledbufnr)
    * [`cacher_backend.buf_job_count(bufnr?)`](#cacher_backendbuf_job_countbufnr)
    * [`cacher_backend.make_prompt_component(bufnr?, component_cb?)`](#cacher_backendmake_prompt_componentbufnr-component_cb)
    * [Built-in Query Callbacks](#built-in-query-callbacks)
* [Debugging and Logging](#debugging-and-logging)

<!-- mtoc-end -->

## Installation
Use your favorite plugin manager. 

```lua 
{
  "Davidyz/VectorCode",
  version = "<0.7.0", -- optional, depending on whether you're on nightly or release
  dependencies = { "nvim-lua/plenary.nvim" },
  cmd = "VectorCode", -- if you're lazy-loading VectorCode
}
```
The VectorCode CLI and neovim plugin share the same release scheme (version
numbers). In other words, CLI 0.1.3 is guaranteed to work with neovim plugin
0.1.3, but if you use CLI 0.1.0 with neovim plugin 0.1.3, they may not work
together because the neovim plugin is built for a newer CLI release and depends
on newer features/breaking changes.

To ensure maximum compatibility, please either:
1. Use release build for VectorCode CLI and pin to the release tags for the
   neovim plugin;

**OR**

2. Use the latest commit for the neovim plugin with VectorCode installed from
   the latest GitHub commit.

It may be helpful to use a `build` hook to automatically upgrade the CLI when
the neovim plugin updates. For example, if you're using lazy.nvim and `pipx`,
you can use the following plugin spec:

```lua
{
  "Davidyz/VectorCode",
  version = "*",
  build = "uv tool upgrade vectorcode", -- This helps keeping the CLI up-to-date
  -- build = "pipx upgrade vectorcode", -- If you used pipx to install the CLI
  dependencies = { "nvim-lua/plenary.nvim" },
}
```

> This plugin is developed and tested on neovim _v0.11_. It may work on older
> versions, but I do not test on them before publishing.

### Nix

There's a community-maintained [nix package](https://nixpk.gs/pr-tracker.html?pr=413395) 
submitted by [@sarahec](https://github.com/sarahec) for the Neovim plugin.

## Integrations

[The wiki](https://github.com/Davidyz/VectorCode/wiki/Neovim-Integrations)
contains instructions to integrate VectorCode with the following plugins:

- [milanglacier/minuet-ai.nvim](https://github.com/milanglacier/minuet-ai.nvim);
- [olimorris/codecompanion.nvim](https://github.com/olimorris/codecompanion.nvim);
- [nvim-lualine/lualine.nvim](https://github.com/nvim-lualine/lualine.nvim);
- [CopilotC-Nvim/CopilotChat.nvim](https://github.com/CopilotC-Nvim/CopilotChat.nvim);
- [ravitemer/mcphub.nvim](https://github.com/ravitemer/mcphub.nvim).

## User Command
### `VectorCode register`

Register the current buffer for async caching. It's possible to register the
current buffer to a different vectorcode project by passing the `project_root`
parameter:
```
:VectorCode register project_root=path/to/another/project/
```
This is useful if you're working on a project that is closely related to a
different project, for example a utility repository for a main library or a
documentation repository. Alternatively, you can call the [lua API](#cached-asynchronous-api) in an autocmd:
```lua
vim.api.nvim_create_autocmd("LspAttach", {
  callback = function()
    local bufnr = vim.api.nvim_get_current_buf()
    cacher.async_check("config", function()
      cacher.register_buffer(
        bufnr,
        { 
          n_query = 10,
        }
      )
    end, nil)
  end,
  desc = "Register buffer for VectorCode",
})
```
The latter avoids the manual registrations, but registering too many buffers
means there will be a lot of background processes/requests being sent to
VectorCode. Choose these based on your workflow and the capability of your
system.

### `VectorCode deregister`

Deregister the current buffer. Any running jobs will be killed, cached results
will be deleted, and no more queries will be run.

## Configuration

### `setup(opts?)`
This function initialises the VectorCode client and sets up some default

```lua
-- Default configuration
require("vectorcode").setup(
  ---@type VectorCode.Opts
  {
    cli_cmds = {
      vectorcode = "vectorcode",
    },
    ---@type VectorCode.RegisterOpts
    async_opts = {
      debounce = 10,
      events = { "BufWritePost", "InsertEnter", "BufReadPost" },
      exclude_this = true,
      n_query = 1,
      notify = false,
      query_cb = require("vectorcode.utils").make_surrounding_lines_cb(-1),
      run_on_register = false,
    },
    async_backend = "default", -- or "lsp"
    exclude_this = true,
    n_query = 1,
    notify = true,
    timeout_ms = 5000,
    on_setup = {
      update = false, -- set to true to enable update when `setup` is called.
      lsp = false,
    }
    sync_log_env_var = false,
  }
)
```

The following are the available options for the parameter of this function:
- `cli_cmds`: A table to customize the CLI command names / paths used by the plugin.
  Supported key:
  - `vectorcode`: The command / path to use for the main CLI tool. Default: `"vectorcode"`.
- `n_query`: number of retrieved documents. A large number gives a higher chance
  of including the right file, but with the risk of saturating the context 
  window and getting truncated. Default: `1`;
- `notify`: whether to show notifications when a query is completed.
  Default: `true`;
- `timeout_ms`: timeout in milliseconds for the query operation. Applies to
  synchronous API only. Default: 
  `5000` (5 seconds);
- `exclude_this`: whether to exclude the file you're editing. Setting this to
  `false` may lead to an outdated version of the current file being sent to the
  LLM as the prompt, and can lead to generations with outdated information;
- `async_opts`: default options used when registering buffers. See 
  [`register_buffer(bufnr?, opts?)`](#register_bufferbufnr-opts) for details;
- `async_backend`: the async backend to use, currently either `"default"` or
  `"lsp"`. Default: `"default"`;
- `on_setup`: some actions that can be registered to run when `setup` is called.
  Supported keys:
  - `update`: if `true`, the plugin will run `vectorcode update` on startup to
    update the embeddings;
  - `lsp`: if `true`, the plugin will try to start the LSP server on startup so
    that you won't need to wait for the server loading when making your first 
    request.
- `sync_log_env_var`: `boolean`. If true, this plugin will automatically set the
  `VECTORCODE_LOG_LEVEL` environment variable for LSP or cmd processes started
  within your neovim session when logging is turned on for this plugin. Use at 
  caution because the CLI write all logs to stderr, which _may_ make this plugin 
  VERY verbose. See [Debugging and Logging](#debugging-and-logging) for details
  on how to turn on logging.

You may notice that a lot of options in `async_opts` are the same as the other
options in the top-level of the main option table. This is because the top-level
options are designated for the [Synchronous API](#synchronous-api) and the ones
in `async_opts` is for the [Cached Asynchronous API](#cached-asynchronous-api).
The `async_opts` will reuse the synchronous API options if not explicitly
configured.

## API Usage
This plugin provides 2 sets of APIs that provides similar functionalities. The
synchronous APIs provide more up-to-date retrieval results at the cost of
blocking the main neovim UI, while the async APIs use a caching mechanism to 
provide asynchronous retrieval results almost instantaneously, but the result
may be slightly out-of-date. For some tasks like chat, the main UI being
blocked/frozen doesn't hurt much because you spend the time waiting for response
anyway, and you can use the synchronous API in this case. For other tasks like 
completion, the async API will minimise the interruption to your workflow.


### Synchronous API
#### `query(query_message, opts?, callback?)`
This function queries VectorCode and returns an array of results.

```lua
require("vectorcode").query("some query message", {
    n_query = 5,
})
```
- `query_message`: string or a list of strings, the query messages;
- `opts`: The following are the available options for this function (see [`setup(opts?)`](#setupopts) for details):
```lua
{
    exclude_this = true,
    n_query = 1,
    notify = true,
    timeout_ms = 5000,
}
```
- `callback`: a callback function that takes the result of the retrieval as the
  only parameter. If this is set, the `query` function will be non-blocking and
  runs in an async manner. In this case, it doesn't return any value and 
  retrieval results can only be accessed by this callback function.

The return value of this function is an array of results in the format of
`{path="path/to/your/code.lua", document="document content"}`. 

For example, in [cmp-ai](https://github.com/tzachar/cmp-ai), you can add 
the path/document content to the prompt like this:
```lua
prompt = function(prefix, suffix)
    local retrieval_results = require("vectorcode").query("some query message", {
        n_query = 5,
    })
    for _, source in pairs(retrieval_results) do
        -- This works for qwen2.5-coder.
        file_context = file_context
            .. "<|file_sep|>"
            .. source.path
            .. "\n"
            .. source.document
            .. "\n"
    end
    return file_context
        .. "<|fim_prefix|>" 
        .. prefix 
        .. "<|fim_suffix|>" 
        .. suffix 
        .. "<|fim_middle|>"
end
```


#### `check(check_item?)`
This function checks if VectorCode has been configured properly for your project. See the [CLI manual for details](./cli.md).

```lua 
require("vectorcode").check()
```

The following are the available options for this function:
- `check_item`: Only supports `"config"` at the moment. Checks if a project-local 
  config is present.
Return value: `true` if passed, `false` if failed.

This involves the `check` command of the CLI that checks the status of the
VectorCode project setup. Use this as a pre-condition of any subsequent
use of other VectorCode APIs that may be more expensive (if this fails,
VectorCode hasn't been properly set up for the project, and you should not use
VectorCode APIs).

The use of this API is entirely optional. You can totally ignore this and call
`query` anyway, but if `check` fails, you might be spending the waiting time for
nothing.

#### `update(project_root?)`
This function calls `vectorcode update` at the current working directory.
`--project_root` will be added if the `project_root` parameter is not `nil`.
This runs async and doesn't block the main UI.

```lua
require("vectorcode").update()
```

### Cached Asynchronous API

The async cache mechanism helps mitigate the issue where the `query` API may
take too long and block the main thread. The following are the functions
available through the `require("vectorcode.cacher")` module.

From 0.4.0, the async cache module came with 2 backends that exposes the same
interface:

1. The `default` backend which works exactly like the original implementation
   used in previous versions;
2. The `lsp` based backend, which make use of the experimental `vectorcode-server`
   implemented in version 0.4.0. If you want to customise the LSP executable or
   any options supported by `vim.lsp.ClientConfig`, you can do so by using
   `vim.lsp.config()` or 
   [nvim-lspconfig](https://github.com/neovim/nvim-lspconfig). The LSP will
   attempt to read configurations from these 2 sources before it starts. (If
   `vim.lsp.config.vectorcode_server` is not `nil`, this will be used and
   nvim-lspconfig will be ignored.)


| Features | `default`                                                                                                 | `lsp`                                                                                                                     |
|----------|-----------------------------------------------------------------------------------------------------------|---------------------------------------------------------------------------------------------------------------------------|
| **Pros** | Fully backward compatible with minimal extra config required                                              | Less IO overhead for loading/unloading embedding models; Progress reports.                                                |
| **Cons** | Heavy IO overhead because the embedding model and database client need to be initialised for every query. | Requires `vectorcode-server`; Only works if you're using a standalone ChromaDB server; May contain bugs because it's new. |

You may choose which backend to use by setting the [`setup`](#setupopts) option `async_backend`, 
and acquire the corresponding backend by the following API:
```lua
local cacher_backend = require("vectorcode.config").get_cacher_backend()
```
and you can use `cacher_backend` wherever you used to use
`require("vectorcode.cacher")`. For example, `require("vectorcode.cacher").query_from_cache(0)` 
becomes `require("vectorcode.config").get_cacher_backend().query_from_cache(0)`.
In the remaining section of this documentation, I'll use `cacher_backend` to
represent either of the backends. Unless otherwise noticed, all the asynchronous APIs 
work for both backends.

#### `cacher_backend.register_buffer(bufnr?, opts?)`
This function registers a buffer to be cached by VectorCode.

```lua
cacher_backend.register_buffer(0, {
    n_query = 1,
})
```

The following are the available options for this function:
- `bufnr`: buffer number. Default: current buffer;
- `opts`: accepts a lua table with the following keys:
  - `project_root`: a string of the path that overrides the detected project root. 
  Default: `nil`. This is mostly intended to use with the [user command](#vectorcode-register), 
  and you probably should not use this directly in your config. **If you're
  using the LSP backend and did not specify this value, it will be automatically 
  detected based on `.vectorcode` or `.git`. If this fails, LSP backend will not 
  work**;
  - `exclude_this`: whether to exclude the file you're editing. Default: `true`;
  - `n_query`: number of retrieved documents. Default: `1`;
  - `debounce`: debounce time in milliseconds. Default: `10`;
  - `notify`: whether to show notifications when a query is completed. Default: `false`;
  - `query_cb`: `fun(bufnr: integer):string|string[]`, a callback function that accepts 
    the buffer ID and returns the query message(s). Default: 
    `require("vectorcode.utils").make_surrounding_lines_cb(-1)`. See 
    [this section](#built-in-query-callbacks) for a list of built-in query callbacks;
  - `events`: list of autocommand events that triggers the query. Default: `{"BufWritePost", "InsertEnter", "BufReadPost"}`;
  - `run_on_register`: whether to run the query when the buffer is registered.
    Default: `false`;
  - `single_job`: boolean. If this is set to `true`, there will only be one running job
    for each buffer, and when a new job is triggered, the last-running job will be
    cancelled. Default: `false`.


#### `cacher_backend.query_from_cache(bufnr?)`
This function queries VectorCode from cache.

```lua
local query_results = cacher_backend.query_from_cache(0, {notify=false})
```

The following are the available options for this function:
- `bufnr`: buffer number. Default: current buffer;
- `opts`: accepts a lua table with the following keys:
  - `notify`: boolean, whether to show notifications when a query is completed. Default:
    `false`;

Return value: an array of results. Each item of the array is in the format of 
`{path="path/to/your/code.lua", document="document content"}`.

#### `cacher_backend.async_check(check_item?, on_success?, on_failure?)`
This function checks if VectorCode has been configured properly for your project.

```lua 
cacher_backend.async_check(
    "config", 
    do_something(), -- on success
    do_something_else()  -- on failure
)
```

The following are the available options for this function:
- `check_item`: any check that works with `vectorcode check` command. If not set, 
  it defaults to `"config"`;
- `on_success`: a callback function that is called when the check passes;
- `on_failure`: a callback function that is called when the check fails.

#### `cacher_backend.buf_is_registered(bufnr?)`
This function checks if a buffer has been registered with VectorCode.

The following are the available options for this function:
- `bufnr`: buffer number. Default: current buffer.
Return value: `true` if registered, `false` otherwise.

#### `cacher_backend.buf_is_enabled(bufnr?)`
This function checks if a buffer has been enabled with VectorCode. It is slightly
different from `buf_is_registered`, because it does not guarantee VectorCode is actively
caching the content of the buffer. It is the same as `buf_is_registered && not is_paused`.

The following are the available options for this function:
- `bufnr`: buffer number. Default: current buffer.
Return value: `true` if enabled, `false` otherwise.

#### `cacher_backend.buf_job_count(bufnr?)`
Returns the number of running jobs in the background.

#### `cacher_backend.make_prompt_component(bufnr?, component_cb?)`
Compile the retrieval results into a string.
Parameters:
- `bufnr`: buffer number. Default: current buffer;
- `component_cb`: a callback function that formats each retrieval result, so
  that you can customise the control token, etc. for the component. The default
  is the following:
```lua
function(result)
    return "<|file_sep|>" .. result.path .. "\n" .. result.document
end
```

`make_prompt_component` returns a table with 2 keys:
- `count`: number of retrieved documents;
- `content`: The retrieval results concatenated together into a string. Each
  result is formatted by `component_cb`.

#### Built-in Query Callbacks

When using async cache, the query message is constructed by a function that
takes the buffer ID as the only parameter, and return a string or a list of
strings. The `vectorcode.utils` module provides the following callback
constructor for you to play around with it, but you can easily build your own!

- `require("vectorcode.utils").make_surrounding_lines_cb(line_count)`: returns a
  callback that uses `line_count` lines around the cursor as the query. When
  `line_count` is negative, it uses the full buffer;
- `require("vectorcode.utils").make_lsp_document_symbol_cb()`: returns a
  callback which uses the `textDocument/documentSymbol` method to retrieve a
  list of symbols in the current document. This will fallback to
  `make_surrounding_lines_cb(-1)` when there's no LSP that supports the
  `documentSymbol` method;
- `require("vectorcode.utils").make_changes_cb(max_num)`: returns a callback
  that fetches `max_num` unique items from the `:changes` list. This will also
  fallback to `make_surrounding_lines_cb(-1)`. The default value for `max_num`
  is 50.

## Debugging and Logging

You can enable logging by setting `VECTORCODE_NVIM_LOG_LEVEL` environment
variable to a 
[supported log level](https://github.com/nvim-lua/plenary.nvim/blob/857c5ac632080dba10aae49dba902ce3abf91b35/lua/plenary/log.lua#L44). 
The log file will be written to `stdpath("log")` or `stdpath("cache")`. On
Linux, this is usually `~/.local/state/nvim/`.
