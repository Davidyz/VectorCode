# NeoVim Plugin
> This plugin depends on the CLI tool. Please go through [the CLI
> documentation](./cli.md) and make sure you know how to use VectorCode from the
> CLI before proceeding.

> [!NOTE]
> When the neovim plugin doesn't work properly, please try upgrading the CLI
> tool to the latest version before opening an issue.


<!-- mtoc-start -->

* [Installation](#installation)
* [Configuration](#configuration)
  * [`setup(opts?)`](#setupopts)
* [Usage](#usage)
  * [Synchronous API](#synchronous-api)
    * [`query(query_message, opts?)`](#queryquery_message-opts)
    * [`check(check_item?)`](#checkcheck_item)
  * [Cached Asynchronous API](#cached-asynchronous-api)
    * [`register_buffer(bufnr?, opts?)`](#register_bufferbufnr-opts)
    * [`query_from_cache(bufnr?)`](#query_from_cachebufnr)
    * [`lualine()`](#lualine)
    * [`async_check(check_item?, on_success?, on_failure?)`](#async_checkcheck_item-on_success-on_failure)
    * [`buf_is_registered(bufnr?)`](#buf_is_registeredbufnr)

<!-- mtoc-end -->

## Installation
Use your favourite plugin manager. 

```lua 
{
  "Davidyz/VectorCode",
  dependencies = { "nvim-lua/plenary.nvim" },
}
```

## Configuration

### `setup(opts?)`
This function initialises the VectorCode client and sets up some default

```lua
require("vectorcode").setup({
    n_query = 1,
})
```

The following are the available options for this function:
- `n_query`: number of retrieved documents. Default: `1`;
- `notify`: whether to show notifications when a query is completed.
  Default: `true`;
- `timeout_ms`: timeout in milliseconds for the query operation. Default: 
  `5000` (5 seconds).

## Usage
This plugin provides 2 sets of APIs that provides similar functionalities. The
synchronous APIs provide more up-to-date retrieval results at the cost of
blocking the main neovim UI, while the async APIs use a caching mechanism to 
provide asynchronous retrieval results almost instantaneously, but the result
may be slightly out-of-date. For some tasks like chat, the main UI being
blocked/frozen doesn't hurt much because you spend the time waiting for response
anyway, and you can use the synchronous API in this case. For other tasks like 
completion, the async API will minimise the interruption to your workflow.


### Synchronous API
#### `query(query_message, opts?)`
This function queries VectorCode and returns an array of results.

```lua
require("vectorcode").query("some query message", {
    n_query = 5,
})
```

The following are the available options for this function:
- `n_query`: number of retrieved documents. Default: `1`;
- `notify`: whether to show notifications when a query is completed.
  Default: `true`;
- `timeout_ms`: timeout in milliseconds for the query operation. Default: 
  `5000` (5 seconds).
The return value of this function is an array of results in the format of
`{path="path/to/your/code.lua", document="document content"}`. 

For example, in [cmp-ai](https://github.com/tzachar/cmp-ai), you can add 
the path/document content to the prompt like this:
```lua
function prompt(prefix, suffix)
    local retrieval_results = require("vectorcode").query("some query message", {
        n_query = 5,
    })
    for _, source in pairs(retrieval_results) do
        -- This works for qwen2.5-coder.
        file_context = file_context
            .. "<|file_sep|>"
            .. "path/to/your/code.lua"
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
This function checks if VectorCode has been configured properly for your project.

```lua 
require("vectorcode").check()
```

The following are the available options for this function:
- `check_item`: one of `"config"`, `"db"` and `"all"`. If not set, it defaults
  to `"all"`.
Return value: `true` if passed, `false` if failed.

This involves the `check` command of the CLI that checks the status of the
VectorCode project setup. Use this as a pre-condition of any subsequent
use of other VectorCode APIs that may be more expensive (if this fails,
VectorCode hasn't been properly set up for the project, and you should not use
VectorCode APIs).

The use of this API is entirely optional. You can totally ignore this and call
`query` anyway, but if `check` fails, you might be spending the waiting time for
nothing.

### Cached Asynchronous API

The async cache mechanism helps mitigate the issue where the `query` API may
take too long and block the main thread.

#### `register_buffer(bufnr?, opts?)`
This function registers a buffer to be cached by VectorCode.

```lua
require("vectorcode").register_buffer({
    n_query = 1,
})
```

The following are the available options for this function:
- `bufnr`: buffer number. Default: current buffer;
- `opts`: same as the `setup` function above. This API will create an autocommand 
  that triggers a query when you write to the file, enter insert mode or read from 
  the file;
- `query_cb`: a callback function that returns the query message. Default: 
  `surrounding_lines_cb(-1)`, which queries the full buffer;
- `events`: a list of events to trigger the query. Default:
  `{"BufWritePost", "InsertEnter", "BufReadPost"}`;
- `debounce`: debounce time in seconds for the query. Default: `10`.

#### `query_from_cache(bufnr?)`
This function queries VectorCode from cache.

```lua
require("vectorcode.cacher").query_from_cache()
```

The following are the available options for this function:
- `bufnr`: buffer number. Default: current buffer.

Return value: an array of results in the format of 
`{path="path/to/your/code.lua", document="document content"}`.

#### `lualine()`
This function returns a lualine component that displays the status of VectorCode
for the current buffer. 

```lua
require("vectorcode.cacher").lualine()
```

#### `async_check(check_item?, on_success?, on_failure?)`
This function checks if VectorCode has been configured properly for your project.

```lua 
require("vectorcode.cacher").async_check(
    "config", 
    do_something(),
    do_something_else()
)
```

The following are the available options for this function:
- `check_item`: any check that works with `vectorcode check` command. If not set, 
  it defaults to `"config"`;
- `on_success`: a callback function that is called when the check passes;
- `on_failure`: a callback function that is called when the check fails.

#### `buf_is_registered(bufnr?)`
This function checks if a buffer has been registered with VectorCode.

```lua 
require("vectorcode.cacher").buf_is_registered()
```

The following are the available options for this function:
- `bufnr`: buffer number. Default: current buffer.
Return value: `true` if registered, `false` otherwise.
