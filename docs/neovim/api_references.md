# Lua API References

This plugin provides 2 sets of _high-level APIs_ that provides similar functionalities. The
synchronous APIs provide more up-to-date retrieval results at the cost of
blocking the main neovim UI, while the async APIs use a caching mechanism to 
provide asynchronous retrieval results almost instantaneously, but the result
may be slightly out-of-date. For some tasks like chat, the main UI being
blocked/frozen doesn't hurt much because you spend the time waiting for response
anyway, and you can use the synchronous API in this case. For other tasks like 
completion, the cached API will minimise the interruption to your workflow, but
at a cost of providing less up-to-date results.

These APIs are wrappers around the _lower-level 
[job runner API](https://github.com/Davidyz/VectorCode/tree/main/lua/vectorcode/jobrunner)_, 
which provides a unified interface for calling VectorCode commands that can be
executed by either the LSP or the generic CLI backend. If the high-level APIs
are sufficient for your use-case, it's usually not necessary to use the job
runners directly.

<!-- mtoc-start -->

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
* [JobRunners](#jobrunners)
  * [`run_async(args, callback, bufnr)` and `run(args, timeout_ms, bufnr)`](#run_asyncargs-callback-bufnr-and-runargs-timeout_ms-bufnr)
  * [`is_job_running(job_handle):boolean`](#is_job_runningjob_handleboolean)
  * [`stop_job(job_handle)`](#stop_jobjob_handle)

<!-- mtoc-end -->

## Synchronous API
### `query(query_message, opts?, callback?)`
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
Keep in mind that this `query` function call will be synchronous and therefore
block the neovim UI. This is where the async cache comes in.

### `check(check_item?)`
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

### `update(project_root?)`
This function calls `vectorcode update` at the current working directory.
`--project_root` will be added if the `project_root` parameter is not `nil`.
This runs async and doesn't block the main UI.

```lua
require("vectorcode").update()
```

## Cached Asynchronous API

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
   `vim.lsp.config()`. This plugin will load the config associated with the name
   `vectorcode_server`. You can override the default config (for example, the
   path to the executable) by calling `vim.lsp.config('vectorcode_server', opts)`.


| Features | `default`                                                                                                 | `lsp`                                                                                                                     |
|----------|-----------------------------------------------------------------------------------------------------------|---------------------------------------------------------------------------------------------------------------------------|
| **Pros** | Fully backward compatible with minimal extra config required                                              | Less IO overhead for loading/unloading embedding models; Progress reports.                                                |
| **Cons** | Heavy IO overhead because the embedding model and database client need to be initialised for every query. | Requires `vectorcode-server` |

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

### `cacher_backend.register_buffer(bufnr?, opts?)`
This function registers a buffer to be cached by VectorCode.

```lua
cacher_backend.register_buffer(0, {
    n_query = 1,
})
```

The following are the available options for this function:
- `bufnr`: buffer number. Default: `0` (current buffer);
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


### `cacher_backend.query_from_cache(bufnr?)`
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

### `cacher_backend.async_check(check_item?, on_success?, on_failure?)`
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

### `cacher_backend.buf_is_registered(bufnr?)`
This function checks if a buffer has been registered with VectorCode.

The following are the available options for this function:
- `bufnr`: buffer number. Default: current buffer.
Return value: `true` if registered, `false` otherwise.

### `cacher_backend.buf_is_enabled(bufnr?)`
This function checks if a buffer has been enabled with VectorCode. It is slightly
different from `buf_is_registered`, because it does not guarantee VectorCode is actively
caching the content of the buffer. It is the same as `buf_is_registered && not is_paused`.

The following are the available options for this function:
- `bufnr`: buffer number. Default: current buffer.
Return value: `true` if enabled, `false` otherwise.

### `cacher_backend.buf_job_count(bufnr?)`
Returns the number of running jobs in the background.

### `cacher_backend.make_prompt_component(bufnr?, component_cb?)`
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

### Built-in Query Callbacks

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


## JobRunners

The `VectorCode.JobRunner` is an abstract class for vectorcode command
execution. There are 2 concrete child classes that you can use: 
- `require("vectorcode.jobrunner.cmd")` uses the CLI (`vectorcode` commands) to
  interact with the database;
- `quire("vectorcode.jobrunner.lsp")` use the LSP server, which avoids some of
  the IO overhead and provides LSP progress notifications.

The available methods for a `VectorCode.JobRunner` object includes:

### `run_async(args, callback, bufnr)` and `run(args, timeout_ms, bufnr)`
Calls a vectorcode command.

The `args` parameter (of type `string[]`) is whatever argument that comes after
`vectorcode` when you run it in the CLI. For example, if you want to query for
10 chunks in the shell, you'd call the following command:

```bash
vectorcode query -n 10 keyword1 keyword2 --include chunk
```

Then for the job runner (either LSP or cmd), the `args` parameter would be:
```lua
args = {"query", "-n", "10", "keyword1", "keyword2", "--include", "chunk"}
```

For the `run_async` method, the `callback` function has the
following signature:
```lua
---@type fun(result: table, error: table, code:integer, signal: integer?)?
```
For the `run` method, the return value can be captured as follow:
```lua
res, err, _code, _signal = jobrunner.run(args, -1, 0)
```

The result (for both synchronous and asynchronous method) is a `vim.json.decode`ed 
table of the result of the command execution. Consult 
[the CLI documentation](../cli.md#for-developers) for the schema of the results for 
the command that you call.

For example, the query command mentioned above will return a
`VectorCode.QueryResult[]`, where `VectorCode.QueryResult` is defined as
follows:
```lua
---@class VectorCode.QueryResult
---@field path string Path to the file
---@field document string? Content of the file
---@field chunk string?
---@field start_line integer?
---@field end_line integer?
---@field chunk_id string?
```

The `run_async` will return a `job_handle` which is defined as an `integer?`.
For the LSP backend, the job handle is the `request_id`. For the cmd runner, the
job handle is the `PID` of the process.

### `is_job_running(job_handle):boolean`
Checks if a job associated with the given handle is currently running;


### `stop_job(job_handle)`
Attempts to stop or cancel the async job associated with the given handle.
