---Type definition of the retrieval result.
---@class VectorCode.Result
---@field path string Path to the file
---@field document string Content of the file

---Type definitions for the cache of a buffer.
---@class VectorCode.Cache
---@field enabled boolean Whether the async jobs are enabled or not. If the buffer is disabled, no cache will be generated for it.
---@field job_count integer
---@field jobs table<integer, boolean> keeps track of running jobs.
---@field last_run integer? Last time the query ran, in seconds from epoch.
---@field options VectorCode.RegisterOpts The options that the buffer was registered with.
---@field retrieval VectorCode.Result[]? The latest retrieval.

---Type definitions for options accepted by `query` API.
---@class VectorCode.QueryOpts
---@field exclude_this boolean Whether to exclude the current buffer. Default: true
---@field n_query integer Number of results.
---@field notify boolean Notify on new results and other key moments.
---@field timeout_ms number Timeout (in milliseconds) for running a vectorcode command. Default: 5000

---@class VectorCode.OnSetup Some actions that may be configured to run when `setup` is called.
---@field update boolean `vectorcode update`

---Options passed to `setup`.
---@class VectorCode.Opts : VectorCode.QueryOpts
---@field async_opts VectorCode.RegisterOpts Default options to use for registering a new buffer for async cache.
---@field on_setup VectorCode.OnSetup
---@field async_backend "default"|"lsp"

---Options for the registration of an async cache for a buffer.
---@class VectorCode.RegisterOpts: VectorCode.QueryOpts
---@field debounce integer Seconds. Default: 10
---@field events string|string[] autocmd events that triggers async jobs. Default: `{"BufWritePost", "InsertEnter", "BufReadPost"}`
---@field single_job boolean Whether to restrict to 1 async job per buffer. Default: false
---@field query_cb VectorCode.QueryCallback Function that accepts the buffer ID and returns the query message(s). Default: `require("vectorcode.utils").make_surrounding_lines_cb(-1)`
---@field run_on_register boolean Whether to run the query when registering. Default: false
---@field project_root string?

---A unified interface used by `lsp` backend and `default` backend
---@class VectorCode.CacheBackend
---@field register_buffer fun(bufnr: integer?, opts: VectorCode.RegisterOpts) Register a buffer and create an async cache for it.
---@field deregister_buffer fun(bufnr: integer?, opts: {notify: boolean}?) Deregister a buffer and destroy its async cache.
---@field query_from_cache fun(bufnr: integer?, opts: {notify: boolean}?): VectorCode.Result[] Get the cached documents.
---@field buf_is_registered fun(bufnr: integer?): boolean Checks if a buffer has been registered.
---@field buf_job_count fun(bufnr: integer?): integer Returns the number of running jobs in the background.
---@field buf_is_enabled fun(bufnr: integer?): boolean Checks if a buffer has been enabled.
---@field make_prompt_component fun(bufnr: integer?, component_cb: fun(result: VectorCode.Result): string?): {content: string, count: integer} Compile the retrieval results into a string.
---@field async_check fun(check_item: string?, on_success: fun(out: vim.SystemCompleted)?, on_failure: fun(out: vim.SystemCompleted)?) Checks if VectorCode has been configured properly for your project.
