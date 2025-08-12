---@module "codecompanion"

---Type definition of the retrieval result.
---@class VectorCode.QueryResult
---@field path string Path to the file
---@field document string? Content of the file
---@field chunk string?
---@field start_line integer?
---@field end_line integer?
---@field chunk_id string?
---@field summary string? Used by the CodeCompanion tool only. Not part of the backend response

---@class VectorCode.LsResult
---@field project-root string

---@class VectorCode.VectoriseResult
---@field add integer
---@field update integer
---@field removed integer
---@field skipped integer
---@field failed integer

---Type definitions for the cache of a buffer.
---@class VectorCode.Cache
---@field enabled boolean Whether the async jobs are enabled or not. If the buffer is disabled, no cache will be generated for it.
---@field job_count integer
---@field jobs table<integer, integer> Job handle:time of creation (in seconds)
---@field last_run integer? Last time the query ran, in seconds from epoch.
---@field options VectorCode.RegisterOpts The options that the buffer was registered with.
---@field retrieval VectorCode.QueryResult[]? The latest retrieval.

---Type definitions for options accepted by `query` API.
---@class VectorCode.QueryOpts
---@field exclude_this boolean? Whether to exclude the current buffer. Default: true
---@field n_query integer? Number of results.
---@field notify boolean? Notify on new results and other key moments.
---@field timeout_ms number? Timeout (in milliseconds) for running a vectorcode command. Default: 5000

---@class VectorCode.OnSetup Some actions that may be configured to run when `setup` is called.
---@field update boolean `vectorcode update`
---@field lsp boolean whether to start LSP server on startup (default is to delay it to the first LSP request)

---@class VectorCode.CliCmds Cli commands to use
---@field vectorcode string vectorcode cli command or full path

---Options passed to `setup`.
---@class VectorCode.Opts : VectorCode.QueryOpts
---@field async_opts VectorCode.RegisterOpts Default options to use for registering a new buffer for async cache.
---@field cli_cmds VectorCode.CliCmds
---@field on_setup VectorCode.OnSetup
---@field async_backend "default"|"lsp"
---@field sync_log_env_var boolean Whether to automatically set `VECTORCODE_LOG_LEVEL` when `VECTORCODE_NVIM_LOG_LEVEL` is detected. !! WARNING: THIS MAY RESULT IN EXCESSIVE LOG MESSAGES DUE TO STDERR BEING POPULATED BY CLI LOGS

---Options for the registration of an async cache for a buffer.
---@class VectorCode.RegisterOpts: VectorCode.QueryOpts
---@field debounce integer? Seconds. Default: 10
---@field events string|string[]|nil autocmd events that triggers async jobs. Default: `{"BufWritePost", "InsertEnter", "BufReadPost"}`
---@field single_job boolean? Whether to restrict to 1 async job per buffer. Default: false
---@field query_cb VectorCode.QueryCallback? Function that accepts the buffer ID and returns the query message(s). Default: `require("vectorcode.utils").make_surrounding_lines_cb(-1)`
---@field run_on_register boolean? Whether to run the query when registering. Default: false
---@field project_root string?

---A unified interface used by `lsp` backend and `default` backend
---@class VectorCode.CacheBackend
---@field register_buffer fun(bufnr: integer?, opts: VectorCode.RegisterOpts) Register a buffer and create an async cache for it.
---@field deregister_buffer fun(bufnr: integer?, opts: {notify: boolean}?) Deregister a buffer and destroy its async cache.
---@field query_from_cache fun(bufnr: integer?, opts: {notify: boolean}?): VectorCode.QueryResult[] Get the cached documents.
---@field buf_is_registered fun(bufnr: integer?): boolean Checks if a buffer has been registered.
---@field buf_job_count fun(bufnr: integer?): integer Returns the number of running jobs in the background.
---@field buf_is_enabled fun(bufnr: integer?): boolean Checks if a buffer has been enabled.
---@field make_prompt_component fun(bufnr: integer?, component_cb: (fun(result: VectorCode.QueryResult): string)?): {content: string, count: integer} Compile the retrieval results into a string.
---@field async_check fun(check_item: string?, on_success: fun(out: vim.SystemCompleted)?, on_failure: fun(out: vim.SystemCompleted)?) Checks if VectorCode has been configured properly for your project.

--- This class defines the options available to the CodeCompanion tool.
---@class VectorCode.CodeCompanion.ToolOpts
--- Whether to use the LSP backend. Default: `false`
---@field use_lsp boolean?
---@field requires_approval boolean?
--- Whether this tool should be included in `vectorcode_toolbox`
---@field include_in_toolbox boolean?

---@class VectorCode.CodeCompanion.LsToolOpts: VectorCode.CodeCompanion.ToolOpts

---@class VectorCode.CodeCompanion.FilesLsToolOpts: VectorCode.CodeCompanion.ToolOpts

---@class VectorCode.CodeCompanion.FilesRmToolOpts: VectorCode.CodeCompanion.ToolOpts

---@class VectorCode.CodeCompanion.QueryToolOpts: VectorCode.CodeCompanion.ToolOpts
--- Maximum number of results provided to the LLM.
--- You may set this to a table to configure different values for document/chunk mode.
--- When set to negative values, it means unlimited.
--- Default: `{ document = -1, chunk = -1 }`
---@field max_num integer|{document:integer, chunk: integer}|nil
--- Default number of results provided to the LLM.
--- This value is written in the system prompt and tool description.
--- Users may ask the LLM to request a different number of results in the chat.
--- You may set this to a table to configure different values for document/chunk mode.
--- Default: `{ document = 10, chunk = 50 }`
---@field default_num integer|{document:integer, chunk: integer}|nil
--- Whether to avoid duplicated references. Default: `true`
---@field no_duplicate boolean?
--- Whether to send chunks instead of full files to the LLM. Default: `false`
--- > Make sure you adjust `max_num` and `default_num` accordingly.
---@field chunk_mode boolean?
---@field summarise VectorCode.CodeCompanion.SummariseOpts?

---@class VectorCode.CodeCompanion.VectoriseToolOpts: VectorCode.CodeCompanion.ToolOpts

---@class VectorCode.CodeCompanion.ToolGroupOpts
--- Whether to register the tool group
---@field enabled boolean
--- Whether to show the individual tools in the references
---@field collapse boolean
--- Other tools that you'd like to include in `vectorcode_toolbox`
---@field extras string[]

--- The result of the query tool should be structured in the following table
---@class VectorCode.CodeCompanion.QueryToolResult
---@field raw_results VectorCode.QueryResult[]
---@field count integer
---@field summary string|nil

---@class VectorCode.CodeCompanion.SummariseOpts
---A boolean flag that controls whether summarisation should be enabled.
---This can also be a function that returns a boolean.
---In this case, you can use this option to dynamically control whether summarisation is enabled during a chat.
---
---This function recieves 2 parameters:
--- - `CodeCompanion.Chat`: the chat object;
--- - `VectorCode.QueryResult[]`: a list of query results.
---@field enabled boolean|(fun(chat: CodeCompanion.Chat, results: VectorCode.QueryResult[]):boolean)|nil
---The adapter used for the summarisation task. When set to `nil`, the adapter from the current chat will be used.
---@field adapter string|CodeCompanion.Adapter|nil
---The system prompt sent to the summariser model.
---When set to a function, it'll recieve the default system prompt as the only parameter,
---and should return the new (full) system prompt. This allows you to customise or rewrite the system prompt.
---@field system_prompt string|(fun(original_prompt: string): string)
---When set to true, include the query messages so that the LLM may make task-related summarisations.
---This happens __after__ the `system_prompt` callback processing
---@field query_augmented boolean
