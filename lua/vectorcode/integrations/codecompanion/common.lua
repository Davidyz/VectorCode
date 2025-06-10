---@module "codecompanion"

local job_runner
local vc_config = require("vectorcode.config")
local cc_config = require("codecompanion.config").config
local cc_schema = require("codecompanion.schema")
local notify_opts = vc_config.notify_opts
local logger = vc_config.logger
local http_client = require("codecompanion.http")

---@class VectorCode.CodeCompanion.SummariseOpts
---@field enabled boolean?
---@field adapter string|CodeCompanion.Adapter|nil
---@field threshold integer?
---@field system_prompt string
---@field timeout integer

--- Suspends the current coroutine until a given amount of time has passed
--- or an interrupt condition is met.
---
--- @param ms number The maximum number of milliseconds to sleep.
--- @param interrupt_fn? fun(): boolean An optional function that is checked periodically.
---   If it returns true, the sleep is interrupted.
--- @return boolean interrupted True if the sleep was interrupted, false otherwise.
local function sleep(ms, interrupt_fn)
  local co = coroutine.running()
  if not co then
    error("sleep() must be called from within a coroutine.", 2)
  end

  local main_timer = vim.uv.new_timer()
  local check_timer

  local function cleanup()
    main_timer:stop()
    main_timer:close()
    if check_timer then
      check_timer:stop()
      check_timer:close()
    end
  end

  local function resume(interrupted)
    if co and coroutine.status(co) == "suspended" then
      vim.schedule(function()
        coroutine.resume(co, interrupted)
      end)
    end
  end

  main_timer:start(ms, 0, function()
    cleanup()
    resume(false)
  end)

  if interrupt_fn then
    check_timer = vim.uv.new_timer()
    check_timer:start(0, 10, function()
      if interrupt_fn() then
        cleanup()
        resume(true)
      end
    end)
  end

  return coroutine.yield()
end

---@type VectorCode.CodeCompanion.QueryToolOpts
local default_query_options = {
  max_num = { chunk = -1, document = -1 },
  default_num = { chunk = 50, document = 10 },
  no_duplicate = true,
  chunk_mode = false,
  summarise = {
    enabled = false,
    timeout = 5000,
    system_prompt = [[You are an expert and experienced code analyzer and summarizer. Your primary task is to analyze provided source code and generate a comprehensive, well-structured Markdown summary. This summary will serve as a concise source of information for others to quickly understand how the code works and how to interact with it, without needing to delve into the full source code. Adhere strictly to the following formatting and content guidelines:

Markdown Structure:

    Top-Level Header (#): The absolute file path of the source code.

    Secondary Headers (##): For each top-level symbol (e.g., functions, classes, global variables) defined directly within the source code file that are importable or includable by other programs.

    Tertiary Headers (###): For symbols nested one level deep within a secondary header's symbol (e.g., methods within a class, inner functions).

    Quaternary Headers (####): For symbols nested two levels deep (e.g., a function defined within a method of a class).

    Continue this pattern, incrementing the header level for each deeper level of nesting.

Content for Each Section:

    Descriptive Summary: Each header section (from secondary headers downwards) must contain a concise and informative summary of the symbol defined by that header.

        For Functions/Methods: Explain their purpose, parameters (including types), return values (including types), high-level implementation details, and any significant side effects or core logic. For example, if summarizing a sorting function, include the sorting algorithm used. If summarizing a function that makes an HTTP request, mention the network library employed.

        For Classes: Describe the class's role, its main responsibilities, and key characteristics.

        For Variables (global or within scope): State their purpose, type (if discernible), and initial value or common usage.

        For Modules/Files (under the top-level header): Provide an overall description of the file's purpose, its main components, and its role within the larger project (if context is available).

General Guidelines:

    Clarity and Conciseness: Summaries should be easy to understand, avoiding jargon where possible, and as brief as possible while retaining essential information.

    Accuracy: Ensure the summary accurately reflects the code's functionality.

    Focus on Public Interface/Behavior: Prioritize describing what a function/class does and how it's used. Only include details about symbols (variables, functions, classes) that are importable/includable by other programs. DO NOT include local variables and functions that are not accessible by other functions outside their immediate scope.

    No Code Snippets: Do not include any actual code snippets in the summary. Focus solely on descriptive text. If you need to refer to a specific element for context (e.g., in an error description), describe it and provide line numbers for reference from the source code.

    Syntax/Semantic Errors: If the code contains syntax or semantic errors, describe them clearly within the summary, indicating the nature of the error.

    Language Agnostic: Adapt the summary to the specific programming language of the provided source code (e.g., Python, JavaScript, Java, C++, etc.).

    Handle Edge Cases/Dependencies: If a symbol relies heavily on external dependencies or handles specific edge cases, briefly mention these if they are significant to its overall function.

    Information Source: There will be no extra information available to you. Provide the summary solely based on the provided file.
]],
  },
}

---@type VectorCode.CodeCompanion.LsToolOpts
local default_ls_options = {}

---@type VectorCode.CodeCompanion.VectoriseToolOpts
local default_vectorise_options = {}

local TOOL_RESULT_SOURCE = "VectorCodeToolResult"

---@alias ChatMessage {role: string, content:string}

---@param adapter CodeCompanion.Adapter
---@param system_prompt string
---@param user_messages string|string[]
---@return {messages: ChatMessage[], tools:table?}
local function make_oneshot_payload(adapter, system_prompt, user_messages)
  if type(user_messages) == "string" then
    user_messages = { user_messages }
  end
  local messages =
    { { role = cc_config.constants.SYSTEM_ROLE, content = system_prompt } }
  for _, m in pairs(user_messages) do
    table.insert(messages, { role = cc_config.constants.USER_ROLE, content = m })
  end
  return { messages = adapter:map_roles(messages) }
end

return {
  tool_result_source = TOOL_RESULT_SOURCE,

  ---@param t table|string
  ---@return string
  flatten_table_to_string = function(t)
    if type(t) == "string" then
      return t
    end
    return table.concat(vim.iter(t):flatten(math.huge):totable(), "\n")
  end,

  ---@param opts VectorCode.CodeCompanion.LsToolOpts|{}|nil
  ---@return VectorCode.CodeCompanion.LsToolOpts
  get_ls_tool_opts = function(opts)
    opts = vim.tbl_deep_extend("force", default_ls_options, opts or {})
    logger.info(
      string.format(
        "Loading `vectorcode_ls` with the following opts:\n%s",
        vim.inspect(opts)
      )
    )
    return opts
  end,

  ---@param opts VectorCode.CodeCompanion.VectoriseToolOpts|{}|nil
  ---@return VectorCode.CodeCompanion.VectoriseToolOpts
  get_vectorise_tool_opts = function(opts)
    opts = vim.tbl_deep_extend("force", default_vectorise_options, opts or {})
    logger.info(
      string.format(
        "Loading `vectorcode_vectorise` with the following opts:\n%s",
        vim.inspect(opts)
      )
    )
    return opts
  end,

  ---@param opts VectorCode.CodeCompanion.QueryToolOpts|{}|nil
  ---@return VectorCode.CodeCompanion.QueryToolOpts
  get_query_tool_opts = function(opts)
    if opts == nil or opts.use_lsp == nil then
      opts = vim.tbl_deep_extend(
        "force",
        opts or {},
        { use_lsp = vc_config.get_user_config().async_backend == "lsp" }
      )
    end
    opts = vim.tbl_deep_extend("force", default_query_options, opts)
    if type(opts.default_num) == "table" then
      if opts.chunk_mode then
        opts.default_num = opts.default_num.chunk
      else
        opts.default_num = opts.default_num.document
      end
      assert(
        type(opts.default_num) == "number",
        "default_num should be an integer or a table: {chunk: integer, document: integer}"
      )
    end
    if type(opts.max_num) == "table" then
      if opts._ then
        opts.max_num = opts.max_num.chunk
      else
        opts.max_num = opts.max_num.document
      end
      assert(
        type(opts.max_num) == "number",
        "max_num should be an integer or a table: {chunk: integer, document: integer}"
      )
    end
    logger.info(
      string.format(
        "Loading `vectorcode_query` with the following opts:\n%s",
        vim.inspect(opts)
      )
    )
    return opts
  end,

  ---@param result VectorCode.QueryResult
  ---@param summarise_opts VectorCode.CodeCompanion.SummariseOpts|{}|nil
  ---@param callback fun(summary:string)?
  ---@return string
  process_result = function(result, summarise_opts, callback)
    -- TODO: Unify the handling of summarised and non-summarised result
    local llm_message

    if result.chunk then
      -- chunk mode
      llm_message =
        string.format("<path>%s</path><chunk>%s</chunk>", result.path, result.chunk)
      if result.start_line and result.end_line then
        llm_message = llm_message
          .. string.format(
            "<start_line>%d</start_line><end_line>%d</end_line>",
            result.start_line,
            result.end_line
          )
      end
    else
      -- full document mode
      llm_message = string.format(
        "<path>%s</path><content>%s</content>",
        result.path,
        result.document
      )
    end

    summarise_opts =
      ---@cast summarise_opts VectorCode.CodeCompanion.SummariseOpts
      vim.tbl_deep_extend(
        "force",
        default_query_options.summarise,
        summarise_opts or {}
      )

    if summarise_opts.enabled and type(callback) == "function" then
      ---@type CodeCompanion.Adapter
      local adapter =
        vim.deepcopy(require("codecompanion.adapters").resolve(summarise_opts.adapter))

      local payload =
        make_oneshot_payload(adapter, summarise_opts.system_prompt, llm_message)
      local settings =
        vim.deepcopy(adapter:map_schema_to_params(cc_schema.get_default(adapter)))
      settings.opts.stream = false

      ---@type CodeCompanion.Client
      local client = http_client.new({ adapter = settings })
      client:request(payload, {
        ---@param _adapter CodeCompanion.Adapter
        callback = function(err, data, _adapter)
          if data then
            local res = _adapter.handlers.chat_output(_adapter, data)
            if res and res.status == "success" then
              local summary = vim.trim(res.output.content or "")
              if summary ~= "" then
                return callback(summary)
              end
            end
            return callback(llm_message)
          end
        end,
      }, { silent = true })
    end
    return llm_message
  end,

  async_sleep = sleep,

  ---@param use_lsp boolean
  ---@return VectorCode.JobRunner
  initialise_runner = function(use_lsp)
    if job_runner == nil then
      if use_lsp then
        job_runner = require("vectorcode.jobrunner.lsp")
      end
      if job_runner == nil then
        job_runner = require("vectorcode.jobrunner.cmd")
        logger.info("Using cmd runner for CodeCompanion tool.")
        if use_lsp then
          vim.schedule_wrap(vim.notify)(
            "Failed to initialise the LSP runner. Falling back to cmd runner.",
            vim.log.levels.WARN,
            notify_opts
          )
        end
      else
        logger.info("Using LSP runner for CodeCompanion tool.")
      end
    end
    return job_runner
  end,
}
