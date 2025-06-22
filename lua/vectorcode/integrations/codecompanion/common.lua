---@module "codecompanion"

local job_runner
local vc_config = require("vectorcode.config")
local cc_config = require("codecompanion.config").config
local notify_opts = vc_config.notify_opts
local logger = vc_config.logger
local http_client = require("codecompanion.http")

---@class VectorCode.CodeCompanion.SummariseOpts
---@field enabled boolean?
---@field adapter string|CodeCompanion.Adapter|nil
---@field threshold integer?
---@field system_prompt string

---@type VectorCode.CodeCompanion.QueryToolOpts
local default_query_options = {
  max_num = { chunk = -1, document = -1 },
  default_num = { chunk = 50, document = 10 },
  no_duplicate = true,
  chunk_mode = false,
  summarise = {
    enabled = false,
    system_prompt = [[
You are an experienced code analyser.
Your task is to write summaries of source code that are informative and concise.
The summary will serve as a source of information for others to quickly understand how the code works and how to work with the code without going through the source code
Your summary should include the following information:
- variables, functions, classes and other objects that are importable/includeable by other programs
- for a function or method, include its signature and high-level implementation details. For example,
  - when summarising a sorting function, include the sorting algorithm, the parameter types and return types
  - when summarising a function that makes an http request, include the network library used by the function, the parameter types and return types
- if the code contains syntax or semantics errors, include them as well.
- for anything that you quote from the source code, include the line numbers from which you're quote them.
- DO NOT include local variables and functions that are not accessible by other functions.
]],
  },
}

---@type VectorCode.CodeCompanion.LsToolOpts
local default_ls_options = {}

---@type VectorCode.CodeCompanion.VectoriseToolOpts
local default_vectorise_options = {}

local TOOL_RESULT_SOURCE = "VectorCodeToolResult"

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
  ---@return string
  process_result = function(result)
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
    return llm_message
  end,

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
