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

---@class VectorCode.CodeCompanion.ToolOpts
---@field max_num integer?
---@field default_num integer?
---@field include_stderr boolean?
---@field use_lsp boolean?
---@field auto_submit table<string, boolean>?
---@field ls_on_start boolean?
---@field no_duplicate boolean?
---@field summarise VectorCode.CodeCompanion.SummariseOpts?

---@type VectorCode.CodeCompanion.ToolOpts
local DEFAULT_TOOL_OPTS = {
  max_num = -1,
  default_num = 10,
  include_stderr = false,
  use_lsp = false,
  auto_submit = { ls = false, query = false },
  ls_on_start = false,
  no_duplicate = true,
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
return {
  tool_result_source = "VectorCodeToolResult",

  ---@param t table|string
  ---@return string
  flatten_table_to_string = function(t)
    if type(t) == "string" then
      return t
    end
    return table.concat(vim.iter(t):flatten(math.huge):totable(), "\n")
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

  ---@param result VectorCode.Result
  ---@param summarise_opts VectorCode.CodeCompanion.SummariseOpts
  ---@return string
  process_documents = function(result, summarise_opts)
    ---@type string?
    local processed_result
    if summarise_opts.enabled then
      -- TODO: implement summarisation logics here.
      -- The summary should be stored in `process_result` as a string.
    end
    if processed_result == nil then
      processed_result = string.format(
        [[Here is a file the VectorCode tool retrieved:
<path>
%s
</path>
<content>
%s
</content>
]],
        result.path,
        result.document
      )
    end
    return processed_result
  end,
  get_default_tool_opts = function()
    return vim.deepcopy(DEFAULT_TOOL_OPTS)
  end,
}
