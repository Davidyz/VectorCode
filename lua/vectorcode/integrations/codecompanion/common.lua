local job_runner
local vc_config = require("vectorcode.config")
local notify_opts = vc_config.notify_opts
local logger = vc_config.logger

---@class VectorCode.CodeCompanion.ToolOpts
---@field max_num integer?
---@field default_num integer|{document:integer, chunk: integer}|nil
---@field include_stderr boolean?
---@field use_lsp boolean?
---@field auto_submit table<string, boolean>?
---@field ls_on_start boolean?
---@field no_duplicate boolean?
---@field chunk_mode boolean?

---@type VectorCode.CodeCompanion.ToolOpts
local default_options = {
  max_num = -1,
  default_num = 10,
  include_stderr = false,
  use_lsp = false,
  auto_submit = { ls = false, query = false },
  ls_on_start = false,
  no_duplicate = true,
  only_chunks = false,
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

  ---@param opts VectorCode.CodeCompanion.ToolOpts|{}|nil
  ---@return VectorCode.CodeCompanion.ToolOpts
  get_tool_opts = function(opts)
    if opts == nil or opts.use_lsp == nil then
      opts = vim.tbl_deep_extend(
        "force",
        opts or {},
        { use_lsp = vc_config.get_user_config().async_backend == "lsp" }
      )
    end
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
    return vim.tbl_deep_extend("force", default_options, opts)
  end,

  ---@param result VectorCode.Result
  ---@return string
  process_result = function(result)
    if result.chunk then
      -- chunk mode
      local chunk =
        string.format("<path>%s</path><chunk>%s</chunk>", result.path, result.chunk)
      if result.start_line and result.end_line then
        chunk = chunk
          .. string.format(
            "<start_line>%d</start_line><end_line>%d</end_line>",
            result.start_line,
            result.end_line
          )
      end
      return chunk
    else
      -- full document mode
      return string.format(
        "<path>%s</path><content>%s</content>",
        result.path,
        result.document
      )
    end
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
