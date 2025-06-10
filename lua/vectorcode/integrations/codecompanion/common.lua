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
