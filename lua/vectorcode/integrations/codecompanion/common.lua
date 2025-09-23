---@module "codecompanion"

local job_runner
local vc_config = require("vectorcode.config")
local notify_opts = vc_config.notify_opts
local logger = vc_config.logger

local TOOL_RESULT_SOURCE = "VectorCodeToolResult"

return {
  tool_result_source = TOOL_RESULT_SOURCE,

  ---@param t table|string|nil
  ---@return string
  flatten_table_to_string = function(t)
    if t == nil then
      return ""
    end
    if type(t) == "string" then
      return t
    end

    -- Handle empty tables or tables with empty strings
    local flattened = vim
      .iter(t)
      :flatten(math.huge)
      :filter(function(item)
        return type(item) == "string" and vim.trim(item) ~= ""
      end)
      :totable()

    if #flattened == 0 then
      return "Unknown error occurred"
    end

    return table.concat(flattened, "\n")
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

  ---Convert `path` to a relative path if it's within the current project.
  ---When `base` is `nil`, this function will attempt to find a project root
  ---or use `cwd`.
  ---@param path string
  ---@param base? string
  ---@return string
  cleanup_path = function(path, base)
    base = base or vim.fs.root(0, { ".vectorcode", ".git" }) or vim.uv.cwd() or "."
    return vim.fs.relpath(base, path) or path
  end,
}
