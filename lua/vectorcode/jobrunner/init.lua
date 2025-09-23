local utils = require("vectorcode.utils")

---@alias VectorCode.JobRunner.Callback fun(result: table|nil, error: table|nil, code:integer, signal: integer?)

--- A class for calling vectorcode commands that aims at providing a unified API for both LSP and command-line backend.
--- Implementations exist for both direct command-line execution (`cmd.lua`) and LSP (`lsp.lua`).
--- For the format of the `result`, see https://github.com/Davidyz/VectorCode/blob/main/docs/cli.md#for-developers
---@class VectorCode.JobRunner
--- Runs a vectorcode command asynchronously.
--- Executes the command specified by `args`. Upon completion, if `callback` is provided,
--- it's invoked with the following arguments:
--- - `result`: the JSON object of the command execution result.
--- - `error`: error messages, if any.
--- - `code`: exit code (or error code) for the process.
--- - `signal`: _for cmd runner only_, the shell signal sent to the process.
--- The `bufnr` is used for context, potentially to find the project root or attach LSP clients.
--- Returns a job handle (e.g., PID or LSP request ID) or nil if the job couldn't be started.
---@field run_async fun(args: string[], callback:VectorCode.JobRunner.Callback?, bufnr: integer):(job_handle:integer?)
--- Runs a vectorcode command synchronously, blocking until completion or timeout.
--- Executes the command specified by `args`. Waits for up to `timeout_ms` milliseconds.
--- The `bufnr` is used for context, potentially to find the project root or attach LSP clients.
--- Returns the following objects:
--- - `result`: the JSON object of the command execution result.
--- - `error`: error messages, if any.
--- - `code`: exit code (or error code) for the process.
--- - `signal`: _for cmd runner only_, the shell signal sent to the process.
---@field run fun(args: string[], timeout_ms: integer?, bufnr: integer):(result:table|nil, error:table|nil, code:integer, signal: integer?)
--- Checks if a job associated with the given handle is currently running.
--- Returns true if the job is running, false otherwise.
---@field is_job_running fun(job_handle: integer):boolean
--- Attempts to stop or cancel the job associated with the given handle.
---@field stop_job fun(job_handle: integer)
--- Optional initialization function. Some runners (like LSP) might require an initialization step.
---@field init function?

return {
  --- Automatically find project_root from buffer path if it's not already specified.
  ---@param args string[]
  ---@param bufnr integer
  ---@return string[]
  find_root = function(args, bufnr)
    if not vim.list_contains(args, "--project_root") then
      local find_root = utils.find_root(bufnr)
      if find_root then
        vim.list_extend(args, { "--project_root", find_root })
      end
    end
    return args
  end,
}
