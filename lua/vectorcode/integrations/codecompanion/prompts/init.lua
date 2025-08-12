local M = {}

local vc_config = require("vectorcode.config")

---@param path string[]|string path to files or wildcards.
---@param project_root? string
---@param callback? VectorCode.JobRunner.Callback
function M.vectorise_files(path, project_root, callback)
  if type(path) == "string" then
    path = { path }
  end
  local jobrunner =
    require("vectorcode.integrations.codecompanion.common").initialise_runner(
      vc_config.get_user_config().async_backend == "lsp"
    )

  local args = { "vectorise", "--pipe" }
  if project_root then
    vim.list_extend(args, { "--project_root", project_root })
  end
  vim.list_extend(args, path)
  jobrunner.run_async(args, function(result, error, code, signal)
    if type(callback) == "function" then
      callback(result, error, code, signal)
    end
  end, 0)
end
return M
