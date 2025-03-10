---@type VectorCode.JobRunner
local jobrunner = {}

---@type vim.lsp.Client
local CLIENT = nil

local vc_config = require("vectorcode.config")
local notify_opts = vc_config.notify_opts

---@param ok_to_fail boolean
local function get_client(ok_to_fail)
  ok_to_fail = ok_to_fail or true
  if #vim.lsp.get_clients({ name = "vectorcode-server" }) == 1 then
    CLIENT = vim.lsp.get_clients({ name = "vectorcode-server" })[1]
  else
    local cmd = { "vectorcode-server" }

    local try_root = vim.fs.root(".", ".vectorcode") or vim.fs.root(".", ".git")
    if try_root ~= nil then
      vim.list_extend(cmd, { "--project_root", try_root })
    else
      vim.schedule(function()
        vim.notify(
          "Failed to start vectorcode-server due to failing to resolve the project root.",
          vim.log.levels.ERROR,
          notify_opts
        )
      end)
      return false
    end
    local id, err = vim.lsp.start_client({
      name = "vectorcode-server",
      cmd = cmd,
    })

    if err ~= nil and (vc_config.get_user_config().notify or not ok_to_fail) then
      vim.schedule(function()
        vim.notify(
          ("Failed to start vectorcode-server due to the following error:\n%s"):format(
            err
          ),
          vim.log.levels.ERROR,
          notify_opts
        )
      end)
      return false
    elseif id ~= nil then
      local cli = vim.lsp.get_client_by_id(id)
      if cli ~= nil then
        CLIENT = cli
        return true
      end
    end
  end
end

function jobrunner.run(args, timeout_ms, bufnr)
  get_client(false)
  assert(CLIENT ~= nil)
  assert(bufnr ~= nil)
  if timeout_ms == nil or timeout_ms < 0 then
    timeout_ms = 2 ^ 31 - 1
  end
  local result, err = CLIENT.request_sync(
    vim.lsp.protocol.Methods.workspace_executeCommand,
    { command = "vectorcode", arguments = args },
    timeout_ms,
    bufnr
  )
  if result == nil then
    return {}, { err }
  end
  return result.result, result.err
end

function jobrunner.run_async(args, callback, bufnr)
  get_client(false)
  assert(CLIENT ~= nil)
  assert(bufnr ~= nil)

  local _, id = CLIENT.request(
    vim.lsp.protocol.Methods.workspace_executeCommand,
    { command = "vectorcode", arguments = args },
    function(err, result, _, _)
      callback(result, err)
    end,
    bufnr
  )
  return id
end

function jobrunner.is_job_running(job_handler)
  get_client(true)
  if CLIENT ~= nil then
    local request_data = CLIENT.requests[job_handler]
    return request_data ~= nil and request_data.type == "pending"
  end
  return false
end

function jobrunner.stop_job(job_handler)
  get_client(true)
  if CLIENT ~= nil then
    CLIENT.cancel_request(job_handler)
  end
end

return jobrunner
