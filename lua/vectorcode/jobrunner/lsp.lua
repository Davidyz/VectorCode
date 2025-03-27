if
  vim.fn.executable("vectorcode-server") ~= 1
  or vim.system({ "vectorcode-server", "--version" }):wait().code ~= 0
then
  return nil
end

---@type VectorCode.JobRunner
local jobrunner = {}

---@type vim.lsp.Client
local CLIENT = nil

local vc_config = require("vectorcode.config")
local notify_opts = vc_config.notify_opts

--- Returns the Client ID if applicable, or `nil` if the language server fails to start
---@param ok_to_fail boolean
---@return integer?
function jobrunner.init(ok_to_fail)
  ok_to_fail = ok_to_fail or true
  local client_id = vim.lsp.start(vc_config.lsp_configs, {})
  if client_id ~= nil then
    -- server started
    CLIENT = vim.lsp.get_client_by_id(client_id) --[[@as vim.lsp.Client]]
  else
    -- failed to start server
    if vc_config.get_user_config().notify or not ok_to_fail then
      vim.schedule(function()
        vim.notify(
          "Failed to start vectorcode-server due some error.",
          vim.log.levels.ERROR,
          notify_opts
        )
      end)
    end
    return nil
  end
  return client_id
end

function jobrunner.run(args, timeout_ms, bufnr)
  jobrunner.init(false)
  assert(CLIENT ~= nil)
  assert(bufnr ~= nil)
  if timeout_ms == nil or timeout_ms < 0 then
    timeout_ms = 2 ^ 31 - 1
  end
  args = require("vectorcode.jobrunner").find_root(args, bufnr)

  local result, err
  jobrunner.run_async(args, function(res, err)
    result = res
    err = err
  end, bufnr)
  vim.wait(timeout_ms, function()
    return (result ~= nil) or (err ~= nil)
  end)
  if result == nil then
    return {}, err
  end
  return result, err
end

function jobrunner.run_async(args, callback, bufnr)
  jobrunner.init(false)
  assert(CLIENT ~= nil)
  assert(bufnr ~= nil)

  if not CLIENT.attached_buffers[bufnr] then
    if vim.lsp.buf_attach_client(bufnr, CLIENT.id) then
      local uri = vim.uri_from_bufnr(bufnr)
      local text = vim.api.nvim_buf_get_lines(bufnr, 0, -1, true)
      vim.schedule_wrap(CLIENT.notify)(vim.lsp.protocol.Methods.textDocument_didOpen, {
        textDocument = {
          uri = uri,
          text = text,
          version = 1,
          languageId = vim.bo[bufnr].filetype,
        },
      })
    else
      vim.notify("Failed to attach lsp client")
    end
  end
  args = require("vectorcode.jobrunner").find_root(args, bufnr)
  local _, id = CLIENT:request(
    vim.lsp.protocol.Methods.workspace_executeCommand,
    { command = "vectorcode", arguments = args },
    function(err, result, _, _)
      if type(callback) == "function" then
        local err_message = {}
        if err ~= nil and err.message ~= nil then
          err_message = { err.message }
        end
        vim.schedule_wrap(callback)(result, err_message)
      end
    end,
    bufnr
  )
  return id
end

function jobrunner.is_job_running(job_handler)
  jobrunner.init(true)
  if CLIENT ~= nil then
    local request_data = CLIENT.requests[job_handler]
    return request_data ~= nil and request_data.type == "pending"
  end
  return false
end

function jobrunner.stop_job(job_handler)
  jobrunner.init(true)
  if CLIENT ~= nil then
    CLIENT:cancel_request(job_handler)
  end
end

return jobrunner
