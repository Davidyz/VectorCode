local vc_config = require("vectorcode.config")

---@type VectorCode.JobRunner
local jobrunner = {}

---@type vim.lsp.Client
local CLIENT = nil

local notify_opts = vc_config.notify_opts
local logger = vc_config.logger

--- Returns the Client ID if applicable, or `nil` if the language server fails to start
---@param ok_to_fail boolean
---@return integer?
function jobrunner.init(ok_to_fail)
  local existing_clients = vim.lsp.get_clients({ name = vc_config.lsp_configs().name })
  if #existing_clients > 0 then
    CLIENT = existing_clients[1]
    return CLIENT.id
  end
  ok_to_fail = ok_to_fail or true
  local client_id = vim.lsp.start(vc_config.lsp_configs(), {})
  if client_id ~= nil then
    -- server started
    CLIENT = vim.lsp.get_client_by_id(client_id) --[[@as vim.lsp.Client]]
  else
    -- failed to start server
    if vc_config.get_user_config().notify or not ok_to_fail then
      local message = "Failed to start vectorcode-server due some error."
      logger.error(message)
      vim.schedule(function()
        vim.notify(message, vim.log.levels.ERROR, notify_opts)
      end)
    end
    return nil
  end
  return client_id
end

function jobrunner.run(args, timeout_ms, bufnr)
  jobrunner.init(false)
  assert(CLIENT ~= nil, "Failed to initialize the LSP server!")
  assert(bufnr ~= nil, "Need to pass the buffer number!")
  if timeout_ms == nil or timeout_ms < 0 then
    timeout_ms = 2 ^ 31 - 1
  end
  args = require("vectorcode.jobrunner").find_root(args, bufnr)

  local result, err, code
  jobrunner.run_async(args, function(res, e, e_code)
    result = res
    err = e
    code = e_code
  end, bufnr)
  vim.wait(timeout_ms, function()
    return (result ~= nil) or (err ~= nil)
  end)
  return result or {}, err, code
end

function jobrunner.run_async(args, callback, bufnr)
  assert(jobrunner.init(false))
  assert(bufnr ~= nil, "Need to pass the buffer number!")
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
      local message = "Failed to attach lsp client"
      vim.schedule(function()
        vim.notify(message)
      end)
      logger.warn(message)
    end
  end
  args = require("vectorcode.jobrunner").find_root(args, bufnr)
  logger.debug(
    ("lsp jobrunner for buffer %s args: %s"):format(bufnr, vim.inspect(args))
  )
  local _, id = CLIENT:request(
    vim.lsp.protocol.Methods.workspace_executeCommand,
    -- NOTE: This is not a hardcoded executable, but rather part of our LSP implementation.
    { command = "vectorcode", arguments = args },
    function(err, result, _, _)
      if type(callback) == "function" then
        local err_message = {}
        if err ~= nil and err.message ~= nil then
          err_message = { err.message }
        end
        local code = 0
        if err and err.code then
          code = err.code
        end
        vim.schedule_wrap(callback)(result, err_message, code)
        if result then
          logger.debug("lsp jobrunner result:\n", result)
        end
        if err then
          logger.info("lsp jobrunner error:\n", err)
        end
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
