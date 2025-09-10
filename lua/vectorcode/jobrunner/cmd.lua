---@type VectorCode.JobRunner
local runner = {}

---@type table<integer, vim.SystemObj>
local jobs = {}
local logger = require("vectorcode.config").logger

function runner.run_async(args, callback, bufnr)
  if type(callback) == "function" then
    callback = vim.schedule_wrap(callback)
  else
    callback = nil
  end
  logger.debug(
    ("cmd jobrunner for buffer %s args: %s"):format(bufnr, vim.inspect(args))
  )

  table.insert(
    args,
    1,
    require("vectorcode.config").get_user_config().cli_cmds.vectorcode
  )

  ---@type vim.SystemObj?
  local job
  job = vim.system(args, {}, function(out)
    if job and job.pid then
      jobs[job.pid] = job
    end
    local stdout = out.stdout or "{}"
    if stdout == "" then
      stdout = "{}"
    end
    local _, decoded = pcall(vim.json.decode, stdout, { object = true, array = true })
    if type(callback) == "function" then
      callback(decoded or {}, out.stderr, out.code, out.signal)
    end
  end)
  jobs[job.pid] = job
  return tonumber(job.pid)
end

function runner.run(args, timeout_ms, bufnr)
  if timeout_ms == nil or timeout_ms < 0 then
    timeout_ms = 2 ^ 31 - 1
  end
  local res, err, code, signal
  local pid = runner.run_async(args, function(result, error, e_code, s)
    res = result
    err = error
    code = e_code
    signal = s
  end, bufnr)
  if pid ~= nil and jobs[pid] ~= nil then
    jobs[pid]:wait(timeout_ms)
  end
  return res or {}, err, code, signal
end

function runner.is_job_running(job)
  return jobs[job] ~= nil
end

function runner.stop_job(job_handle)
  local job = jobs[job_handle]
  if job ~= nil then
    job:kill(15)
  end
end

return runner
