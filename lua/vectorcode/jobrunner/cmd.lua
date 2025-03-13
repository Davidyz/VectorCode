---@type VectorCode.JobRunner
local runner = {}

local Job = require("plenary.job")
---@type {integer: Job}
local jobs = {}

function runner.run_async(args, callback)
  if type(callback) == "function" then
    callback = vim.schedule_wrap(callback)
  else
    callback = nil
  end
  local cmd = { "vectorcode" }
  vim.list_extend(cmd, args)
  local job = Job:new({
    command = "vectorcode",
    args = args,
    on_exit = function(self, _, _)
      jobs[self.pid] = nil
      local result = self:result()
      local ok, decoded = pcall(vim.json.decode, table.concat(result, ""))
      if callback ~= nil then
        if ok then
          callback(decoded, self:stderr_result())
        else
          callback({ result }, self:stderr_result())
        end
      end
    end,
  })
  jobs[job.pid] = job
  job:start()
  return tonumber(job.pid)
end

function runner.run(args, timeout_ms, bufnr)
  if timeout_ms == nil or timeout_ms < 0 then
    timeout_ms = 2 ^ 31 - 1
  end
  local res, err
  local pid = runner.run_async(args, function(result, error)
    res = result
    err = error
  end, bufnr)
  if pid ~= nil then
    jobs[pid]:wait(timeout_ms)
    jobs[pid] = nil
    return res, err
  else
    return {}, err
  end
end

function runner.is_job_running(job)
  return jobs[job] ~= nil
end

function runner.stop_job(job_handle)
  local job = jobs[job_handle]
  if job ~= nil then
    job:shutdown(1, 15)
  end
end

return runner
