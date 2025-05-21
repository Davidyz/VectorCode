---@type VectorCode.JobRunner
local runner = {}

local Job = require("plenary.job")
---@type {integer: Job}
local jobs = {}
local logger = require("vectorcode.config").logger

function runner.run_async(args, callback, bufnr)
  if type(callback) == "function" then
    callback = vim.schedule_wrap(callback)
  else
    callback = nil
  end
  local cmd = { "vectorcode" }
  args = require("vectorcode.jobrunner").find_root(args, bufnr)
  vim.list_extend(cmd, args)
  logger.debug(
    ("cmd jobrunner for buffer %s args: %s"):format(bufnr, vim.inspect(args))
  )
  ---@diagnostic disable-next-line: missing-fields
  local job = Job:new({
    command = "vectorcode",
    args = args,
    on_exit = function(self, _, _)
      jobs[self.pid] = nil
      local result = self:result()
      logger.debug(result)
      local ok, decoded = pcall(vim.json.decode, table.concat(result, ""))
      if callback ~= nil then
        if ok then
          callback(decoded or {}, self:stderr_result())
          if vim.islist(result) then
            logger.debug(
              "cmd jobrunner result:\n",
              vim.tbl_map(function(item)
                if type(item) == "table" then
                  item.document = nil
                  item.chunk = nil
                end
                return item
              end, vim.deepcopy(result))
            )
          end
        else
          callback({ result }, self:stderr_result())
          logger.warn("cmd runner: failed to decode result:\n", result)
        end
      end
    end,
  })
  job:start()
  jobs[job.pid] = job
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
    vim.wait(timeout_ms, function()
      return res ~= nil or err ~= nil
    end)
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
