---@type VectorCode.CacheBackend
local M = {}
local vc_config = require("vectorcode.config")
local notify_opts = vc_config.notify_opts
local jobrunner = require("vectorcode.jobrunner.cmd")

local logger = vc_config.logger

---@type table<integer, VectorCode.Cache>
local CACHE = {}

---@param bufnr integer
local function kill_jobs(bufnr)
  ---@type VectorCode.Cache?
  local cache = CACHE[bufnr]
  if cache ~= nil then
    for job_pid, is_running in pairs(cache.jobs) do
      if type(is_running) == "number" then
        vim.uv.kill(job_pid, 15)
      end
    end
  end
end

---@param query_message string|string[]
---@param buf_nr integer
local function async_runner(query_message, buf_nr)
  if CACHE[buf_nr] == nil or not CACHE[buf_nr].enabled then
    return
  end
  local buf_name
  vim.schedule(function()
    buf_name = vim.api.nvim_buf_get_name(buf_nr)
    logger.debug("Started default cacher job on :", buf_name)
  end)
  ---@type VectorCode.Cache
  local cache = CACHE[buf_nr]
  local args = {
    "query",
    "--pipe",
    "-n",
    tostring(cache.options.n_query),
  }

  if type(query_message) == "string" then
    query_message = { query_message }
  end
  vim.list_extend(args, query_message)

  if cache.options.exclude_this then
    vim.list_extend(args, { "--exclude", vim.api.nvim_buf_get_name(buf_nr) })
  end

  local project_root = cache.options.project_root
  if project_root ~= nil then
    assert(
      vim.uv.fs_stat(vim.fs.normalize(project_root)).type == "directory",
      ("%s is not a valid directory!"):format(project_root)
    )
    vim.list_extend(args, { "--project_root", project_root })
  end

  if cache.options.single_job then
    kill_jobs(buf_nr)
  end

  CACHE[buf_nr].job_count = CACHE[buf_nr].job_count + 1
  logger.debug("vectorcode default cacher job args: ", args)

  -- jobrunner is assumed to be defined at the module level, e.g., local jobrunner = require("vectorcode.jobrunner.cmd")
  local job_pid
  job_pid = jobrunner.run_async(
    args,
    function(json_result, stderr_error, exit_code, signal)
      if not M.buf_is_registered(buf_nr) then
        return
      end
      logger.debug("vectorcode ", buf_name, " default cacher results: ", json_result)
      CACHE[buf_nr].job_count = CACHE[buf_nr].job_count - 1
      assert(job_pid ~= nil)
      CACHE[buf_nr].jobs[job_pid] = nil

      if exit_code ~= 0 then
        vim.schedule(function()
          if CACHE[buf_nr].options.notify then
            if signal == 15 then
              vim.notify("Retrieval aborted.", vim.log.levels.INFO, notify_opts)
            else
              vim.notify(
                "Retrieval failed:\\n" .. table.concat(stderr_error, "\n"),
                vim.log.levels.WARN,
                notify_opts
              )
            end
          end
        end)
        return
      end
      cache = CACHE[buf_nr]
      cache.retrieval = json_result or {}
      vim.schedule(function()
        if cache.options.notify then
          vim.notify(
            ("Caching for buffer %d has completed."):format(buf_nr),
            vim.log.levels.INFO,
            notify_opts
          )
        end
      end)
    end,
    buf_nr
  )

  ---@type VectorCode.Cache
  cache = CACHE[buf_nr]
  if job_pid then
    cache.last_run = vim.uv.clock_gettime("realtime").sec
    cache.jobs[job_pid] = vim.uv.clock_gettime("realtime").sec
  end
  vim.schedule(function()
    if cache.options.notify then
      vim.notify(
        ("Caching for buffer %d has started."):format(buf_nr),
        vim.log.levels.INFO,
        notify_opts
      )
    end
  end)
end

M.register_buffer = vc_config.check_cli_wrap(
  ---This function registers a buffer to be cached by VectorCode. The
  ---registered buffer can be acquired by the `query_from_cache` API.
  ---The retrieval of the files occurs in the background, so this
  ---function will not block the main thread.
  ---
  ---NOTE: this function uses an autocommand to track the changes to the buffer and trigger retrieval.
  ---@param bufnr integer? Default to the current buffer.
  ---@param opts VectorCode.RegisterOpts? Async options.
  function(bufnr, opts)
    if bufnr == 0 or bufnr == nil then
      bufnr = vim.api.nvim_get_current_buf()
    end
    logger.info(
      ("Registering buffer %s %s for default cacher."):format(
        bufnr,
        vim.api.nvim_buf_get_name(bufnr)
      )
    )
    if M.buf_is_registered(bufnr) then
      opts = vim.tbl_deep_extend("force", CACHE[bufnr].options, opts or {})
    end
    opts =
      vim.tbl_deep_extend("force", vc_config.get_user_config().async_opts, opts or {})

    if M.buf_is_registered(bufnr) then
      -- update the options and/or query_cb
      CACHE[bufnr].options =
        vim.tbl_deep_extend("force", CACHE[bufnr].options, opts or {})
      logger.debug(
        ("Updated `default` cacher opts for buffer %s:\n%s"):format(
          bufnr,
          vim.inspect(opts)
        )
      )
    else
      CACHE[bufnr] = {
        enabled = true,
        retrieval = nil,
        options = opts,
        jobs = {},
        job_count = 0,
      }
    end
    if opts.run_on_register then
      async_runner(opts.query_cb(bufnr), bufnr)
    end
    local group = vim.api.nvim_create_augroup(
      ("VectorCodeCacheGroup%d"):format(bufnr),
      { clear = true }
    )
    vim.api.nvim_create_autocmd(opts.events, {
      group = group,
      callback = function()
        assert(CACHE[bufnr] ~= nil, "buffer vectorcode cache not registered")
        local cache = CACHE[bufnr]
        if
          cache.last_run == nil
          or (vim.uv.clock_gettime("realtime").sec - cache.last_run) > opts.debounce
        then
          local cb = cache.options.query_cb
          assert(type(cb) == "function")
          async_runner(cb(bufnr), bufnr)
        end
      end,
      buffer = bufnr,
      desc = "Run query on certain autocmd",
    })
    vim.api.nvim_create_autocmd("BufWinLeave", {
      buffer = bufnr,
      desc = "Kill all running VectorCode async jobs.",
      group = group,
      callback = function()
        kill_jobs(bufnr)
      end,
    })
  end
)

M.deregister_buffer = vc_config.check_cli_wrap(
  ---This function deregisters a buffer from VectorCode. This will kill all
  ---running jobs, delete cached results, and deregister the autocommands
  ---associated with the buffer. If the caching has not been registered, an
  ---error notification will bef ired.
  ---@param bufnr integer?
  ---@param opts {notify:boolean}
  function(bufnr, opts)
    opts = opts or { notify = false }
    if bufnr == nil or bufnr == 0 then
      bufnr = vim.api.nvim_get_current_buf()
    end
    logger.info(
      ("Deregistering buffer %s %s"):format(bufnr, vim.api.nvim_buf_get_name(bufnr))
    )
    if M.buf_is_registered(bufnr) then
      kill_jobs(bufnr)
      vim.api.nvim_del_augroup_by_name(("VectorCodeCacheGroup%d"):format(bufnr))
      CACHE[bufnr] = nil
      if opts.notify then
        vim.notify(
          ("VectorCode Caching has been unregistered for buffer %d."):format(bufnr),
          vim.log.levels.INFO,
          notify_opts
        )
      end
    else
      vim.notify(
        ("VectorCode Caching hasn't been registered for buffer %d."):format(bufnr),
        vim.log.levels.ERROR,
        notify_opts
      )
    end
  end
)

---@param bufnr integer?
---@return boolean
M.buf_is_registered = function(bufnr)
  if bufnr == 0 or bufnr == nil then
    bufnr = vim.api.nvim_get_current_buf()
  end
  return type(CACHE[bufnr]) == "table" and CACHE[bufnr] ~= {}
end

M.query_from_cache = vc_config.check_cli_wrap(
  ---This function queries VectorCode from cache. Returns an array of results. Each item
  ---of the array is in the format of `{path="path/to/your/code.lua", document="document content"}`.
  ---@param bufnr integer?
  ---@param opts {notify: boolean}?
  ---@return VectorCode.Result[]
  function(bufnr, opts)
    local result = {}
    if bufnr == 0 or bufnr == nil then
      bufnr = vim.api.nvim_get_current_buf()
    end
    if M.buf_is_registered(bufnr) then
      opts = vim.tbl_deep_extend(
        "force",
        { notify = CACHE[bufnr].options.notify },
        opts or {}
      )
      result = CACHE[bufnr].retrieval or {}
      local message = ("Retrieved %d documents from cache."):format(#result)
      logger.trace(("vectorcode cmd cacher for buf %s: %s"):format(bufnr, message))
      if opts.notify then
        vim.schedule(function()
          vim.notify(message, vim.log.levels.INFO, notify_opts)
        end)
      end
    end
    return result
  end
)

---@alias ComponentCallback fun(result:VectorCode.Result):string

---Compile the retrieval results into a string.
---@param bufnr integer
---@param component_cb ComponentCallback? The component callback that formats a retrieval result.
---@return {content:string, count:integer}
function M.make_prompt_component(bufnr, component_cb)
  if bufnr == 0 or bufnr == nil then
    bufnr = vim.api.nvim_get_current_buf()
  end
  if not M.buf_is_registered(bufnr) then
    return { content = "", count = 0 }
  end
  if component_cb == nil then
    ---@type fun(result:VectorCode.Result):string
    component_cb = function(result)
      return "<|file_sep|>" .. result.path .. "\n" .. result.document
    end
  end
  local final_component = ""
  local retrieval = M.query_from_cache(bufnr)
  for _, file in pairs(retrieval) do
    final_component = final_component .. component_cb(file)
  end
  return { content = final_component, count = #retrieval }
end

---Checks if VectorCode has been configured properly for your project.
---See the CLI manual for details.
---@param check_item string?
---@param on_success fun(out: vim.SystemCompleted)?
---@param on_failure fun(out: vim.SystemCompleted?)?
function M.async_check(check_item, on_success, on_failure)
  vim.deprecate(
    "vectorcode.cacher.default.async_check",
    'require("vectorcode.cacher").utils.async_check',
    "0.7.0",
    "VectorCode",
    true
  )
  require("vectorcode.cacher").utils.async_check(check_item, on_success, on_failure)
end

---@param bufnr integer?
---@return integer
function M.buf_job_count(bufnr)
  if bufnr == nil or bufnr == 0 then
    bufnr = vim.api.nvim_get_current_buf()
  end
  if M.buf_is_registered(bufnr) then
    return CACHE[bufnr].job_count
  else
    return 0
  end
end

---@param bufnr integer?
---@return boolean
function M.buf_is_enabled(bufnr)
  if bufnr == nil or bufnr == 0 then
    bufnr = vim.api.nvim_get_current_buf()
  end
  return CACHE[bufnr] ~= nil and CACHE[bufnr].enabled
end

return M
