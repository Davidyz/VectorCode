---@type VectorCode.CacheBackend
local M = {}
local vc_config = require("vectorcode.config")
local logger = vc_config.logger

local notify_opts = vc_config.notify_opts

local job_runner = require("vectorcode.jobrunner.lsp")

if job_runner == nil then
  vim.notify(
    "vectorcode-server is not found. Please make sure you installed `vectorcode[lsp]`.",
    vim.log.levels.ERROR,
    notify_opts
  )
  return nil
end

---@type integer?
local client_id = nil

---@param bufnr integer
---@param project_root string
---@return string?
local function check_project_root(bufnr, project_root)
  assert(bufnr ~= 0, "Need a non-zero bufnr")
  if project_root == nil then
    local cwd = vim.uv.cwd() or "."
    local result = vim.fs.root(cwd, ".vectorcode") or vim.fs.root(cwd, ".git")
    if result ~= nil then
      return vim.fs.normalize(result)
    end
  else
    return vim.fs.normalize(project_root)
  end
end

---@return boolean
local function is_lsp_running()
  client_id = job_runner.init()
  return client_id ~= nil
end

---@type table<integer, VectorCode.Cache>
local CACHE = {}

local function cleanup_lsp_requests()
  if client_id == nil then
    return
  end
  local client = vim.lsp.get_client_by_id(client_id)
  if client == nil then
    return
  end
  for request, data in pairs(client.requests) do
    if data.type ~= "pending" and CACHE[data.bufnr] ~= nil then
      CACHE[data.bufnr].jobs[request] = nil
    end
  end
  for _, cache in pairs(CACHE) do
    for req, _ in pairs(cache.jobs) do
      if client.requests[req] == nil then
        cache.jobs[req] = nil
      end
    end
    cache.job_count = #vim.tbl_keys(cache.jobs)
  end
end

---@params bufnr integer
local function kill_jobs(bufnr)
  if client_id == nil then
    return
  end
  local client = vim.lsp.get_client_by_id(client_id)
  if client == nil then
    return
  end
  for request_id, _ in pairs(CACHE[bufnr].jobs) do
    job_runner.stop_job(request_id)
  end
  cleanup_lsp_requests()
end

---@param query_message string|string[]
---@param buf_nr integer
local function async_runner(query_message, buf_nr)
  if CACHE[buf_nr] == nil or not CACHE[buf_nr].enabled or not is_lsp_running() then
    return
  end
  local buf_name
  vim.schedule(function()
    buf_name = vim.api.nvim_buf_get_name(buf_nr)
    logger.debug("Started lsp cacher job on :", buf_name)
  end)
  assert(client_id ~= nil, "LSP client hasn't been initialised.")
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

  local project_root = check_project_root(buf_nr, cache.options.project_root)
  if project_root == nil then
    if cache.options.notify then
      vim.schedule(function()
        vim.notify(
          ("Failed to auto-detect VectorCode project-root for buffer %d. Aborting."):format(
            buf_nr
          ),
          vim.log.levels.ERROR,
          notify_opts
        )
      end)
    end
    return
  end

  local client = vim.lsp.get_client_by_id(client_id)
  if client ~= nil then
    if CACHE[buf_nr].options.single_job then
      kill_jobs(buf_nr)
    end
    CACHE[buf_nr].job_count = CACHE[buf_nr].job_count + 1

    logger.debug("vectorcode lsp cacher job args: ", args)
    local request_id = job_runner.run_async(args, function(result, err)
      CACHE[buf_nr].retrieval = result or CACHE[buf_nr].retrieval or {}
      logger.debug("vectorcode ", buf_name, " lsp cacher results: ", result)
      if err then
        logger.debug("vectorcode ", buf_name, " lsp cacher error: ", err)
      end
    end, buf_nr)

    if request_id ~= nil then
      CACHE[buf_nr].jobs[request_id] = vim.uv.clock_gettime("realtime").sec
    end
    vim.schedule(function()
      if CACHE[buf_nr].options.notify then
        vim.notify(
          ("Caching for buffer %d has started."):format(buf_nr),
          vim.log.levels.INFO,
          notify_opts
        )
      end
    end)
  end
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
      ("Registering buffer %s %s for lsp cacher."):format(
        bufnr,
        vim.api.nvim_buf_get_name(bufnr)
      )
    )
    if M.buf_is_registered(bufnr) then
      opts = vim.tbl_deep_extend("force", CACHE[bufnr].options, opts or {})
      logger.debug(
        ("Updated `lsp` cacher opts for buffer %s:\n%s"):format(
          bufnr,
          vim.inspect(opts)
        )
      )
    end
    opts =
      vim.tbl_deep_extend("force", vc_config.get_user_config().async_opts, opts or {})

    assert(is_lsp_running())

    if CACHE[bufnr] ~= nil then
      -- update the options and/or query_cb
      CACHE[bufnr].options =
        vim.tbl_deep_extend("force", CACHE[bufnr].options, opts or {})
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
        async_runner(CACHE[bufnr].options.query_cb(bufnr), bufnr)
      end,
      buffer = bufnr,
      desc = "Run query on certain autocmd",
    })
    vim.api.nvim_create_autocmd("BufWinLeave", {
      buffer = bufnr,
      desc = "Kill all running VectorCode async jobs.",
      group = group,
      callback = function()
        if client_id ~= nil then
          vim.lsp.buf_detach_client(bufnr, client_id)
        end
      end,
    })
    return true
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
      if client_id ~= nil then
        vim.lsp.buf_detach_client(bufnr, client_id)
      end
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
  return type(CACHE[bufnr]) == "table"
    and CACHE[bufnr] ~= nil
    and client_id ~= nil
    and vim.lsp.buf_is_attached(bufnr, client_id)
end

M.query_from_cache = vc_config.check_cli_wrap(
  ---This function queries VectorCode from cache. Returns an array of results. Each item
  ---of the array is in the format of `{path="path/to/your/code.lua", document="document content"}`.
  ---@param bufnr integer?
  ---@param opts {notify: boolean}?
  ---@return VectorCode.QueryResult[]
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
    ---@type fun(result:VectorCode.QueryResult):string
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
  cleanup_lsp_requests()
  return #vim.tbl_keys(CACHE[bufnr].jobs)
end

---@param bufnr integer?
---@return boolean
function M.buf_is_enabled(bufnr)
  if bufnr == nil or bufnr == 0 then
    bufnr = vim.api.nvim_get_current_buf()
  end
  return CACHE[bufnr] ~= nil and CACHE[bufnr].enabled and client_id ~= nil
end

return M
