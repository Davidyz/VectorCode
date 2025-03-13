---@type VectorCode.CacheBackend
local M = {}
local vc_config = require("vectorcode.config")
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
  assert(bufnr ~= 0)
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
  return client_id ~= nil
end
---@param project_root string?
---@param ok_to_fail boolean?
---@return boolean
local function start_server(project_root, ok_to_fail)
  if is_lsp_running() then
    return true
  end

  if ok_to_fail == nil then
    ok_to_fail = true
  end
  local cmd = { "vectorcode-server" }
  if
    type(project_root) == "string"
    and vim.list_contains({ "directory" }, vim.uv.fs_stat(project_root).type)
  then
    vim.list_extend(cmd, { "--project_root", project_root })
  else
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
  else
    client_id = id
    return true
  end
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
  for request_id, time in pairs(CACHE[bufnr].jobs) do
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
  assert(client_id ~= nil)
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
    local request_id = job_runner.run_async(args, function(result, err)
      CACHE[buf_nr].retrieval = result or CACHE[buf_nr].retrieval or {}
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
    if M.buf_is_registered(bufnr) then
      opts = vim.tbl_deep_extend("force", CACHE[bufnr].options, opts or {})
    end
    opts =
      vim.tbl_deep_extend("force", vc_config.get_user_config().async_opts, opts or {})

    if
      not start_server(opts.project_root or vim.api.nvim_buf_get_name(bufnr), false)
    then
      -- failed to start the server
      return false
    end
    assert(client_id ~= nil)
    if not vim.lsp.buf_is_attached(bufnr, client_id) then
      if vim.lsp.buf_attach_client(bufnr, client_id) then
        local uri = vim.uri_from_bufnr(bufnr)
        local text = vim.api.nvim_buf_get_lines(bufnr, 0, -1, true)
        vim.schedule_wrap(vim.lsp.get_client_by_id(client_id).notify)(
          vim.lsp.protocol.Methods.textDocument_didOpen,
          {
            textDocument = {
              uri = uri,
              text = text,
              version = 1,
              languageId = vim.bo[bufnr].filetype,
            },
          }
        )
      else
        vim.notify("Failed to attach lsp client")
      end
    end

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
      if opts.notify then
        vim.schedule(function()
          vim.notify(
            ("Retrieved %d documents from cache."):format(#result),
            vim.log.levels.INFO,
            notify_opts
          )
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
  if not vc_config.has_cli() then
    if on_failure ~= nil then
      on_failure()
    end
    return
  end

  check_item = check_item or "config"
  vim.system({ "vectorcode", "check", check_item }, {}, function(out)
    if out.code == 0 and type(on_success) == "function" then
      vim.schedule_wrap(on_success)(out)
    elseif out.code ~= 0 and type(on_failure) == "function" then
      vim.schedule_wrap(on_failure)(out)
    end
  end)
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
