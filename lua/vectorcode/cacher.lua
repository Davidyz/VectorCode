local M = {}
local vc_config = require("vectorcode.config")
local notify_opts = vc_config.notify_opts

---@class VectorCodeCache
---@field enabled boolean
---@field retrieval VectorCodeResult[]?
---@field options VectorCodeConfig
---@field query_cb fun(buf_number: integer): string
---@field jobs table<integer, boolean>
---@field last_run integer?
local default_vectorcode_cache = {
  enabled = true,
  retrieval = {},
  options = vc_config.get_user_config(),
  jobs = {},
}

---@param query_message string
---@param buf_nr integer
local function async_runner(query_message, buf_nr)
  if not vim.b[buf_nr].vectorcode_cache.enabled then
    return
  end
  local cache = vim.api.nvim_buf_get_var(buf_nr, "vectorcode_cache")
  vim.api.nvim_buf_set_var(buf_nr, "vectorcode_cache", cache)
  local args = {
    "query",
    "--pipe",
    "-n",
    tostring(cache.options.n_query),
    query_message,
  }

  if cache.exclude_this then
    vim.list_extend(args, { "--exclude", vim.api.nvim_buf_get_name(buf_nr) })
  end
  local job = require("plenary.job"):new({
    command = "vectorcode",
    args = args,
    on_start = function() end,
    on_exit = function(self, code, signal)
      vim.schedule(function()
        ---@type VectorCodeCache
        local cache = vim.api.nvim_buf_get_var(buf_nr, "vectorcode_cache")
        cache.jobs[self.pid] = nil
        vim.api.nvim_buf_set_var(buf_nr, "vectorcode_cache", cache)
      end)
      local ok, json = pcall(
        vim.json.decode,
        table.concat(self:result()) or "[]",
        { array = true, object = true }
      )
      if not ok or code ~= 0 then
        vim.schedule(function()
          if vim.api.nvim_buf_get_var(buf_nr, "vectorcode_cache").options.notify then
            vim.notify(
              "Retrieval failed:\n" .. table.concat(self:result()),
              vim.log.levels.WARN,
              notify_opts
            )
          end
        end)
        return
      end
      vim.schedule(function()
        ---@type VectorCodeCache
        local cache = vim.api.nvim_buf_get_var(buf_nr, "vectorcode_cache")
        cache.retrieval = json or {}
        vim.api.nvim_buf_set_var(buf_nr, "vectorcode_cache", cache)
        if cache.options.notify then
          vim.notify(
            ("Caching for buffer %d has completed."):format(buf_nr),
            vim.log.levels.INFO,
            notify_opts
          )
        end
      end)
    end,
  })
  vim.schedule(function()
    job:start()
    ---@type VectorCodeCache
    local cache = vim.api.nvim_buf_get_var(buf_nr, "vectorcode_cache")
    cache.jobs[job.pid] = true
    if cache.options.notify then
      vim.notify(
        ("Caching for buffer %d has started."):format(buf_nr),
        vim.log.levels.INFO,
        notify_opts
      )
    end
    cache.last_run = vim.uv.clock_gettime("realtime").sec
    vim.api.nvim_buf_set_var(buf_nr, "vectorcode_cache", cache)
  end)
end

---@param bufnr integer?
---@param opts VectorCodeConfig?
---@param query_cb VectorCodeQueryCallback?
---@param events string[]?
---@param debounce integer?
function M.register_buffer(bufnr, opts, query_cb, events, debounce)
  debounce = debounce or 10
  if bufnr == 0 or bufnr == nil then
    bufnr = vim.api.nvim_get_current_buf()
  end
  if M.buf_is_registered(bufnr) then
    -- update the options and/or query_cb
    vim.schedule(function()
      local cache = vim.b[bufnr].vectorcode_cache
      cache.options = vim.tbl_deep_extend("force", cache.options, opts or {})
      if type(query_cb) == "function" then
        cache.query_cb = query_cb
      end
      vim.api.nvim_buf_set_var(
        bufnr or vim.api.nvim_get_current_buf(),
        "vectorcode_cache",
        cache
      )
    end)
    return
  end
  events = events or { "BufWritePost", "InsertEnter", "BufReadPost" }
  query_cb = query_cb or require("vectorcode.utils").surrounding_lines_cb(-1)
  opts = vim.tbl_deep_extend("keep", opts or {}, vc_config.get_user_config())
  vim.schedule(function()
    ---@type VectorCodeCache
    vim.api.nvim_buf_set_var(bufnr, "vectorcode_cache", {
      enabled = true,
      retrieval = nil,
      options = opts,
      query_cb = query_cb,
      jobs = {},
    })
  end)
  vim.schedule(function()
    vim.api.nvim_create_autocmd(events, {
      callback = function()
        assert(
          vim.b[bufnr].vectorcode_cache ~= nil,
          "buffer vectorcode cache not registered"
        )
        vim.schedule(function()
          local cache = vim.api.nvim_buf_get_var(bufnr, "vectorcode_cache") ---@cast cache VectorCodeCache
          if
            cache.last_run == nil
            or (vim.uv.clock_gettime("realtime").sec - cache.last_run) > debounce
          then
            local cb = cache.query_cb
            async_runner(cb(bufnr), bufnr)
          end
        end)
      end,
      buffer = bufnr,
    })
  end)
end

---@param bufnr integer?
---@return boolean
M.buf_is_registered = function(bufnr)
  if bufnr == 0 or bufnr == nil then
    bufnr = vim.api.nvim_get_current_buf()
  end
  return type(vim.b[bufnr].vectorcode_cache) == "table"
end

---@param bufnr integer?
---@return VectorCodeResult[]
M.query_from_cache = function(bufnr)
  local result = {}
  if bufnr == 0 or bufnr == nil then
    bufnr = vim.api.nvim_get_current_buf()
  end
  if M.buf_is_registered(bufnr) then
    ---@type VectorCodeCache
    local vectorcode_cache = vim.api.nvim_buf_get_var(bufnr, "vectorcode_cache")
    result = vectorcode_cache.retrieval or {}
    if vectorcode_cache.options.notify then
      vim.notify(
        ("Retrieved %d documents from cache."):format(#result),
        vim.log.levels.INFO,
        notify_opts
      )
    end
  end
  return result
end

function M.lualine()
  return {
    function()
      local message = "VectorCode: "
      ---@type VectorCodeCache
      local cache = vim.b.vectorcode_cache
      if cache.enabled then
        if cache.retrieval ~= nil then
          message = message .. tostring(#cache.retrieval)
        end
        if #(vim.tbl_keys(cache.jobs)) > 0 then
          message = message .. "  "
        else
          message = message .. "  "
        end
      else
        message = message .. " "
      end
      return message
    end,
    cond = function()
      return vim.b.vectorcode_cache ~= nil
    end,
  }
end
return M
