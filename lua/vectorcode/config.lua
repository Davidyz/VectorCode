---@type VectorCode.Opts
local config = {
  async_opts = {
    debounce = 10,
    events = { "BufWritePost", "InsertEnter", "BufReadPost" },
    exclude_this = true,
    n_query = 1,
    notify = false,
    query_cb = require("vectorcode.utils").make_surrounding_lines_cb(-1),
    run_on_register = false,
    single_job = false,
  },
  async_backend = "default",
  exclude_this = true,
  n_query = 1,
  notify = true,
  timeout_ms = 5000,
  on_setup = { update = false, lsp = false },
}

---@type vim.lsp.ClientConfig
local lsp_configs =
  { cmd = { "vectorcode-server" }, root_markers = { ".vectorcode", ".git" } }
if vim.lsp.config ~= nil then
  -- nvim >= 0.11.0
  lsp_configs =
    vim.tbl_deep_extend("force", lsp_configs, vim.lsp.config.vectorcode_server or {})
else
  -- nvim < 0.11.0
  local ok, lspconfig = pcall(require, "lspconfig.configs")
  if ok and lspconfig.vectorcode_server ~= nil then
    lsp_configs = lspconfig.vectorcode_server.config_def.default_config
  end
end

local setup_config = vim.deepcopy(config, true)
local notify_opts = { title = "VectorCode" }

---@param opts {notify:boolean}?
local has_cli = function(opts)
  opts = opts or { notify = false }
  local ok = vim.fn.executable("vectorcode") == 1
  if not ok and opts.notify then
    vim.notify("VectorCode CLI is not executable!", vim.log.levels.ERROR, notify_opts)
  end
  return ok
end

---@generic T: function
---@param func T
---@return T
local check_cli_wrap = function(func)
  if not has_cli() then
    vim.notify("VectorCode CLI is not executable!", vim.log.levels.ERROR, notify_opts)
  end
  return func
end

--- Handles startup actions.
---@param configs VectorCode.Opts
local startup_handler = check_cli_wrap(function(configs)
  if configs.on_setup.update then
    require("vectorcode").check("config", function(out)
      if out.code == 0 then
        local path = string.gsub(out.stdout, "^%s*(.-)%s*$", "%1")
        if path ~= "" then
          require("vectorcode").update(path)
        end
      end
    end)
  end
  if configs.on_setup.lsp then
    local ok, _ = pcall(vim.lsp.start, lsp_configs)
    if not ok then
      vim.notify("Failed to start vectorcode-server.", vim.log.levels.WARN, notify_opts)
    end
  end
end)

return {
  get_default_config = function()
    return vim.deepcopy(config, true)
  end,

  setup = check_cli_wrap(
    ---@param opts VectorCode.Opts?
    function(opts)
      opts = opts or {}
      setup_config = vim.tbl_deep_extend("force", config, opts or {})
      for k, v in pairs(setup_config.async_opts) do
        if
          setup_config[k] ~= nil
          and (opts.async_opts == nil or opts.async_opts[k] == nil)
        then
          -- NOTE: a lot of options are mutual between `setup_config` and `async_opts`.
          -- If users do not explicitly set them `async_opts`, copy them from `setup_config`.
          setup_config.async_opts = vim.tbl_deep_extend(
            "force",
            setup_config.async_opts,
            { [k] = setup_config[k] }
          )
        end
      end
      startup_handler(setup_config)
    end
  ),

  ---@return VectorCode.CacheBackend
  get_cacher_backend = function()
    if setup_config.async_backend == "lsp" then
      local ok, cacher = pcall(require, "vectorcode.cacher.lsp")
      if ok and type(cacher) == "table" then
        return cacher
      else
        vim.notify("Falling back to default backend.", vim.log.levels.WARN, notify_opts)
        setup_config.async_backend = "default"
      end
    end

    if setup_config.async_backend ~= "default" then
      vim.notify(
        ("Unrecognised vectorcode backend: %s! Falling back to `default`."):format(
          setup_config.async_backend
        ),
        vim.log.levels.ERROR,
        notify_opts
      )
      setup_config.async_backend = "default"
    end
    return require("vectorcode.cacher.default")
  end,

  ---@return VectorCode.Opts
  get_user_config = function()
    return vim.deepcopy(setup_config, true)
  end,
  ---@return VectorCode.QueryOpts
  get_query_opts = function()
    return {
      exclude_this = setup_config.exclude_this,
      n_query = setup_config.n_query,
      notify = setup_config.notify,
      timeout_ms = setup_config.timeout_ms,
    }
  end,
  notify_opts = notify_opts,

  ---@return boolean
  has_cli = has_cli,

  check_cli_wrap = check_cli_wrap,
  ---@type vim.lsp.ClientConfig
  lsp_configs = lsp_configs,
}
