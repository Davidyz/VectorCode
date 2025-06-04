local log_level = os.getenv("VECTORCODE_NVIM_LOG_LEVEL")
if log_level == nil then
  log_level = "error"
else
  log_level = log_level:lower()
end
local logger = require("plenary.log").new({
  plugin = "vectorcode.nvim",
  level = log_level,
  use_console = log_level ~= nil and "async" or false,
  use_file = log_level ~= nil,
})

local cacher = nil

---@type VectorCode.Opts
local config = {
  cli_cmds = {
    vectorcode = "vectorcode",
  },
  async_opts = {
    debounce = 10,
    events = { "BufWritePost", "InsertEnter", "BufReadPost" },
    exclude_this = true,
    n_query = 1,
    notify = false,
    query_cb = require("vectorcode.utils").make_surrounding_lines_cb(-1),
    run_on_register = false,
    single_job = false,
    timeout_ms = 5000,
  },
  async_backend = "default",
  exclude_this = true,
  n_query = 1,
  notify = true,
  timeout_ms = 5000,
  on_setup = { update = false, lsp = false },
  sync_log_env_var = false,
}

local setup_config = vim.deepcopy(config, true)

---@return vim.lsp.ClientConfig
local lsp_configs = function()
  ---@type vim.lsp.ClientConfig
  local cfg =
    { cmd = { "vectorcode-server" }, root_markers = { ".vectorcode", ".git" } } -- NOTE: This can be overriden by `vim.lsp.config`
  if vim.lsp.config ~= nil and vim.lsp.config.vectorcode_server ~= nil then
    -- nvim >= 0.11.0
    cfg = vim.tbl_deep_extend("force", cfg, vim.lsp.config.vectorcode_server)
    logger.debug("Using vim.lsp.config.vectorcode_server for LSP config:\n", cfg)
  else
    -- nvim < 0.11.0
    local ok, lspconfig = pcall(require, "lspconfig.configs")
    if ok and lspconfig.vectorcode_server ~= nil then
      cfg = lspconfig.vectorcode_server.config_def.default_config
      logger.debug("Using nvim-lspconfig for LSP config:\n", cfg)
    end
  end
  cfg.name = "vectorcode_server"
  if setup_config.sync_log_env_var then
    local level = os.getenv("VECTORCODE_NVIM_LOG_LEVEL") or nil
    if level ~= nil then
      level = string.upper(level)
      if level == "TRACE" then
        -- there's no `TRACE` in python logging
        level = "DEBUG"
      end
      cfg.cmd_env["VECTORCODE_LOG_LEVEL"] = level
    end
  end
  return cfg
end

local notify_opts = { title = "VectorCode" }

---@param opts {notify:boolean}?
local has_cli = function(opts)
  opts = opts or { notify = false }
  local ok = vim.fn.executable(setup_config.cli_cmds.vectorcode) == 1
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
          logger.info("Running `vectorcode update` on start up.")
          require("vectorcode").update(path)
        end
      end
    end)
  end
  if configs.on_setup.lsp then
    local ok, runner = pcall(require, "vectorcode.jobrunner.lsp")
    if not ok or not type(runner) == "table" or runner == nil then
      vim.notify("Failed to start vectorcode-server.", vim.log.levels.WARN, notify_opts)
      logger.error("Failed to start vectorcode-server.")
      return
    end
    runner.init()
  end
end)

return {
  get_default_config = function()
    return vim.deepcopy(config, true)
  end,

  setup = check_cli_wrap(
    ---@param opts VectorCode.Opts?
    function(opts)
      logger.info("Received setup opts:\n", opts)
      opts = opts or {}
      setup_config = vim.tbl_deep_extend("force", config, opts or {})
      for k, _ in pairs(setup_config.async_opts) do
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
      setup_config.cli_cmds.vectorcode =
        vim.fs.normalize(setup_config.cli_cmds.vectorcode)
      startup_handler(setup_config)
      logger.info("Finished processing opts:\n", setup_config)
    end
  ),

  ---@return VectorCode.CacheBackend
  get_cacher_backend = function()
    if cacher ~= nil then
      return cacher
    end
    if setup_config.async_backend == "lsp" then
      local ok, lsp_cacher = pcall(require, "vectorcode.cacher.lsp")
      if ok and type(lsp_cacher) == "table" then
        logger.debug("Using LSP backend for cacher.")
        cacher = lsp_cacher
        return cacher
      else
        vim.notify("Falling back to default backend.", vim.log.levels.WARN, notify_opts)
        logger.warn("Fallback to default (cmd) backend for cacher.")
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
      logger.warn("Fallback to default (cmd) backend for cacher.")
      setup_config.async_backend = "default"
    end
    logger.debug("Defaulting to cmd backend for cacher.")
    cacher = require("vectorcode.cacher.default")
    return cacher
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

  lsp_configs = lsp_configs,
  logger = logger,
}
