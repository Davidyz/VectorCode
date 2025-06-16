local M = {}

local vc_config = require("vectorcode.config")
local logger = vc_config.logger
local get_config = vc_config.get_user_config
local notify_opts = vc_config.notify_opts
local jobrunner = require("vectorcode.jobrunner.cmd")
local notify = vim.schedule_wrap(vim.notify)

M.query = vc_config.check_cli_wrap(
  ---This function wraps the `query` subcommand of the VectorCode CLI. When used without the `callback` parameter,
  ---this function works as a synchronous function and return the results. Otherwise, this function will run async
  ---and the results are accessible by the `callback` function (the results will be passed as the argument to the
  ---callback).
  ---@param query_message string|string[] Query message(s) to send to the `vecctorcode query` command
  ---@param opts VectorCode.QueryOpts? A table of config options. If nil, the default config or `setup` config will be used.
  ---@param callback fun(result:VectorCode.Result[])? Use the result async style.
  ---@return VectorCode.Result[]? An array of results.
  function(query_message, opts, callback)
    logger.info("vectorcode.query: ", query_message, opts, callback)
    opts = vim.tbl_deep_extend("force", vc_config.get_query_opts(), opts or {})
    if opts.n_query == 0 then
      if opts.notify then
        vim.notify("n_query is 0. Not sending queries.")
      end
      return {}
    end

    ---@type integer?
    local timeout_ms = opts.timeout_ms
    if timeout_ms < 1 then
      timeout_ms = nil
    end
    if opts.notify then
      vim.notify(
        ("Started retrieving %s documents."):format(tostring(opts.n_query)),
        vim.log.levels.INFO,
        notify_opts
      )
    end
    local bufnr = vim.api.nvim_get_current_buf()
    local args = { "query", "--pipe", "-n", tostring(opts.n_query) }
    if type(query_message) == "string" then
      query_message = { query_message }
    end
    vim.list_extend(args, query_message)

    if opts.exclude_this then
      vim.list_extend(args, { "--exclude", vim.api.nvim_buf_get_name(bufnr) })
    end

    logger.debug("vectorcode.query cmd args: ", args)
    if callback == nil then
      local result, err = jobrunner.run(args, timeout_ms, bufnr)
      if err then
        logger.warn(vim.inspect(err))
      end
      logger.debug(result)
      return result
    else
      jobrunner.run_async(args, function(result, error)
        logger.debug(result)
        callback(result)
        if error then
          logger.warn(vim.inspect(error))
        end
      end, bufnr)
    end
  end
)

M.vectorise = vc_config.check_cli_wrap(
  ---This function wraps the `vectorise` subcommand. By default this function doesn't pass a `--project_root` flag.
  ---The command will be run from the current working directory, and the normal project root detection logic in the
  ---CLI will work as normal. You may also pass a `project_root` as the second argument, in which case the
  ---`--project_root` will be passed.
  ---@param files string|string[] Files to vectorise.
  ---@param project_root string? Add the `--project_root` flag and the passed argument to the command.
  function(files, project_root)
    logger.info("vectorcode.vectorise: ", files, project_root)
    local args = { "--pipe", "vectorise" }
    if
      project_root ~= nil
      or (
        M.check("config", function(obj)
          if obj.code == 0 then
            project_root = obj.stdout
          end
        end)
      )
    then
      vim.list_extend(args, { "--project_root", project_root })
    end
    if type(files) == "string" then
      files = { files }
    end
    local valid_files = {}
    for k, v in pairs(files) do
      if vim.fn.filereadable(v) == 1 then
        vim.list_extend(valid_files, { files[k] })
      end
    end
    if #valid_files > 0 then
      vim.list_extend(args, valid_files)
    else
      return
    end
    if get_config().notify then
      vim.schedule(function()
        vim.notify(
          ("Vectorising %s"):format(table.concat(files, ", ")),
          vim.log.levels.INFO,
          notify_opts
        )
      end)
    end
    local bufnr = vim.api.nvim_get_current_buf()
    logger.debug("vectorcode.vectorise cmd args: ", args)
    jobrunner.run_async(args, function(result, error)
      if result then
        if vc_config.get_user_config().notify then
          vim.schedule_wrap(vim.notify)(
            "Indexing successful.",
            vim.log.levels.INFO,
            notify_opts
          )
        end
        logger.info("Vectorise result:", vim.inspect(result))
      elseif error then
        vim.schedule_wrap(vim.notify)(
          string.format("Indexing failed:\n%s", vim.inspect(error)),
          vim.log.levels.WARN,
          notify_opts
        )
        logger.warn(vim.inspect(error))
      else
        vim.schedule_wrap(vim.notify)(
          "Indexing failed.",
          vim.log.levels.WARN,
          notify_opts
        )
      end
    end, bufnr)
  end
)

---@param project_root string?
M.update = vc_config.check_cli_wrap(function(project_root)
  logger.info("vectorcode.update: ", project_root)
  local args = { "update" }
  if
    type(project_root) == "string"
    and vim.uv.fs_stat(vim.fs.normalize(project_root)).type == "directory"
  then
    vim.list_extend(args, { "--project_root", project_root })
  end
  logger.debug("vectorcode.update cmd args: ", args)
  jobrunner.run_async(args, function(result, error)
    if result then
      if vc_config.get_user_config().notify then
        notify("Indexing successful.", vim.log.levels.INFO, notify_opts)
      end
      logger.info("Update result:", vim.inspect(result))
    elseif error then
      notify(
        string.format("Update failed:\n%s", vim.inspect(error)),
        vim.log.levels.WARN,
        notify_opts
      )
      logger.warn(vim.inspect(error))
    else
      notify("Update failed.", vim.log.levels.WARN, notify_opts)
    end
  end, vim.api.nvim_get_current_buf())

  if get_config().notify then
    notify("Updating VectorCode embeddings...", vim.log.levels.INFO, notify_opts)
  end
end)

---@param check_item string? See `vectorcode check` documentation.
---@param stdout_cb fun(stdout: vim.SystemCompleted)? Gives user access to the exit code, stdout and signal.
---@return boolean
function M.check(check_item, stdout_cb)
  if not vc_config.has_cli() then
    return false
  end
  check_item = check_item or "config"
  local return_code
  jobrunner.run_async({ "check", check_item }, function(result, error, code, signal)
    return_code = code
    if type(stdout_cb) == "function" then
      stdout_cb({
        stdout = table.concat(vim.iter(result):flatten(math.huge):totable()),
        stderr = table.concat(vim.iter(error):flatten(math.huge):totable()),
        code = code,
        signal = signal,
      })
    end
  end, 0)
  return return_code == 0
end

---@alias prompt_type "ls"|"query"|"vectorise"
---@param item prompt_type|prompt_type[]|nil
---@return string[]
M.prompts = vc_config.check_cli_wrap(function(item)
  local args = { "prompts", "-p" }
  if item then
    if type(item) == "string" then
      table.insert(args, item)
    else
      vim.list_extend(args, item)
    end
  end
  local result, error = jobrunner.run(args, -1, 0)
  if result == nil or vim.tbl_isempty(result) then
    logger.warn(vim.inspect(error))
    if vc_config.get_user_config().notify then
      notify(vim.inspect(error))
    end
    return {}
  end
  return result
end)

M.setup = vc_config.setup
return M
