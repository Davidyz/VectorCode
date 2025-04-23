local M = {}

local vc_config = require("vectorcode.config")

local logger = vc_config.logger

local get_config = vc_config.get_user_config

local notify_opts = vc_config.notify_opts

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
    local args = { "vectorcode", "query", "--pipe", "-n", tostring(opts.n_query) }
    if type(query_message) == "string" then
      query_message = { query_message }
    end
    vim.list_extend(args, query_message)

    if opts.exclude_this then
      vim.list_extend(args, { "--exclude", vim.api.nvim_buf_get_name(0) })
    end

    logger.debug("vectorcode.query cmd args: ", args)
    local decoded_response = {}
    local job = vim.system(args, { text = true }, function(out)
      logger.debug("vectorcode.query cmd output: ", out)
      local raw_response
      if out.code == 124 and out.signal == 9 then
        -- killed due to timeout
        raw_response = nil
        if opts.notify then
          vim.schedule(function()
            vim.notify(
              "VectorCode process killed due to timeout.",
              vim.log.levels.WARN,
              notify_opts
            )
          end)
        end
      else
        raw_response = out.stdout
      end

      if raw_response ~= nil and raw_response ~= "" then
        decoded_response =
          vim.json.decode(raw_response, { object = true, array = true })
        if opts.notify then
          vim.notify(
            ("Retrieved %s documents."):format(tostring(#decoded_response)),
            vim.log.levels.INFO,
            notify_opts
          )
        end
        if type(callback) == "function" then
          callback(decoded_response)
        end
        logger.info("vectorcode.query result:\n", decoded_response)
      end
    end)

    if callback == nil then
      job:wait(timeout_ms)
      return decoded_response
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
    if project_root ~= nil then
      vim.list_extend(args, { "--project_root", project_root })
    elseif
      M.check("config", function(obj)
        if obj.code == 0 then
          project_root = obj.stdout
        end
      end)
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
    logger.debug("vectorcode.vectorise cmd args: ", args)
    require("plenary.job")
      :new({
        command = "vectorcode",
        args = args,
        on_exit = function(job, return_code)
          if get_config().notify then
            if return_code == 0 then
              vim.notify(
                "Indexing successful.",
                vim.log.levels.INFO,
                { title = "VectorCode" }
              )
            else
              vim.notify(
                "Indexing failed.",
                vim.log.levels.WARN,
                { title = "VectorCode" }
              )
            end
          end
        end,
      })
      :start()
  end
)

---@param project_root string?
M.update = vc_config.check_cli_wrap(function(project_root)
  logger.info("vectorcode.update: ", project_root)
  local args = { "vectorcode", "update" }
  if
    type(project_root) == "string"
    and vim.uv.fs_stat(vim.fs.normalize(project_root)).type == "directory"
  then
    vim.list_extend(args, { "--project_root", project_root })
  end
  logger.debug("vectorcode.update cmd args: ", args)
  vim.system(args, { stdout = nil, stderr = nil }, function(out)
    logger.debug("vectorcode.update cmd out:\n", out)
    if get_config().notify then
      vim.schedule(function()
        if out.code == 0 then
          vim.notify(
            "VectorCode embeddings has been updated.",
            vim.log.levels.INFO,
            notify_opts
          )
        else
          vim.notify(
            ("Failed to update the embeddings due to the following error:\n%s"):format(
              out.stderr
            ),
            vim.log.levels.ERROR,
            notify_opts
          )
        end
      end)
    end
  end)
  if get_config().notify then
    vim.schedule(function()
      vim.notify("Updating VectorCode embeddings...", vim.log.levels.INFO, notify_opts)
    end)
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
  vim
    .system({ "vectorcode", "check", check_item }, {}, function(out)
      return_code = out.code
      if type(stdout_cb) == "function" then
        stdout_cb(out)
      end
    end)
    :wait()
  return return_code == 0
end

---@return string[]
M.prompts = vc_config.check_cli_wrap(function()
  local vectorcode_prompts = {}
  vim
    .system({ "vectorcode", "prompts", "-p" }, {}, function(out)
      local successful = false
      if out.code == 0 then
        local ok, result =
          pcall(vim.json.decode, out.stdout, { array = true, object = true })

        if ok then
          for _, str in pairs(result) do
            table.insert(vectorcode_prompts, str)
          end
          successful = true
        end
      end
      if not successful then
        vim.schedule_wrap(vim.notify)(
          "Failed to run `vectorcode prompts`. Please ensure your vectorcode CLI is up to date.\n"
            .. tostring(out.stderr),
          vim.log.levels.ERROR,
          notify_opts
        )
      end
    end)
    :wait()
  return vectorcode_prompts
end)

M.setup = vc_config.setup
return M
