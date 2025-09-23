local M = {}

local vc_config = require("vectorcode.config")

local utils = require("vectorcode.utils")

---@param path string[]|string path to files or wildcards.
---@param project_root? string
---@param callback? VectorCode.JobRunner.Callback
function M.vectorise_files(path, project_root, callback)
  if type(path) == "string" then
    path = { path }
  end
  assert(not vim.tbl_isempty(path), "`path` cannot be empty")

  local jobrunner =
    require("vectorcode.integrations.codecompanion.common").initialise_runner(
      vc_config.get_user_config().async_backend == "lsp"
    )

  local args = { "vectorise", "--pipe" }
  if project_root then
    vim.list_extend(args, { "--project_root", project_root })
  end
  vim.list_extend(args, path)
  jobrunner.run_async(args, function(result, error, code, signal)
    if type(callback) == "function" then
      callback(result, error, code, signal)
    end
  end, 0)
end

---@class VectorCode.CodeCompanion.PromptFactory.Opts
---@field name string? human-readable name of this prompt
---@field project_root string|(fun():string) project_root of the files to be added to the database
---Paths to the files in the local directory to be added to the database.
---
---These should either be absolute paths, or relative to the project root.
---@field file_patterns string[]|(fun():string[])
---See https://codecompanion.olimorris.dev/extending/prompts.html#recipe-2-using-context-in-your-prompts
---
---Note: If a system prompt is set here, your default chat system prompt will be ignored.
---@field system_prompt? string|fun(context:table):string
---This contains some preliminary messages (filled into the chat buffer) that tells the LLM about the task.
---If you're overwriting the default message, make sure to include the tool (`@{vectorcode_query}`).
---
---See https://codecompanion.olimorris.dev/extending/prompts.html#recipe-2-using-context-in-your-prompts
---@field user_prompt? string|fun(context:table):string

---@param opts VectorCode.CodeCompanion.PromptFactory.Opts
function M.register_prompt(opts)
  opts = vim.deepcopy(opts)

  if type(opts.file_patterns) == "function" then
    opts.file_patterns = opts.file_patterns()
  end

  assert(
    ---@diagnostic disable-next-line: param-type-mismatch
    type(opts.project_root) == "string" and utils.is_directory(opts.project_root),
    string.format("`%s` is not a valid directory.", opts.project_root)
  )
  assert(
    ---@diagnostic disable-next-line: param-type-mismatch
    opts.file_patterns ~= nil and (not vim.tbl_isempty(opts.file_patterns)),
    "Recieved empty path specs."
  )

  assert(type(opts.name) == "string", "`name` cannot be `nil`.")

  local constants = require("codecompanion.config").config.constants
  local prompts = {}

  if opts.system_prompt then
    table.insert(
      prompts,
      { role = constants.SYSTEM_ROLE, content = opts.system_prompt }
    )
  end
  table.insert(prompts, #prompts + 1, {
    role = constants.USER_ROLE,
    content = opts.user_prompt
      or string.format(
        [[I have some questions about the documents under the `%s` directory.
The files have been added to the database and can be searched by calling the @{vectorcode_query} tool.
When you call the tool, use `%s` as the value for the argument `project_root`.
Use the information returned by the tool to answer my questions, and cite the sources when appropriate.
If you need more information, call the tool with different search keywords or ask for more context and/or tools.

Here's my question:  

- ]],
        opts.project_root,
        opts.project_root
      ),
  })
  return {
    name = opts.name,
    strategy = "chat",
    opts = {
      ignore_system_prompt = opts.system_prompt ~= nil,
      pre_hook = function()
        if vc_config.get_user_config().notify then
          vim.notify(
            string.format("Adding files under `%s` to the database.", opts.project_root),
            vim.log.levels.INFO,
            vc_config.notify_opts
          )
        end
        M.vectorise_files(
          vim
            .iter(opts.file_patterns)
            :map(function(p)
              if vim.fn.isabsolutepath(p) == 1 then
                return p
              else
                return vim.fs.joinpath(opts.project_root, p)
              end
            end)
            :totable(),
          opts.project_root,
          function(result, err, _, _)
            if result ~= nil and not vim.tbl_isempty(result) then
              vim.schedule_wrap(vim.notify)(
                string.format(
                  "Vectorised %d new files.",
                  result.add or 0,
                  opts.project_root
                ),
                vim.log.levels.INFO,
                vc_config.notify_opts
              )
            elseif err ~= nil then
              err = utils.flatten_table_to_string(err)
              if err ~= "" then
                vim.schedule_wrap(vim.notify)(
                  err,
                  vim.log.levels.WARN,
                  vc_config.notify_opts
                )
              end
            end
          end
        )
      end,
    },
    prompts = prompts,
  }
end
return M
