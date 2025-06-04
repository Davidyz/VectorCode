---@module "CopilotChat"

---@class VectorCode.CopilotChat.ContextOpts
---@field max_num number?
---@field use_lsp boolean?

local async = require("plenary.async")
local vc_config = require("vectorcode.config")
local logger = vc_config.logger
local notify_opts = vc_config.notify_opts
local check_cli_wrap = vc_config.check_cli_wrap
local job_runner = nil

---@param use_lsp boolean
local function get_runner(use_lsp)
  if job_runner == nil then
    if use_lsp then
      job_runner = require("vectorcode.jobrunner.lsp")
    end
    if job_runner == nil then
      job_runner = require("vectorcode.jobrunner.cmd")
      if use_lsp then
        vim.schedule_wrap(vim.notify)(
          "Failed to initialise the LSP runner. Falling back to cmd runner.",
          vim.log.levels.WARN,
          notify_opts
        )
      end
    end
  end
  return job_runner
end

---@param args string[]
---@param use_lsp boolean
---@param bufnr integer
---@async
local run_job = async.wrap(function(args, use_lsp, bufnr, callback)
  local runner = get_runner(use_lsp)
  assert(runner ~= nil, "Failed to initialize the runner!")
  runner.run_async(args, callback, bufnr)
end, 4)

---@param opts VectorCode.CopilotChat.ContextOpts?
---@return CopilotChat.config.context
local make_context_provider = check_cli_wrap(function(opts)
  opts = vim.tbl_deep_extend("force", {
    max_num = 5,
    use_lsp = vc_config.get_user_config().async_backend == "lsp",
  }, opts or {})
  logger.info("Creating CopilotChat context provider with the following opts:\n", opts)

  local utils = require("CopilotChat.utils")

  return {
    description = [[This gives you the ability to access the repository to find information that you may need to assist the user. Supports input (query).

- **Use at your discretion** when you feel you don't have enough information about the repository or project.
- **Don't escape** special characters.
- If a class, type or function has been imported from another file, this context may be able to find its source. Add the name of the imported symbol to the query.
- The embeddings are mostly generated from source code, so using keywords that may be present in source code may help with the retrieval.
- Avoid retrieving one single file because the retrieval mechanism may not be very accurate.
= If a query failed to retrieve desired results, a new attempt should use different keywords that are orthogonal to the previous ones but with similar meanings
- Do not use exact query keywords that you have used in a previous context call in the conversation, unless the user instructed otherwise
]],

    input = function(callback)
      vim.ui.input({
        prompt = "Enter query> ",
      }, callback)
    end,

    resolve = function(input, source, prompt)
      if not input or input == "" then
        input = prompt
      end

      local args = {
        "query",
        "--pipe",
        "-n",
        tostring(opts.max_num),
        '"' .. input .. '"',
      }

      local cwd = source.cwd()
      local try_root = vim.fs.root(cwd, ".vectorcode") or vim.fs.root(cwd, ".git")
      if try_root ~= nil then
        vim.list_extend(args, { "--project_root", try_root })
      end
      logger.info("CopilotChat ctx provider called with the following args: ", args)
      local result, err = run_job(args, opts.use_lsp, source.bufnr)
      if utils.empty(result) and err then
        error(utils.make_string(err))
      end

      utils.schedule_main()
      return vim.tbl_map(function(item)
        return {
          content = item.document,
          filename = item.path,
          filetype = utils.filetype(item.path),
        }
      end, result)
    end,
  }
end)

return {
  make_context_provider = make_context_provider,
}
