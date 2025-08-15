---@module "codecompanion"

local config = require("vectorcode.config")

---@param rtp string path to the project_root
local prepare = function(rtp)
  if config.get_user_config().notify then
    vim.notify(
      "Vectorising neovim runtime files...",
      vim.log.levels.INFO,
      config.notify_opts
    )
  end
  require("vectorcode.integrations.codecompanion.prompts").vectorise_files(
    { vim.fs.joinpath(rtp, "lua/**/*.lua"), vim.fs.joinpath(rtp, "doc/**/*.txt") },
    rtp,
    function(result, _, _, _)
      if
        result ~= nil
        and not vim.tbl_isempty(result)
        and config.get_user_config().notify
      then
        vim.schedule_wrap(vim.notify)(
          string.format("Added %d files to the database!", result.add or 0),
          vim.log.levels.INFO,
          config.notify_opts
        )
      end
    end
  )
end

return require("vectorcode.config").check_cli_wrap(function()
  local constants = require("codecompanion.config").config.constants
  local rtp = vim.fs.normalize(vim.env.VIMRUNTIME)

  if not rtp then
    return error("Failed to locate the neovim runtime!", vim.log.levels.ERROR)
  end
  local stat = vim.uv.fs_stat(rtp)
  if not stat or stat.type ~= "directory" then
    return error(
      string.format(
        "$VIMRUNTIME is %s, which is not a valid directory to be accessed.",
        rtp
      ),
      vim.log.levels.ERROR
    )
  end

  return {
    name = "Neovim Assistant",
    prompts = {
      strategy = "chat",
      description = "Use VectorCode to index and query from neovim documentation and lua runtime.",
      opts = {
        pre_hook = function()
          prepare(rtp)
        end,
        ignore_system_prompt = true,
      },
      prompts = {
        {
          role = constants.SYSTEM_ROLE,
          content = string.format(
            [[You are an neovim expert.
You will be given tools to index and query from the neovim runtime library.
This will include lua APIs for the neovim runtime and documentation for the neovim build that the user is running.
The runtime files are stored in `%s`.
You can ONLY use the vectorcode tools to interact with these files or directory.
DO NOT attempt to read from or write into this directory.
When the user asked a question that is not part of a previous query tool call, make a new query using new keywords that are directly relevant to the new question.
If the tool returns an error that says the collection doesn't exist, it's because the files are still being indexed, and you should ask the user to wait for it to finish.
Do not cite information that was not part of the provided context or tool output.
]],
            rtp
          ),
        },
        {
          role = constants.USER_ROLE,
          content = string.format(
            [[
You are given the @{vectorcode_query} tool.
Use `%s` as the value of the `project_root` argument;
I'll be asking you questions related to these documents.
Use the vectorcode_query tool to query from the project root and answer my question.

Here's my question:

- 
]],
            rtp
          ),
        },
      },
    },
  }
end)
