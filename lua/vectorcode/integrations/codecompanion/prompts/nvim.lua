---@module "codecompanion"

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

      prompts = {
        {
          role = constants.SYSTEM_ROLE,
          content = string.format(
            [[You are an neovim expert.
You will be given tools to index and query from the neovim runtime library.
This will include lua APIs for the neovim runtime and documentation for the neovim build that the user is running.
Use the tools to explain the user's questions related to neovim.
The runtime files are stored in `%s`.
You can ONLY use the vectorcode tools to interact with these files or directory.
DO NOT attempt to read from or write into this directory.
]],
            rtp
          ),
        },
        {
          role = constants.USER_ROLE,
          content = string.format(
            [[
You are given the @{vectorcode_vectorise} and @{vectorcode_query} tools.
Vectorrise all lua files and `*.txt` files under this directory using the `vectorcode_vectorise` tool. Use wildcards to match all lua and `txt` files. Use absolute paths when supplying paths to the vectorcode_vectorise tool;
Use `%s` as the value of the `project_root` argument;
When you're done, I'll be asking you questions related to these documents.
Use the vectorcode_query tool to query from the project root and answer my question.
]],
            rtp,
            rtp
          ),
        },
      },
    },
  }
end)
