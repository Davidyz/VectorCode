---@module "codecompanion"

return {
  chat = {
    ---@param subcommand sub_cmd
    ---@param opts VectorCode.CodeCompanion.ToolOpts
    ---@return CodeCompanion.Tools.Tool
    make_tool = function(subcommand, opts)
      local has = require("codecompanion").has
      if has ~= nil and has("function-calling") then
        return require(
          string.format("vectorcode.integrations.codecompanion.%s_tool", subcommand)
        )(opts)
      else
        error("Unsupported version of codecompanion!")
      end
    end,
  },
  prompts = {
    require("vectorcode.integrations.codecompanion.prompts.nvim"),
  },
}
