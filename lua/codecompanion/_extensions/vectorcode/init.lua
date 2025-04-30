---@module "codecompanion"

---@class VectorCode.CodeCompanion.ExtensionOpts
---@field add_tools boolean
---@field tool_opts VectorCode.CodeCompanion.ToolOpts
---@field add_slash_command boolean

local vc_config = require("vectorcode.config")

---@type CodeCompanion.Extension
local M = {
  ---@param opts VectorCode.CodeCompanion.ExtensionOpts
  setup = vc_config.check_cli_wrap(function(opts)
    opts = vim.tbl_deep_extend(
      "force",
      { add_tools = true, add_slash_command = false },
      opts or {}
    )
    local cc_config = require("codecompanion.config").config
    local cc_integration = require("vectorcode.integrations").codecompanion.chat
    if opts.add_tools then
      cc_config.strategies.chat.tools["vectorcode"] = {
        description = "Run VectorCode to retrieve the project context.",
        callback = cc_integration.make_tool(opts.tool_opts),
      }
    end
    if opts.add_slash_command then
      cc_config.strategies.chat.slash_commands["vectorcode"] =
        cc_integration.make_slash_command()
    end
  end),
}

return M
