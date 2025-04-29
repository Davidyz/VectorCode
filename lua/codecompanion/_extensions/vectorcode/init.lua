---@module "codecompanion"

---@class VectorCode.CodeCompanion.ExtensionOpts
---@field add_tools boolean
---@field tool_opts VectorCode.CodeCompanion.ToolOpts

local vc_config = require("vectorcode.config")

---@type CodeCompanion.Extension
local M = {
  ---@param opts VectorCode.CodeCompanion.ExtensionOpts
  setup = vc_config.check_cli_wrap(function(opts)
    local cc_config = require("codecompanion.config").config
    if opts.add_tools then
      cc_config.strategies.chat.tools["vectorcode"] = {
        description = "Run VectorCode to retrieve the project context.",
        callback = require("vectorcode.integrations").codecompanion.chat.make_tool(
          opts.tool_opts
        ),
      }
    end
  end),
}

return M
