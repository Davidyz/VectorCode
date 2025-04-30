---@module "codecompanion"

---@class VectorCode.CodeCompanion.ExtensionOpts
---@field add_tools boolean|string|nil
---@field tool_opts VectorCode.CodeCompanion.ToolOpts
---@field add_slash_command boolean

local vc_config = require("vectorcode.config")
local logger = vc_config.logger

---@type CodeCompanion.Extension
local M = {
  ---@param opts VectorCode.CodeCompanion.ExtensionOpts
  setup = vc_config.check_cli_wrap(function(opts)
    opts = vim.tbl_deep_extend(
      "force",
      { add_tool = true, add_slash_command = false },
      opts or {}
    )
    logger.info("Received codecompanion extension opts:\n", opts)
    local cc_config = require("codecompanion.config").config
    local cc_integration = require("vectorcode.integrations").codecompanion.chat
    if opts.add_tool then
      local tool_name = "vectorcode"
      if type(opts.add_tool) == "string" then
        tool_name = tostring(opts.add_tool)
      end
      if cc_config.strategies.chat.tools[tool_name] ~= nil then
        vim.notify(
          "There's an existing tool named `vectorcode`. Please either remove it or rename it.",
          vim.log.levels.ERROR,
          vc_config.notify_opts
        )
        logger.warn(
          string.format(
            "Not creating a tool because there is an existing tool named %s.",
            tool_name
          )
        )
      else
        cc_config.strategies.chat.tools[tool_name] = {
          description = "Run VectorCode to retrieve the project context.",
          callback = cc_integration.make_tool(opts.tool_opts),
        }
        logger.info(string.format("%s tool has been created.", tool_name))
      end
    end
    if opts.add_slash_command then
      local command_name = "vectorcode"
      if type(opts.add_slash_command) == "string" then
        command_name = tostring(opts.add_slash_command)
      end
      if cc_config.strategies.chat.slash_commands[command_name] ~= nil then
        vim.notify(
          "There's an existing slash command named `vectorcode`. Please either remove it or rename it.",
          vim.log.levels.ERROR,
          vc_config.notify_opts
        )
        logger.warn(
          string.format(
            "Not creating a command because there is an existing slash command named %s.",
            command_name
          )
        )
      else
        cc_config.strategies.chat.slash_commands[command_name] =
          cc_integration.make_slash_command()
        logger.info(string.format("%s command has been created.", command_name))
      end
    end
  end),
}

return M
