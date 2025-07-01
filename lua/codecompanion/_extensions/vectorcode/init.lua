---@module "codecompanion"

---@alias sub_cmd "ls"|"query"|"vectorise"|"files_ls"|"files_rm"

---@class VectorCode.CodeCompanion.ExtensionOpts
--- A table where the keys are the subcommand name (`ls`, `query`, `vectorise`)
--- and the values are their config options.
---@field tool_opts table<sub_cmd, VectorCode.CodeCompanion.ToolOpts>
--- Whether to add a tool group that contains all vectorcode tools.
---@field tool_group VectorCode.CodeCompanion.ToolGroupOpts

local vc_config = require("vectorcode.config")
local logger = vc_config.logger

local use_lsp = vc_config.get_user_config().async_backend == "lsp"

---@type VectorCode.CodeCompanion.ExtensionOpts|{}
local default_extension_opts = {
  tool_opts = {
    ls = { use_lsp = use_lsp, requires_approval = false, include_in_toolbox = true },
    query = { use_lsp = use_lsp, requires_approval = false, include_in_toolbox = true },
    vectorise = {
      use_lsp = use_lsp,
      requires_approval = true,
      include_in_toolbox = true,
    },
    files_ls = {
      use_lsp = use_lsp,
      requires_approval = false,
      include_in_toolbox = false,
    },
    files_rm = {
      use_lsp = use_lsp,
      requires_approval = true,
      include_in_toolbox = false,
    },
  },
  tool_group = { enabled = true, collapse = true, extras = {} },
}

---@type sub_cmd[]
local valid_tools = { "ls", "query", "vectorise", "files_ls", "files_rm" }

---@type CodeCompanion.Extension
local M = {
  ---@param opts VectorCode.CodeCompanion.ExtensionOpts
  setup = vc_config.check_cli_wrap(function(opts)
    opts = vim.tbl_deep_extend("force", default_extension_opts, opts or {})
    logger.info("Received codecompanion extension opts:\n", opts)
    local cc_config = require("codecompanion.config").config
    local cc_integration = require("vectorcode.integrations").codecompanion.chat
    for _, sub_cmd in pairs(valid_tools) do
      local tool_name = string.format("vectorcode_%s", sub_cmd)
      if cc_config.strategies.chat.tools[tool_name] ~= nil then
        vim.notify(
          string.format(
            "There's an existing tool named `%s`. Please either remove it or rename it.",
            tool_name
          ),
          vim.log.levels.ERROR,
          vc_config.notify_opts
        )
        logger.warn(
          string.format(
            "Not creating this tool because there is an existing tool named %s.",
            tool_name
          )
        )
      else
        cc_config.strategies.chat.tools[tool_name] = {
          description = string.format("Run VectorCode %s tool", sub_cmd),
          callback = cc_integration.make_tool(sub_cmd, opts.tool_opts[sub_cmd]),
          opts = { requires_approval = opts.tool_opts[sub_cmd].requires_approval },
        }
        logger.info(string.format("%s tool has been created.", tool_name))
      end
    end

    if opts.tool_group.enabled then
      local included_tools = vim
        .iter(valid_tools)
        :filter(function(cmd_name)
          return opts.tool_opts[cmd_name].include_in_toolbox
        end)
        :map(function(s)
          return "vectorcode_" .. s
        end)
        :totable()
      if opts.tool_group.extras and not vim.tbl_isempty(opts.tool_group.extras) then
        vim.list_extend(included_tools, opts.tool_group.extras)
      end
      logger.info(
        string.format(
          "Loading the following tools into `vectorcode_toolbox` tool group:\n%s",
          vim.inspect(included_tools)
        )
      )
      cc_config.strategies.chat.tools.groups["vectorcode_toolbox"] = {
        opts = { collapse_tools = opts.tool_group.collapse },
        description = "Use VectorCode to automatically build and retrieve repository-level context.",
        tools = included_tools,
      }
    end
  end),
}

return M
