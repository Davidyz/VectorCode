---@module "codecompanion"

---@alias CodeCompanion.Tools.Tool.CmdFunc fun(self: CodeCompanion.Tools.Tool, args: any, input: table, cb: function|nil)

local function should_use_new_api()
  ---@type vim.Version|nil
  local cc_version = nil
  local _version = require("codecompanion").version
  if type(_version) == "function" then
    _version = _version()
  end
  if type(_version) == "string" then
    cc_version = vim.version.parse(_version)
  end
  return cc_version and cc_version.major >= 19
end

return {
  chat = {
    ---@param subcommand VectorCode.CodeCompanion.SubCommand
    ---@param opts VectorCode.CodeCompanion.ToolOpts
    ---@return CodeCompanion.Tools.Tool
    make_tool = function(subcommand, opts)
      local has = require("codecompanion").has
      if has ~= nil and has("function-calling") then
        local tool_cb = require(
          string.format("vectorcode.integrations.codecompanion.%s_tool", subcommand)
        )(opts)
        local tool_transformer =
          require("vectorcode.integrations").codecompanion.chat.transform
        tool_cb.cmds = tool_transformer.cmd(tool_cb.cmds)
        tool_cb.output = tool_transformer.output(tool_cb.output)

        local require_approval = opts.requires_approval or opts.require_approval_before
        local tool_info = {
          description = string.format("Run VectorCode %s tool", subcommand),
          opts = {
            requires_approval = require_approval,
            require_approval_before = require_approval,
          },
        }
        if should_use_new_api() then
          return vim.tbl_deep_extend("force", tool_info, tool_cb)
        else
          tool_info.callback = tool_cb
          return tool_info
        end
      else
        error("Unsupported version of codecompanion!")
      end
    end,

    --- compatibility shims for the new tool API.
    transform = {
      ---@param orig_cmd CodeCompanion.Tools.Tool.CmdFunc|CodeCompanion.Tools.Tool.CmdFunc[]
      cmd = function(orig_cmd)
        if type(orig_cmd) == "table" then
          return vim
            .iter(orig_cmd)
            :map(require("vectorcode.integrations").codecompanion.chat.transform.cmd)
            :totable()
        end

        return function(self, call_args, input, cb)
          if cb == nil then
            -- `cb` would be included in `input`
            return orig_cmd(self, call_args, input.input, input.output_cb)
          else
            return orig_cmd(self, call_args, input, cb)
          end
        end
      end,

      output = function(orig_cmd)
        if type(orig_cmd) == "table" then
          local new_handlers = {}
          for k, v in pairs(orig_cmd) do
            new_handlers[k] =
              require("vectorcode.integrations").codecompanion.chat.transform.output(v)
          end
          return new_handlers
        end
        return function(self, ...)
          local extra_args = { ... }
          if #extra_args == 2 then
            -- new API
            local output, _meta = unpack(extra_args)
            return orig_cmd(self, _meta.tools, _meta.cmd, output)
          else
            return orig_cmd(self, ...)
          end
        end
      end,
    },

    prompts = require("vectorcode.integrations.codecompanion.prompts"),
  },
}
