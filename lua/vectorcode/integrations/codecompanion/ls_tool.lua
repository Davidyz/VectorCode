---@module "codecompanion"

local cc_common = require("vectorcode.integrations.codecompanion.common")
local vc_config = require("vectorcode.config")
local logger = vc_config.logger

---@type VectorCode.CodeCompanion.LsToolOpts
local default_ls_options = {
  use_lsp = vc_config.get_user_config().async_backend == "lsp",
  requires_approval = false,
  include_in_toolbox = true,
}

---@param opts VectorCode.CodeCompanion.LsToolOpts|{}|nil
---@return VectorCode.CodeCompanion.LsToolOpts
local get_ls_tool_opts = function(opts)
  opts = vim.tbl_deep_extend("force", default_ls_options, opts or {})
  logger.info(
    string.format(
      "Loading `vectorcode_ls` with the following opts:\n%s",
      vim.inspect(opts)
    )
  )
  return opts
end

---@param opts VectorCode.CodeCompanion.LsToolOpts
---@return CodeCompanion.Tools.Tool
return function(opts)
  opts = get_ls_tool_opts(opts)
  local job_runner =
    require("vectorcode.integrations.codecompanion.common").initialise_runner(
      opts.use_lsp
    )
  local tool_name = "vectorcode_ls"
  ---@type CodeCompanion.Tools.Tool|{}
  return {
    name = tool_name,
    cmds = {
      ---@param tools CodeCompanion.Tools
      ---@return nil|{ status: string, data: string }
      function(tools, _, _, cb)
        job_runner.run_async({ "ls", "--pipe" }, function(result, error)
          if vim.islist(result) and #result > 0 then
            cb({ status = "success", data = result })
          else
            if type(error) == "table" then
              error = cc_common.flatten_table_to_string(error)
            end
            cb({
              status = "error",
              data = error,
            })
          end
        end, tools.chat.bufnr)
      end,
    },
    schema = {
      type = "function",
      ["function"] = {
        name = tool_name,
        description = [[
Retrieve a list of projects accessible via the VectorCode tools.
Where relevant, use paths from this tool as the `project_root` parameter in other vectorcode tools.
]],
        parameters = {
          -- make anthropic models happy.
          type = "object",
          properties = vim.empty_dict(),
          required = {},
          additionalProperties = false,
        },
      },
    },
    output = {
      ---@param tools CodeCompanion.Tools
      ---@param stdout VectorCode.LsResult[][]
      success = function(_, tools, _, stdout)
        stdout = stdout[#stdout]
        local user_message
        for i, col in ipairs(stdout) do
          if i == 1 then
            user_message =
              string.format("**VectorCode `ls` Tool**: Found %d collections.", #stdout)
          else
            user_message = ""
          end
          tools.chat:add_tool_output(
            tools.tool,
            string.format("<collection>%s</collection>", col["project-root"]),
            user_message
          )
        end
      end,
    },
  }
end
