---@module "codecompanion"

local cc_common = require("vectorcode.integrations.codecompanion.common")
local vectorcode = require("vectorcode")

---@param opts VectorCode.CodeCompanion.LsToolOpts
---@return CodeCompanion.Agent.Tool
return function(opts)
  opts = cc_common.get_ls_tool_opts(opts)
  local job_runner =
    require("vectorcode.integrations.codecompanion.common").initialise_runner(
      opts.use_lsp
    )
  local tool_name = "vectorcode_ls"
  ---@type CodeCompanion.Agent.Tool|{}
  return {
    name = tool_name,
    cmds = {
      ---@param agent CodeCompanion.Agent
      ---@return nil|{ status: string, data: string }
      function(agent, _, _, cb)
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
        end, agent.chat.bufnr)
      end,
    },
    schema = {
      type = "function",
      ["function"] = {
        name = tool_name,
        description = string.format(
          "Retrieve a list of projects accessible via the VectorCode tools.\n%s",
          table.concat(vectorcode.prompts("ls"), "\n")
        ),
      },
    },
    output = {
      ---@param agent CodeCompanion.Agent
      ---@param stdout VectorCode.LsResult[][]
      success = function(_, agent, _, stdout)
        stdout = stdout[1]
        local user_message
        for i, col in ipairs(stdout) do
          if i == 1 then
            user_message =
              string.format("**VectorCode `ls` Tool**: Found %d collections.", #stdout)
          else
            user_message = ""
          end
          agent.chat:add_tool_output(
            agent.tool,
            string.format("<collection>%s</collection>", col["project-root"]),
            user_message
          )
        end
      end,
    },
  }
end
