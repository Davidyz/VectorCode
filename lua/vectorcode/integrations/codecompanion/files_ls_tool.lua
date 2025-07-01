---@module "codecompanion"

local cc_common = require("vectorcode.integrations.codecompanion.common")

---@param opts VectorCode.CodeCompanion.FilesLsToolOpts
---@return CodeCompanion.Agent.Tool
return function(opts)
  local job_runner =
    require("vectorcode.integrations.codecompanion.common").initialise_runner(
      opts.use_lsp
    )
  local tool_name = "vectorcode_files_ls"
  ---@type CodeCompanion.Agent.Tool|{}
  return {
    name = tool_name,
    cmds = {
      ---@param agent CodeCompanion.Agent
      ---@param action {project_root: string}
      ---@return nil|{ status: string, data: string }
      function(agent, action, _, cb)
        local args = { "files", "ls", "--pipe" }
        if action ~= nil then
          action.project_root = action.project_root
            or vim.fs.root(0, { ".vectorcode", ".git" })
          if action.project_root ~= nil then
            action.project_root = vim.fs.normalize(action.project_root)
            local stat = vim.uv.fs_stat(action.project_root)
            if stat and stat.type == "directory" then
              vim.list_extend(args, { "--project_root", action.project_root })
            end
          end
        end
        job_runner.run_async(args, function(result, error)
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
        description = "Retrieve a list of files that have been added to the database for a given project.",
        parameters = {
          type = "object",
          properties = {
            project_root = {
              type = "string",
              description = "The project for which the indexed files will be listed. Leave this empty for the current project.",
            },
          },
        },
      },
    },
    output = {
      ---@param agent CodeCompanion.Agent
      ---@param stdout string[][]
      success = function(_, agent, _, stdout)
        stdout = stdout[1]
        local user_message
        for i, col in ipairs(stdout) do
          if i == 1 then
            user_message =
              string.format("**VectorCode `files_ls` Tool**: Found %d files.", #stdout)
          else
            user_message = ""
          end
          agent.chat:add_tool_output(
            agent.tool,
            string.format("<path>%s</path>", col),
            user_message
          )
        end
      end,
    },
  }
end
