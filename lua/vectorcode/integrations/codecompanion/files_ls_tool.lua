---@module "codecompanion"

local cc_common = require("vectorcode.integrations.codecompanion.common")
local vc_config = require("vectorcode.config")

local default_opts = {
  use_lsp = vc_config.get_user_config().async_backend == "lsp",
  requires_approval = false,
  include_in_toolbox = false,
}

---@param opts VectorCode.CodeCompanion.FilesLsToolOpts
---@return CodeCompanion.Tools.Tool
return function(opts)
  opts = vim.tbl_deep_extend("force", default_opts, opts or {})
  local job_runner =
    require("vectorcode.integrations.codecompanion.common").initialise_runner(
      opts.use_lsp
    )
  local tool_name = "vectorcode_files_ls"
  ---@type CodeCompanion.Tools.Tool|{}
  return {
    name = tool_name,
    cmds = {
      ---@param tools CodeCompanion.Tools
      ---@param action {project_root: string}
      ---@return nil|{ status: string, data: string }
      function(tools, action, _, cb)
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
        end, tools.chat.bufnr)
      end,
    },
    schema = {
      type = "function",
      ["function"] = {
        name = tool_name,
        description = [[
Retrieve a list of files that have been added to the database for a given project.
**ABSOLUTE PATHS** in the results indicate that the files are OUTSIDE of the current working directories and you can **ONLY** access them via the VectorCode tools.
**RELATIVE PATHS** in the results indicate that the files are INSIDE the current project. You can use VectorCode tools or any other tools that the user provided to interact with them.
          ]],
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
      ---@param tools CodeCompanion.Tools
      ---@param stdout string[][]
      success = function(_, tools, _, stdout)
        stdout = stdout[#stdout]
        local user_message
        for i, col in ipairs(stdout) do
          if i == 1 then
            user_message =
              string.format("**VectorCode `files_ls` Tool**: Found %d files.", #stdout)
          else
            user_message = ""
          end
          tools.chat:add_tool_output(
            tools.tool,
            string.format("<path>%s</path>", cc_common.cleanup_path(col)),
            user_message
          )
        end
      end,
    },
  }
end
