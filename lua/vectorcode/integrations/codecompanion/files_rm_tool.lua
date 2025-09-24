---@module "codecompanion"

local cc_common = require("vectorcode.integrations.codecompanion.common")
local vc_config = require("vectorcode.config")
local utils = require("vectorcode.utils")

local default_opts = {
  use_lsp = vc_config.get_user_config().async_backend == "lsp",
  requires_approval = true,
  include_in_toolbox = false,
}

---@alias FilesRmArgs { paths: string[], project_root: string? }

---@param opts VectorCode.CodeCompanion.FilesRmToolOpts
---@return CodeCompanion.Tools
return function(opts)
  opts = vim.tbl_deep_extend("force", default_opts, opts or {})

  local tool_name = "vectorcode_files_rm"
  local job_runner = cc_common.initialise_runner(opts.use_lsp)

  ---@type CodeCompanion.Tools|{}
  return {
    name = tool_name,
    schema = {
      type = "function",
      ["function"] = {
        name = tool_name,
        description = "Remove files from the VectorCode database. The files will remain in the file system.",
        parameters = {
          type = "object",
          properties = {
            paths = {
              type = "array",
              items = { type = "string" },
              description = "Paths to the files to be removed from the database.",
            },
            project_root = {
              type = "string",
              description = [[
The project that the files belong to.
The value should be one of the following:
- One of the paths from the `vectorcode_ls` tool;
- User input;
- `null` (omit this parameter), which means the current project, if found.
                ]],
            },
          },
          required = { "paths" },
        },
        strict = true,
      },
    },
    cmds = {
      ---@param tools CodeCompanion.Tools
      ---@param action VectoriseToolArgs
      ---@return nil|{ status: string, data: string }
      function(tools, action, _, cb)
        local args = { "files", "rm", "--pipe" }
        if action.project_root then
          local project_root = vim.fs.abspath(vim.fs.normalize(action.project_root))
          if utils.is_directory(project_root) then
            vim.list_extend(args, { "--project_root", project_root })
          else
            return { status = "error", data = "Invalid path " .. project_root }
          end
        end
        if action.paths == nil or #action.paths == 0 then
          return { status = "error", data = "Please specify at least one path." }
        end
        vim.list_extend(
          args,
          vim
            .iter(action.paths)
            :filter(
              ---@param item string
              function(item)
                return utils.is_file(item)
              end
            )
            :totable()
        )
        job_runner.run_async(
          args,
          ---@param result VectoriseResult
          function(result, error, code, _)
            if result then
              cb({ status = "success", data = result })
            else
              cb({ status = "error", data = { error = error, code = code } })
            end
          end,
          tools.chat.bufnr
        )
      end,
    },
    output = {
      ---@param self CodeCompanion.Tools.Tool
      prompt = function(self, _)
        return string.format(
          "Remove %d files from VectorCode database?",
          #self.args.paths
        )
      end,
      ---@param self CodeCompanion.Tools.Tool
      ---@param tools CodeCompanion.Tools
      success = function(self, tools, _, _)
        tools.chat:add_tool_output(self, "**VectorCode `files_rm` tool**: successful.")
      end,
    },
  }
end
