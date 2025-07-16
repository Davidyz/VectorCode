---@module "codecompanion"

local cc_common = require("vectorcode.integrations.codecompanion.common")
local vc_config = require("vectorcode.config")

local default_opts = {
  use_lsp = vc_config.get_user_config().async_backend == "lsp",
  requires_approval = true,
  include_in_toolbox = false,
}

---@alias FilesRmArgs { paths: string[], project_root: string }

---@param opts VectorCode.CodeCompanion.FilesRmToolOpts
---@return CodeCompanion.Agent.Tool
return function(opts)
  opts = vim.tbl_deep_extend("force", default_opts, opts or {})

  local tool_name = "vectorcode_files_rm"
  local job_runner = cc_common.initialise_runner(opts.use_lsp)

  ---@type CodeCompanion.Agent.Tool|{}
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
              description = "The project that the files belong to. Either use a path from the `vectorcode_ls` tool, or leave empty to use the current git project. If the user did not specify a path, use empty string for this parameter.",
            },
          },
          required = { "paths", "project_root" },
          additionalProperties = false,
        },
        strict = true,
      },
    },
    cmds = {
      ---@param agent CodeCompanion.Agent
      ---@param action VectoriseToolArgs
      ---@return nil|{ status: string, data: string }
      function(agent, action, _, cb)
        local args = { "files", "rm", "--pipe" }
        local project_root = vim.fs.abspath(vim.fs.normalize(action.project_root or ""))
        if project_root ~= "" then
          local stat = vim.uv.fs_stat(project_root)
          if stat and stat.type == "directory" then
            vim.list_extend(args, { "--project_root", project_root })
          else
            return { status = "error", data = "Invalid path " .. project_root }
          end
        else
          project_root = vim.fs.root(".", { ".vectorcode", ".git" }) or ""
          if project_root == "" then
            return {
              status = "error",
              data = "Please specify a project root. You may use the `vectorcode_ls` tool to find a list of existing projects.",
            }
          end
        end
        if project_root ~= "" then
          action.project_root = project_root
        end
        vim.list_extend(
          args,
          vim
            .iter(action.paths)
            :filter(
              ---@param item string
              function(item)
                local stat = vim.uv.fs_stat(item)
                if stat and stat.type == "file" then
                  return true
                else
                  return false
                end
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
          agent.chat.bufnr
        )
      end,
    },
    output = {
      ---@param self CodeCompanion.Agent.Tool
      prompt = function(self, _)
        return string.format(
          "Remove %d files from VectorCode database?",
          #self.args.paths
        )
      end,
      ---@param self CodeCompanion.Agent.Tool
      ---@param agent CodeCompanion.Agent
      success = function(self, agent, _, _)
        agent.chat:add_tool_output(self, "**VectorCode `files_rm` tool**: successful.")
      end,
    },
  }
end
