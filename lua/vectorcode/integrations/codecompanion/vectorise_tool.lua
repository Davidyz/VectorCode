---@module "codecompanion"

local cc_common = require("vectorcode.integrations.codecompanion.common")
local vc_config = require("vectorcode.config")
local logger = vc_config.logger

---@alias VectoriseToolArgs { paths: string[], project_root: string }

---@alias VectoriseResult { add: integer, update: integer, removed: integer }

---@type VectorCode.CodeCompanion.VectoriseToolOpts
local default_vectorise_options = {
  use_lsp = vc_config.get_user_config().async_backend == "lsp",
  requires_approval = true,
  include_in_toolbox = true,
}

---@param opts VectorCode.CodeCompanion.VectoriseToolOpts|{}|nil
---@return VectorCode.CodeCompanion.VectoriseToolOpts
local get_vectorise_tool_opts = function(opts)
  opts = vim.tbl_deep_extend("force", default_vectorise_options, opts or {})
  logger.info(
    string.format(
      "Loading `vectorcode_vectorise` with the following opts:\n%s",
      vim.inspect(opts)
    )
  )
  return opts
end

---@param opts VectorCode.CodeCompanion.VectoriseToolOpts|{}|nil
---@return CodeCompanion.Tools
return function(opts)
  opts = get_vectorise_tool_opts(opts)
  local tool_name = "vectorcode_vectorise"
  local job_runner = cc_common.initialise_runner(opts.use_lsp)

  ---@type CodeCompanion.Tools|{}
  return {
    name = tool_name,
    schema = {
      type = "function",
      ["function"] = {
        name = tool_name,
        description = [[
Vectorise files in a project so that they'll be available from the `vectorcode_query` tool.
The paths should be accurate (DO NOT ASSUME A PATH EXIST) and case case-sensitive.
]],
        parameters = {
          type = "object",
          properties = {
            paths = {
              type = "array",
              items = { type = "string" },
              description = "Paths to the files to be vectorised. DO NOT use directories for this parameter.",
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
      ---@param tools CodeCompanion.Tools
      ---@param action VectoriseToolArgs
      ---@return nil|{ status: string, data: string }
      function(tools, action, _, cb)
        local args = { "vectorise", "--pipe" }
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
        if
          vim.iter(action.paths):any(function(p)
            local stat = vim.uv.fs_stat(vim.fs.normalize(p))
            if stat and stat.type == "directory" then
              return true
            end
            return false
          end)
        then
          return {
            status = "error",
            data = "Please only supply paths to files as the `paths` parameter, not directories.",
          }
        end

        vim.list_extend(args, action.paths)
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
        return string.format("Vectorise %d files with VectorCode?", #self.args.paths)
      end,
      ---@param self CodeCompanion.Tools.Tool
      ---@param tools CodeCompanion.Tools
      ---@param cmd VectoriseToolArgs
      error = function(self, tools, cmd, stderr)
        logger.error(
          ("CodeCompanion tool with command %s thrown with the following error: %s"):format(
            vim.inspect(cmd),
            vim.inspect(stderr)
          )
        )
        stderr = cc_common.flatten_table_to_string(stderr)
        tools.chat:add_tool_output(
          self,
          string.format("**VectorCode Vectorise Tool: %s", stderr)
        )
      end,
      ---@param self CodeCompanion.Tools.Tool
      ---@param tools CodeCompanion.Tools
      ---@param cmd VectoriseToolArgs
      ---@param stdout VectorCode.VectoriseResult[]
      success = function(self, tools, cmd, stdout)
        stdout = stdout[#stdout]
        tools.chat:add_tool_output(
          self,
          string.format(
            [[**VectorCode Vectorise Tool**:
  - New files added: %d
  - Existing files updated: %d
  - Orphaned files removed: %d
  - Up-to-date files skipped: %d
  - Failed to decode: %d
  ]],
            stdout.add or 0,
            stdout.update or 0,
            stdout.removed or 0,
            stdout.skipped or 0,
            stdout.failed or 0
          )
        )
        if cmd.project_root and cmd.project_root then
          tools.chat:add_tool_output(
            self,
            string.format("\nThe files were added to `%s`", cmd.project_root),
            ""
          )
        end
      end,
    },
  }
end
