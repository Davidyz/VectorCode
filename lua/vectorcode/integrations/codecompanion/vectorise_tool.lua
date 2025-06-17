---@module "codecompanion"

local cc_common = require("vectorcode.integrations.codecompanion.common")
local vectorcode = require("vectorcode")

---@alias VectoriseToolArgs { paths: string[], project_root: string }

---@alias VectoriseResult { add: integer, update: integer, removed: integer }

---@param opts VectorCode.CodeCompanion.VectoriseToolOpts|{}|nil
---@return CodeCompanion.Agent.Tool
return function(opts)
  opts = cc_common.get_vectorise_tool_opts(opts)
  local tool_name = "vectorcode_vectorise"
  local job_runner = cc_common.initialise_runner(opts.use_lsp)

  ---@type CodeCompanion.Agent.Tool|{}
  return {
    name = tool_name,
    schema = {
      type = "function",
      ["function"] = {
        name = tool_name,
        description = string.format(
          "Vectorise files in a project so that they'll be available from the vectorcode_query tool\n%s",
          table.concat(vectorcode.prompts("vectorise"), "\n")
        ),
        parameters = {
          type = "object",
          properties = {
            paths = {
              type = "array",
              items = { type = "string" },
              description = "Paths to the files to be vectorised",
            },
            project_root = {
              type = "string",
              description = "The project that the files belong to. Either use a path from the `vectorcode_ls` tool, or leave empty to use the current git project.",
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
      ---@return nil|{ status: string, msg: string }
      function(agent, action, _, cb)
        local args = { "vectorise", "--pipe" }
        local project_root = action.project_root or ""
        if project_root ~= "" then
          local stat = vim.uv.fs_stat(project_root)
          if stat and stat.type == "directory" then
            vim.list_extend(args, { "--project_root", project_root })
          end
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
            vim.schedule_wrap(vim.notify)(vim.inspect(result))
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
  }
end
