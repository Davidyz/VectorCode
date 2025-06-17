---@module "codecompanion"

local cc_common = require("vectorcode.integrations.codecompanion.common")
local vc_config = require("vectorcode.config")
local check_cli_wrap = vc_config.check_cli_wrap
local logger = vc_config.logger

local job_runner = nil

---@alias QueryToolArgs { project_root:string, count: integer, query: string[] }

---@param opts VectorCode.CodeCompanion.QueryToolOpts?
---@return CodeCompanion.Agent.Tool
return check_cli_wrap(function(opts)
  opts = cc_common.get_query_tool_opts(opts)
  assert(
    type(opts.max_num) == "number" and type(opts.default_num) == "number",
    string.format("Options are not correctly formatted:%s", vim.inspect(opts))
  )
  ---@type "file"|"chunk"
  local mode
  if opts.chunk_mode then
    mode = "chunk"
  else
    mode = "file"
  end

  logger.info("Creating CodeCompanion tool with the following args:\n", opts)
  local capping_message = ""
  if opts.max_num > 0 then
    capping_message = ("  - Request for at most %d documents"):format(opts.max_num)
  end

  local tool_name = "vectorcode_query"
  return {
    name = tool_name,
    cmds = {
      ---@param agent CodeCompanion.Agent
      ---@param action QueryToolArgs
      ---@return nil|{ status: string, msg: string }
      function(agent, action, _, cb)
        logger.info(
          "CodeCompanion query tool called with the following arguments:\n",
          action
        )
        job_runner = cc_common.initialise_runner(opts.use_lsp)
        assert(job_runner ~= nil, "Jobrunner not initialised!")
        assert(
          type(cb) == "function",
          "Please upgrade CodeCompanion.nvim to at least 13.5.0"
        )

        if action.query == nil then
          return {
            status = "error",
            data = "Missing argument: option.query, please refine the tool argument.",
          }
        end

        local args = { "query" }
        vim.list_extend(args, action.query)
        vim.list_extend(args, { "--pipe", "-n", tostring(action.count) })
        if opts.chunk_mode then
          vim.list_extend(args, { "--include", "path", "chunk" })
        else
          vim.list_extend(args, { "--include", "path", "document" })
        end
        if action.project_root == "" then
          action.project_root = nil
        end
        if action.project_root ~= nil then
          action.project_root = vim.fs.normalize(action.project_root)
          if
            vim.uv.fs_stat(action.project_root) ~= nil
            and vim.uv.fs_stat(action.project_root).type == "directory"
          then
            vim.list_extend(args, { "--project_root", action.project_root })
          else
            return {
              status = "error",
              data = "INVALID PROJECT ROOT! USE THE LS COMMAND!",
            }
          end
        end

        if opts.no_duplicate and agent.chat.refs ~= nil then
          -- exclude files that has been added to the context
          local existing_files = { "--exclude" }
          for _, ref in pairs(agent.chat.refs) do
            if ref.source == cc_common.tool_result_source then
              table.insert(existing_files, ref.id)
            elseif type(ref.path) == "string" then
              table.insert(existing_files, ref.path)
            elseif ref.bufnr then
              local fname = vim.api.nvim_buf_get_name(ref.bufnr)
              if fname ~= nil then
                local stat = vim.uv.fs_stat(fname)
                if stat and stat.type == "file" then
                  table.insert(existing_files, fname)
                end
              end
            end
          end
          if #existing_files > 1 then
            vim.list_extend(args, existing_files)
          end
        end
        vim.list_extend(args, { "--absolute" })
        logger.info(
          "CodeCompanion query tool called the runner with the following args: ",
          args
        )

        job_runner.run_async(args, function(result, error)
          if vim.islist(result) and #result > 0 and result[1].path ~= nil then ---@cast result VectorCode.QueryResult[]
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
        description = "Retrieves code documents using semantic search or lists indexed projects",
        parameters = {
          type = "object",
          properties = {
            query = {
              type = "array",
              items = { type = "string" },
              description = "Query messages used for the search.",
            },
            count = {
              type = "integer",
              description = string.format(
                "Number of documents to retrieve, must be positive. Use %d by default",
                tonumber(opts.default_num)
              ),
            },
            project_root = {
              type = "string",
              description = "Project path to search within (must be from 'ls' results). Use empty string for the current project.",
            },
          },
          required = { "query", "count", "project_root" },
          additionalProperties = false,
        },
        strict = true,
      },
    },
    system_prompt = function()
      local guidelines = {
        "  - The path of a retrieved file will be wrapped in `<path>` and `</path>` tags. Its content will be right after the `</path>` tag, wrapped by `<content>` and `</content>` tags. Do not include the `<path>``</path>` tags in your answers when you mention the paths.",
        "  - The results may also be chunks of the source code. In this case, the text chunks will be wrapped in <chunk></chunk>. If the starting and ending line ranges are available, they will be wrapped in <start_line></start_line> and <end_line></end_line> tags. Make use of the line numbers (NOT THE XML TAGS) when you're quoting the source code.",
        "  - If you used the tool, tell users that they may need to wait for the results and there will be a virtual text indicator showing the tool is still running",
        "  - Include one single command call for VectorCode each time. You may include multiple keywords in the command",
        "  - VectorCode is the name of this tool. Do not include it in the query unless the user explicitly asks",
        "  - Use the `ls` command to retrieve a list of indexed project and pick one that may be relevant, unless the user explicitly mentioned 'this project' (or in other equivalent expressions)",
        "  - **The project root option MUST be a valid path on the filesystem. It can only be one of the results from the `ls` command or from user input**",
        capping_message,
        ("  - If the user did not specify how many documents to retrieve, **start with %d documents**"):format(
          opts.default_num
        ),
        "  - If you decide to call VectorCode tool, do not start answering the question until you have the results. Provide answers based on the results and let the user decide whether to run the tool again",
      }
      vim.list_extend(
        guidelines,
        vim.tbl_map(function(line)
          return "  - " .. line
        end, require("vectorcode").prompts({ "query", "ls" }))
      )
      local root = vim.fs.root(0, { ".vectorcode", ".git" })
      if root ~= nil then
        vim.list_extend(guidelines, {
          string.format(
            "  - The current working directory is %s. Assume the user query is about this project, unless the user asked otherwise or queries from the current project fails to return useful results.",
            root
          ),
        })
      end
      return string.format(
        [[### VectorCode, a repository indexing and query tool.

1. **Purpose**: This gives you the ability to access the repository to find information that you may need to assist the user.

2. **Key Points**:
%s 
]],
        table.concat(guidelines, "\n")
      )
    end,
    output = {
      ---@param agent CodeCompanion.Agent
      ---@param cmd QueryToolArgs
      ---@param stderr table|string
      error = function(self, agent, cmd, stderr)
        logger.error(
          ("CodeCompanion tool with command %s thrown with the following error: %s"):format(
            vim.inspect(cmd),
            vim.inspect(stderr)
          )
        )
        stderr = cc_common.flatten_table_to_string(stderr)
        agent.chat:add_tool_output(
          self,
          string.format("**VectorCode Tool**: Failed with error:\n```\n%s\n```", stderr)
        )
      end,
      ---@param agent CodeCompanion.Agent
      ---@param cmd QueryToolArgs
      ---@param stdout VectorCode.QueryResult[][]
      success = function(self, agent, cmd, stdout)
        stdout = stdout[1]
        logger.info(
          ("CodeCompanion tool with command %s finished."):format(vim.inspect(cmd))
        )
        local user_message
        local max_result = #stdout
        if opts.max_num > 0 then
          max_result = math.min(opts.max_num or 1, max_result)
        end
        for i, file in pairs(stdout) do
          if i <= max_result then
            if i == 1 then
              user_message = string.format(
                "**VectorCode Tool**: Retrieved %d %s(s)",
                max_result,
                mode
              )
              if cmd.project_root then
                user_message = user_message .. " from " .. cmd.project_root
              end
              user_message = user_message .. "\n"
            else
              user_message = ""
            end
            agent.chat:add_tool_output(
              self,
              cc_common.process_result(file),
              user_message
            )
            if not opts.chunk_mode then
              -- skip referencing because there will be multiple chunks with the same path (id).
              -- TODO: figure out a way to deduplicate.
              agent.chat.references:add({
                source = cc_common.tool_result_source,
                id = file.path,
                path = file.path,
                opts = { visible = false },
              })
            end
          end
        end
      end,
    },
  }
end)
