---@module "codecompanion"

local cc_common = require("vectorcode.integrations.codecompanion.common")
local cc_config = require("codecompanion.config").config
local cc_schema = require("codecompanion.schema")
local http_client = require("codecompanion.http")
local vc_config = require("vectorcode.config")
local check_cli_wrap = vc_config.check_cli_wrap
local logger = vc_config.logger

local job_runner = nil

---@alias QueryToolArgs { project_root:string, count: integer, query: string[], allow_summary: boolean }

---@alias chat_id integer
---@alias result_id string
---@type <chat_id: result_id>
local result_tracker = {}

---@param results VectorCode.QueryResult[]
---@param chat CodeCompanion.Chat
---@return VectorCode.QueryResult[]
local filter_results = function(results, chat)
  local existing_refs = chat.refs or {}

  existing_refs = vim
    .iter(existing_refs)
    :filter(
      ---@param ref CodeCompanion.Chat.Ref
      function(ref)
        return ref.source == cc_common.tool_result_source or ref.path or ref.bufnr
      end
    )
    :map(
      ---@param ref CodeCompanion.Chat.Ref
      function(ref)
        if ref.source == cc_common.tool_result_source then
          return ref.id
        elseif ref.path then
          return ref.path
        elseif ref.bufnr then
          return vim.api.nvim_buf_get_name(ref.bufnr)
        end
      end
    )
    :totable()

  ---@type VectorCode.QueryResult[]
  local filtered_results = vim
    .iter(results)
    :filter(
      ---@param res VectorCode.QueryResult
      function(res)
        -- return true if res should be kept
        if res.chunk then
          if res.chunk_id == nil then
            -- no chunk_id, always include
            return true
          end
          if
            result_tracker[chat.id] ~= nil and result_tracker[chat.id][res.chunk_id]
          then
            return false
          end
          return not vim.tbl_contains(existing_refs, res.chunk_id)
        else
          if result_tracker[chat.id] ~= nil and result_tracker[chat.id][res.path] then
            return false
          end
          return not vim.tbl_contains(existing_refs, res.path)
        end
      end
    )
    :totable()

  for _, res in pairs(filtered_results) do
    if result_tracker[chat.id] == nil then
      result_tracker[chat.id] = {}
    end
    result_tracker[chat.id][res.chunk_id or res.path] = true
  end

  return filtered_results
end

---@alias ChatMessage {role: string, content:string}

---@param adapter CodeCompanion.Adapter
---@param system_prompt string
---@param user_messages string|string[]
---@return {messages: ChatMessage[], tools:table?}
local function make_oneshot_payload(adapter, system_prompt, user_messages)
  if type(user_messages) == "string" then
    user_messages = { user_messages }
  end
  local messages =
    { { role = cc_config.constants.SYSTEM_ROLE, content = system_prompt } }
  for _, m in pairs(user_messages) do
    table.insert(messages, { role = cc_config.constants.USER_ROLE, content = m })
  end
  return { messages = adapter:map_roles(messages) }
end

---@param result VectorCode.QueryResult[]
---@param cmd QueryToolArgs
---@param summarise_opts VectorCode.CodeCompanion.SummariseOpts
---@param callback fun(summary:string)
local function generate_summary(result, summarise_opts, cmd, callback)
  assert(vim.islist(result), "result should be a list of VectorCode.QueryResult")
  local result_xml = table.concat(vim
    .iter(result)
    :map(function(res)
      return cc_common.process_result(res)
    end)
    :totable())

  if summarise_opts.enabled and cmd.allow_summary and type(callback) == "function" then
    ---@type CodeCompanion.Adapter
    local adapter =
      vim.deepcopy(require("codecompanion.adapters").resolve(summarise_opts.adapter))

    local system_prompt = summarise_opts.system_prompt
    if type(system_prompt) == "function" then
      system_prompt = system_prompt(
        cc_common.get_query_tool_opts().summarise.system_prompt --[[@as string]]
      )
    end

    assert(
      type(system_prompt) == "string",
      "`system_prompt` should have been converted to a string."
    )
    if summarise_opts.query_augmented then
      system_prompt = string.format(
        [[%s
        
The code provided to you is the result of a search in a codebase from the following query: %s.
When summarising the code, pay extra attention on information related to the queries.
      ]],
        system_prompt,
        table.concat(cmd.query, ", ")
      )
    end
    local payload = make_oneshot_payload(adapter, system_prompt, result_xml)
    local settings =
      vim.deepcopy(adapter:map_schema_to_params(cc_schema.get_default(adapter)))
    settings.opts.stream = false

    ---@type CodeCompanion.Client
    local client = http_client.new({ adapter = settings })
    client:request(payload, {
      ---@param _adapter CodeCompanion.Adapter
      callback = function(_, data, _adapter)
        if data then
          local res = _adapter.handlers.chat_output(_adapter, data)
          if res and res.status == "success" then
            local gen_summary = vim.trim(res.output.content or "")
            if gen_summary ~= "" then
              return callback(gen_summary)
            end
          end
        end
        return callback(result_xml)
      end,
    }, { silent = true })
  else
    callback(result_xml)
  end
end

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

  local tool_name = "vectorcode_query"
  return {
    name = tool_name,
    cmds = {
      ---@param agent CodeCompanion.Agent
      ---@param action QueryToolArgs
      ---@return nil|{ status: string, data: string }
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
            action.project_root = vim.fs.abspath(vim.fs.normalize(action.project_root))
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
            local summary_opts = vim.deepcopy(opts.summarise) or {}
            if type(summary_opts.enabled) == "function" then
              summary_opts.enabled = summary_opts.enabled(agent.chat, result) --[[@as  boolean]]
            end

            if opts.no_duplicate and not summary_opts.enabled then
              -- NOTE: deduplication in summary mode prevents the model from requesting
              -- the same content without summarysation.
              result = filter_results(result, agent.chat)
            end

            local max_result = #result
            if opts.max_num > 0 then
              max_result = math.min(tonumber(opts.max_num) or 1, max_result)
            end
            while #result > max_result do
              table.remove(result)
            end
            generate_summary(result, summary_opts, action, function(s)
              cb({
                status = "success",
                ---@type VectorCode.CodeCompanion.QueryToolResult
                data = { raw_results = result, count = #result, summary = s },
              })
            end)
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
        description = [[Retrieves code documents using semantic search.
The path of a retrieved file will be wrapped in `<path>` and `</path>` tags.
Its content will be right after the `</path>` tag, wrapped by `<content>` and `</content>` tags.
Do not include the xml tags in your answers when you mention the paths.
The results may also be chunks of the source code.
In this case, the text chunks will be wrapped in <chunk></chunk>.
If the starting and ending line ranges are available, they will be wrapped in <start_line></start_line> and <end_line></end_line> tags.
Make use of the line numbers (NOT THE XML TAGS) when you're quoting the source code.
Include one single command call for VectorCode each time.
You may include multiple keywords in the command.
**The project root option MUST be a valid path on the filesystem. It can only be one of the results from the `vectorcode_ls` tool or from user input**
        ]],
        parameters = {
          type = "object",
          properties = {
            query = {
              type = "array",
              items = { type = "string" },
              description = [[
Query messages used for the search. They should also contain relevant keywords.
For example, you should include `parameter`, `arguments` and `return value` for the query `function`.
If a query returned empty or repeated results, you should avoid using these query keywords, unless the user instructed otherwise.
              ]],
            },
            count = {
              type = "integer",
              description = string.format(
                "Number of documents or chunks to retrieve, must be positive. Use %d by default. Do not query for more than %d.",
                tonumber(opts.default_num),
                tonumber(opts.max_num)
              ),
            },
            project_root = {
              type = "string",
              description = "Project path to search within (must be from 'ls' results or user instructions). Use empty string for the current project. If the user did not specify a project, assume that they're referring to the current project and use an empty string for this parameter. If this fails, use the `vectorcode_ls` tool and ask the user to clarify the project.",
            },
            allow_summary = {
              type = "boolean",
              description = [[
When this option is set to `true`, you're allowing the retrieval results to be summarised.
Leave this to `true` by default.
Set this to `false` only if you've been instructed by the user to not enable summarisation, or if the summary is missing information that you'd need for the current task.
]],
            },
          },
          required = { "query", "count", "project_root", "allow_summary" },
          additionalProperties = false,
        },
        strict = true,
      },
    },
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
        if string.find(stderr, "InvalidCollectionException") then
          if cmd.project_root then
            agent.chat:add_tool_output(
              self,
              string.format(
                "`%s` hasn't been vectorised. Please use the `vectorcode_vectorise` tool or vectorise it from the CLI.",
                cmd.project_root
              )
            )
          else
            agent.chat:add_tool_output(
              self,
              "Failed to query from the requested project. Please verify the available projects via the `vectorcode_ls` tool or run it from the CLI."
            )
          end
        else
          agent.chat:add_tool_output(
            self,
            string.format(
              "**VectorCode `query` Tool**: Failed with error:\n```\n%s\n```",
              stderr
            )
          )
        end
      end,
      ---@param agent CodeCompanion.Agent
      ---@param cmd QueryToolArgs
      ---@param stdout VectorCode.CodeCompanion.QueryToolResult[]
      success = function(self, agent, cmd, stdout)
        stdout = stdout[1]
        logger.info(
          ("CodeCompanion tool with command %s finished."):format(vim.inspect(cmd))
        )
        agent.chat:add_tool_output(
          self,
          stdout.summary
            or table.concat(vim
              .iter(stdout.raw_results or {})
              :map(function(res)
                return cc_common.process_result(res)
              end)
              :totable()),
          string.format("**VectorCode Tool**: Retrieved %d %s(s)", stdout.count, mode)
        )
        if not opts.chunk_mode then
          for _, result in pairs(stdout.raw_results) do
            -- skip referencing because there will be multiple chunks with the same path (id).
            agent.chat.references:add({
              source = cc_common.tool_result_source,
              id = result.path,
              path = result.path,
              opts = { visible = false },
            })
          end
        end
      end,
    },
  }
end)
