---@module "codecompanion"

local cc_common = require("vectorcode.integrations.codecompanion.common")
local cc_config = require("codecompanion.config").config
local cc_schema = require("codecompanion.schema")
local http_client = require("codecompanion.http")
local vc_config = require("vectorcode.config")
local check_cli_wrap = vc_config.check_cli_wrap
local logger = vc_config.logger

local job_runner = nil

---@alias QueryToolArgs { project_root:string?, count: integer?, query: string[], allow_summary: boolean?, deduplicate: boolean? }

---@type VectorCode.CodeCompanion.QueryToolOpts
local default_query_options = {
  use_lsp = vc_config.get_user_config().async_backend == "lsp",
  requires_approval = false,
  include_in_toolbox = true,
  max_num = { chunk = -1, document = -1 },
  default_num = { chunk = 50, document = 10 },
  no_duplicate = true,
  chunk_mode = false,
  summarise = {
    enabled = false,
    query_augmented = true,
    system_prompt = [[You are an expert and experienced code analyzer and summarizer. Your primary task is to analyze provided source code, which will be given as a list of XML objects, and generate a comprehensive, well-structured Markdown summary. This summary will serve as a concise source of information for others to quickly understand how the code works and how to interact with it, without needing to delve into the full source code.

Input Format:
Each XML object represents either a full file or a chunk of a file, containing the following tags:
- `<path>...</path>`: The absolute file path of the source code.
- `<document>...</document>`: The full content of a source code file. This tag will not coexist with `<chunk>`.
- `<chunk>...</chunk>`: A segment of source code from a file. This tag will not coexist with `<document>`.
- `<start_line>...</start_line>` and `<end_line>...</end_line>`: These tags will be present only when a `<chunk>` tag is used, indicating the starting and ending line numbers of the chunk within its respective file.

Your goal is to process each of these XML objects. If multiple chunks belong to the same file, you must synthesize them to form a cohesive understanding of that file. Generate a single Markdown summary that combines insights from all provided objects.

Markdown Structure:

    Top-Level Header (#): The absolute or relative file path of the source code.

    Secondary Headers (##): For each top-level symbol (e.g., functions, classes, global variables) defined directly within the source code file that are importable or includable by other programs.

    Tertiary Headers (###): For symbols nested one level deep within a secondary header's symbol (e.g., methods within a class, inner functions).

    Quaternary Headers (####): For symbols nested two levels deep (e.g., a function defined within a method of a class).

    Continue this pattern, incrementing the header level for each deeper level of nesting.

Content for Each Section:

    Descriptive Summary: Each header section (from secondary headers downwards) must contain a concise and informative summary of the symbol defined by that header.

        For Functions/Methods: Explain their purpose, parameters (including types), return values (including types), high-level implementation details, and any significant side effects or core logic. For example, if summarizing a sorting function, include the sorting algorithm used. If summarizing a function that makes an HTTP request, mention the network library employed.

        For Classes: Describe the class's role, its main responsibilities, and key characteristics.

        For Variables (global or within scope): State their purpose, type (if discernible), and initial value or common usage.

        For Modules/Files (under the top-level header): Provide an overall description of the file's purpose, its main components, and its role within the larger project (if context is available).

General Guidelines:

    Clarity and Conciseness: Summaries should be easy to understand, avoiding jargon where possible, and as brief as possible while retaining essential information. The full summary MUST NOT be longer than the original code input. When quoting a symbol in the code, include the line numbers where possible.

    Accuracy: Ensure the summary accurately reflects the code's functionality.

    Focus on Public Interface/Behavior: Prioritize describing what a function/class does and how it's used. Only include details about symbols (variables, functions, classes) that are importable/includable by other programs. DO NOT include local variables and functions that are not accessible by other functions outside their immediate scope.

    No Code Snippets: Do not include any actual code snippets in the summary. Focus solely on descriptive text. If you need to refer to a specific element for context (e.g., in an error description), describe it and provide line numbers for reference from the source code.

    Syntax/Semantic Errors: If the code contains syntax or semantic errors, describe them clearly within the summary, indicating the nature of the error.

    Language Agnostic: Adapt the summary to the specific programming language of the provided source code (e.g., Python, JavaScript, Java, C++, etc.).

    Handle Edge Cases/Dependencies: If a symbol relies heavily on external dependencies or handles specific edge cases, briefly mention these if they are significant to its overall function.

    Information Source: There will be no extra information available to you. Provide the summary solely based on the provided XML objects.

    Omit meaningless results: For an xml object that contains no meaningful information, you're free to omit it, but please leave a sentence in the summary saying that you did this.

    No extra reply: Your reply should solely consist of the summary. Do not say anything else.

    Merge chunks from the same file: When there are chunks that belong to the same file, merge their content so that they're grouped under the same top level header.
]],
  },
}

---Returns the adapter if it works with summarysation.
---@return CodeCompanion.HTTPAdapter|nil
local function check_adapter_for_summarisation(adapter)
  local resolved_adapter = require("codecompanion.adapters").resolve(adapter)
  if resolved_adapter.type == nil or resolved_adapter.type == "http" then
    return vim.deepcopy(resolved_adapter) --[[@as CodeCompanion.HTTPAdapter]]
  else
    vim.schedule_wrap(vim.notify)(
      [[VectorCode query summarisation doesn't work with ACP adapters yet.
Summarysation will be disabled.
Please configure an HTTP adapter for it.]],
      vim.log.levels.WARN,
      vc_config.notify_opts
    )
  end
end

---@param opts VectorCode.CodeCompanion.QueryToolOpts|{}|nil
---@return VectorCode.CodeCompanion.QueryToolOpts
local get_query_tool_opts = function(opts)
  if opts == nil or opts.use_lsp == nil then
    opts = vim.tbl_deep_extend(
      "force",
      opts or {},
      { use_lsp = vc_config.get_user_config().async_backend == "lsp" }
    )
  end
  opts = vim.tbl_deep_extend("force", default_query_options, opts)
  if type(opts.default_num) == "table" then
    if opts.chunk_mode then
      opts.default_num = opts.default_num.chunk
    else
      opts.default_num = opts.default_num.document
    end
    assert(
      type(opts.default_num) == "number",
      "default_num should be an integer or a table: {chunk: integer, document: integer}"
    )
  end
  if type(opts.max_num) == "table" then
    if opts.chunk_mode then
      opts.max_num = opts.max_num.chunk
    else
      opts.max_num = opts.max_num.document
    end
    assert(
      type(opts.max_num) == "number",
      "max_num should be an integer or a table: {chunk: integer, document: integer}"
    )
  end
  logger.info(
    string.format(
      "Loading `vectorcode_query` with the following opts:\n%s",
      vim.inspect(opts)
    )
  )
  return opts
end

---@param result VectorCode.QueryResult
---@return string
local process_result = function(result)
  local llm_message
  if result.chunk then
    -- chunk mode
    llm_message =
      string.format("<path>%s</path><chunk>%s</chunk>", result.path, result.chunk)
    if result.start_line and result.end_line then
      llm_message = llm_message
        .. string.format(
          "<start_line>%d</start_line><end_line>%d</end_line>",
          result.start_line,
          result.end_line
        )
    end
  else
    -- full document mode
    llm_message = string.format(
      "<path>%s</path><content>%s</content>",
      result.path,
      result.document
    )
  end
  return llm_message
end

---@alias chat_id integer
---@alias results table<string,boolean>
---@type table<chat_id, results>
local result_tracker = {}

---@param results VectorCode.QueryResult[]
---@param chat CodeCompanion.Chat
---@return VectorCode.QueryResult[]
local filter_results = function(results, chat)
  local existing_refs = chat.context_items or {}

  existing_refs = vim
    .iter(existing_refs)
    :filter(
      ---@param ref CodeCompanion.Chat.ContextItem
      function(ref)
        return ref.source == cc_common.tool_result_source or ref.path or ref.bufnr
      end
    )
    :map(
      ---@param ref CodeCompanion.Chat.ContextItem
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

---@param adapter CodeCompanion.HTTPAdapter
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
      return process_result(res)
    end)
    :totable())

  if
    summarise_opts.enabled
    and cmd.allow_summary ~= false
    and type(callback) == "function"
    and #result > 0
  then
    ---@type CodeCompanion.HTTPAdapter?
    local adapter = check_adapter_for_summarisation(summarise_opts.adapter)
    if adapter == nil then
      summarise_opts.enabled = false
      return callback(result_xml)
    end

    local system_prompt = summarise_opts.system_prompt
    if type(system_prompt) == "function" then
      system_prompt =
        system_prompt(get_query_tool_opts().summarise.system_prompt --[[@as string]])
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

    ---@type CodeCompanion.HTTPClient
    local client = http_client.new({ adapter = settings })
    client:request(payload, {
      ---@param _adapter CodeCompanion.HTTPAdapter
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

---@param results VectorCode.QueryResult[]
---@return VectorCode.QueryResult[]
local function cleanup_paths(results)
  local cwd = vim.fs.root(0, { ".vectorcode", ".git" }) or vim.uv.cwd()
  if cwd then
    results = vim
      .iter(results)
      :map(
        ---@param res VectorCode.QueryResult
        function(res)
          res.path = cc_common.cleanup_path(res.path)
          return res
        end
      )
      :totable()
  end
  return results
end

---@param opts VectorCode.CodeCompanion.QueryToolOpts?
---@return CodeCompanion.Tools.Tool
return check_cli_wrap(function(opts)
  opts = get_query_tool_opts(opts)
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
      ---@param tools CodeCompanion.Tools
      ---@param action QueryToolArgs
      ---@return nil|{ status: string, data: string }
      function(tools, action, _, cb)
        logger.info(
          "CodeCompanion query tool called with the following arguments:\n",
          action
        )

        if action.deduplicate == nil then
          action.deduplicate = opts.no_duplicate
        end

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
        vim.list_extend(
          args,
          { "--pipe", "-n", tostring(action.count or opts.default_num) }
        )
        if opts.chunk_mode then
          vim.list_extend(args, { "--include", "path", "chunk" })
        else
          vim.list_extend(args, { "--include", "path", "document" })
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

        if
          opts.no_duplicate
          and tools.chat.context_items ~= nil
          and action.deduplicate
        then
          -- exclude files that has been added to the context
          local existing_files = { "--exclude" }
          for _, ref in pairs(tools.chat.context_items) do
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
              summary_opts.enabled = summary_opts.enabled(tools.chat, result) --[[@as  boolean]]
            end

            if
              opts.no_duplicate
              and not summary_opts.enabled
              and action.deduplicate
            then
              -- NOTE: deduplication in summary mode prevents the model from requesting
              -- the same content without summarysation.
              result = filter_results(result, tools.chat)
            end

            result = cleanup_paths(result)

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
        end, tools.chat.bufnr)
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
**ABSOLUTE PATHS** in the results indicate that the files are OUTSIDE of the current working directories and you can **ONLY** access them via the VectorCode tools.
**RELATIVE PATHS** in the results indicate that the files are INSIDE the current project. You can use VectorCode tools or any other tools that the user provided to interact with them. They are relative to the project root.
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
                "Number of documents or chunks to retrieve, must be positive. The default value of this parameter is %d. Do not query for more than %d results.",
                tonumber(opts.default_num),
                tonumber(opts.max_num)
              ),
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
            allow_summary = {
              type = "boolean",
              description = [[
When this option is set to `true`, you're allowing the retrieval results to be summarised.
Leave this to `true` by default.
Set this to `false` only if you've been instructed by the user to not enable summarisation, or if the summary is missing information that you'd need for the current task.
]],
            },
            deduplicate = {
              type = "boolean",
              description = [[
Set this to `false` to deduplicate the search results with references in the chat context. 
Default to `true`.
DO NOT MODIFY UNLESS INSTRUCTED BY THE USER, OR A PREVIOUS QUERY RETURNED NO RESULTS.
]],
            },
          },
          required = {
            "query",
          },
        },
      },
    },
    output = {
      ---@param tools CodeCompanion.Tools
      ---@param cmd QueryToolArgs
      ---@param stderr table|string
      error = function(self, tools, cmd, stderr)
        logger.error(
          ("CodeCompanion tool with command %s thrown with the following error: %s"):format(
            vim.inspect(cmd),
            vim.inspect(stderr)
          )
        )
        stderr = cc_common.flatten_table_to_string(stderr)
        if string.find(stderr, "InvalidCollectionException") then
          if cmd.project_root then
            tools.chat:add_tool_output(
              self,
              string.format(
                "`%s` hasn't been vectorised. Please use the `vectorcode_vectorise` tool or vectorise it from the CLI.",
                cmd.project_root
              )
            )
          else
            tools.chat:add_tool_output(
              self,
              "Failed to query from the requested project. Please verify the available projects via the `vectorcode_ls` tool or run it from the CLI."
            )
          end
        else
          tools.chat:add_tool_output(
            self,
            string.format(
              "**VectorCode `query` Tool**: Failed with error:\n```\n%s\n```",
              stderr
            )
          )
        end
      end,
      ---@param tools CodeCompanion.Tools
      ---@param cmd QueryToolArgs
      ---@param stdout VectorCode.CodeCompanion.QueryToolResult[]
      success = function(self, tools, cmd, stdout)
        stdout = stdout[#stdout]
        logger.info(
          ("CodeCompanion tool with command %s finished."):format(vim.inspect(cmd))
        )
        if vim.tbl_isempty(stdout.raw_results) then
          logger.info("CodeCompanion query tool recieved empty result.")
          return tools.chat:add_tool_output(
            self,
            string.format(
              "`%s` tool returned empty result. Please retry without deduplication.",
              tool_name
            ),
            "**VectorCode `query` Tool**: Retrieved 0 result. Retrying..."
          )
        end
        tools.chat:add_tool_output(
          self,
          stdout.summary
            or table.concat(vim
              .iter(stdout.raw_results or {})
              :map(function(res)
                return process_result(res)
              end)
              :totable()),
          string.format(
            "**VectorCode `query` Tool**: Retrieved %d %s(s)",
            stdout.count,
            mode
          )
        )
        if not opts.chunk_mode then
          for _, result in pairs(stdout.raw_results) do
            -- skip referencing because there will be multiple chunks with the same path (id).
            tools.chat.context:add({
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
