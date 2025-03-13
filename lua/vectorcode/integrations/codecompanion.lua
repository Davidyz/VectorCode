---@module "codecompanion"

---@alias tool_opts {max_num: integer?, default_num: integer?, include_stderr: boolean?, use_lsp: boolean}

local vc_config = require("vectorcode.config")
local check_cli_wrap = vc_config.check_cli_wrap
local notify_opts = vc_config.notify_opts

local job_runner = nil
---@param use_lsp boolean
local function initialise_runner(use_lsp)
  if job_runner == nil then
    if use_lsp then
      job_runner = require("vectorcode.jobrunner.lsp")
    end
    if job_runner == nil then
      job_runner = require("vectorcode.jobrunner.cmd")
      if use_lsp then
        vim.schedule_wrap(vim.notify)(
          "Failed to initialise the LSP runner. Falling back to cmd runner.",
          vim.log.levels.WARN,
          notify_opts
        )
      end
    end
  end
end
---@param opts tool_opts?
---@return CodeCompanion.Agent.Tool
local make_tool = check_cli_wrap(function(opts)
  if opts == nil or opts.use_lsp == nil then
    opts = vim.tbl_deep_extend(
      "force",
      opts or {},
      { use_lsp = vc_config.get_user_config().async_backend == "lsp" }
    )
  end
  opts = vim.tbl_deep_extend(
    "force",
    { max_num = -1, default_num = 10, include_stderr = false, use_lsp = false },
    opts or {}
  )
  local capping_message = ""
  if opts.max_num > 0 then
    capping_message = ("  - Request for at most %d documents"):format(opts.max_num)
  end

  return {
    name = "vectorcode",
    cmds = {
      ---@param agent CodeCompanion.Agent
      ---@param action table
      ---@param input table
      ---@return nil|{ status: string, msg: string }
      function(agent, action, input, cb)
        initialise_runner(opts.use_lsp)
        assert(job_runner ~= nil)
        assert(
          type(cb) == "function",
          "Please upgrade CodeCompanion.nvim to at least 13.5.0"
        )
        assert(vim.list_contains({ "ls", "query" }, action.command))
        if action.command == "query" then
          local args = { "query", "--pipe", "-n", tostring(action.options.count) }
          if type(action.options.query) == "string" then
            action.options.query = { action.options.query }
          end
          vim.list_extend(args, action.options.query)
          if
            action.options.project_root ~= nil
            and vim.uv.fs_stat(action.options.project_root).type == "directory"
          then
            vim.list_extend(args, { "--project_root", action.options.project_root })
          end
          job_runner.run_async(args, function(result, error)
            if vim.islist(result) and #result > 0 and result[1].path ~= nil then ---@cast result VectorCode.Result[]
              cb({ status = "success", data = result })
            else
              cb({
                status = "error",
                data = table.concat(vim.iter(error):flatten(math.huge):totable(), "\n"),
              })
            end
          end, agent.chat.bufnr)
        elseif action.command == "ls" then
          job_runner.run_async({ "ls", "--pipe" }, function(result, error)
            if vim.islist(result) and #result > 0 then
              cb({ status = "success", data = result })
            else
              cb({
                status = "error",
                data = table.concat(vim.iter(error):flatten(math.huge):totable(), "\n"),
              })
            end
          end, agent.chat.bufnr)
        end
      end,
    },
    schema = {
      {
        tool = {
          _attr = { name = "vectorcode" },
          action = {
            command = "query",
            options = {
              query = { "keyword1", "keyword2" },
              count = 5,
            },
          },
        },
      },
      {
        tool = {
          _attr = { name = "vectorcode" },
          action = {
            command = "query",
            options = {
              query = { "keyword1" },
              count = 2,
            },
          },
        },
      },
      {
        tool = {
          _attr = { name = "vectorcode" },
          action = {
            command = "query",
            options = {
              query = { "keyword1" },
              count = 3,
              project_root = "path/to/other/project",
            },
          },
        },
      },
      {
        tool = {
          _attr = { name = "vectorcode" },
          action = {
            command = "ls",
          },
        },
      },
    },
    system_prompt = function(schema, xml2lua)
      return string.format(
        [[### VectorCode, a repository indexing and query tool.

1. **Purpose**: This gives you the ability to access the repository to find information that you may need to assist the user.

2. **Usage**: Return an XML markdown code block that retrieves relevant documents corresponding to the generated query.

3. **Key Points**:
  - **Use at your discretion** when you feel you don't have enough information about the repository or project
  - Ensure XML is **valid and follows the schema**
  - **Don't escape** special characters
  - Make sure the tools xml block is **surrounded by ```xml**
  - separate phrases into distinct keywords when appropriate
  - If a class, type or function has been imported from another file, this tool may be able to find its source. Add the name of the imported symbol to the query
  - The embeddings are mostly generated from source code, so using keywords that may be present in source code may help with the retrieval
  - The path of a retrieved file will be wrapped in `<path>` and `</path>` tags. Its content will be right after the `</path>` tag, wrapped by `<content>` and `</content>` tags
  - If you used the tool, tell users that they may need to wait for the results and there will be a virtual text indicator showing the tool is still running
  - Avoid retrieving one single file because the retrieval mechanism may not be very accurate
  - When providing answers based on VectorCode results, try to give references such as paths to files and line ranges, unless you're told otherwise
  - Include one single command call for VectorCode each time. You may include multiple keywords in the command
  - VectorCode is the name of this tool. Do not include it in the query unless the user explicitly asks
  - If the retrieval results do not contain the needed context, increase the file count so that the result will more likely contain the desired files
  - If the returned paths are relative, they are relative to the root of the project directory
  - Do not suggest edits to retrieved files that are outside of the current working directory, unless the user instructed otherwise
  - If a query failed to retrieve desired results, a new attempt should use different keywords that are orthogonal to the previous ones but with similar meanings
  - When asked about information in other project, use the `ls` command to see if there's any other indexed project that might help
  - DO NOT MAKE UP A PATH. ONLY USE PROJECT ROOTS RETURNED BY THE LS COMMAND OR PROVIDED BY THE USER
  %s
  %s

4. **Actions**:

a) **Query for 5 documents using 2 keywords: `keyword1` and `keyword2`**:

```xml
%s
```

b) **Query for 2 documents using one keyword: `keyword1`**:

```xml
%s
```
c) **Query for 3 documents using one keyword: `keyword1` in a different project located at `path/to/other/project` (relative to current working directory)**:
```xml
%s
```
d) **Get all indexed project**
```xml
%s
```

Remember:
- Minimize explanations unless prompted. Focus on generating correct XML.]],
        capping_message,
        ("  - If the user did not specify how many documents to retrieve, **start with %d documents**"):format(
          opts.default_num
        ),
        xml2lua.toXml({ tools = { schema[1] } }),
        xml2lua.toXml({ tools = { schema[2] } }),
        xml2lua.toXml({ tools = { schema[3] } }),
        xml2lua.toXml({ tools = { schema[4] } })
      )
    end,
    output = {
      success = function(agent, cmd, stdout)
        stdout = stdout[1]
        if cmd.command == "query" then
          for i, file in pairs(stdout) do
            if opts.max_num < 0 or i <= opts.max_num then
              agent.chat:add_message({
                role = "user",
                content = string.format(
                  [[Here is a file the VectorCode tool retrieved:
<path>
%s
</path>
<content>
%s
</content>
]],
                  file.path,
                  file.document
                ),
              }, { visible = false })
            end
          end
        elseif cmd.command == "ls" then
          for _, path in pairs(stdout) do
            agent.chat:add_message({
              role = "user",
              content = string.format("<collection>%s</collection>", path),
            }, { visible = false })
          end
        end
      end,
    },
  }
end)

return {
  chat = {
    ---@param component_cb (fun(result:VectorCode.Result):string)?
    make_slash_command = check_cli_wrap(function(component_cb)
      return {
        description = "Add relevant files from the codebase.",
        ---@param chat CodeCompanion.Chat
        callback = function(chat)
          local codebase_prompt = ""
          local vc_cache = vc_config.get_cacher_backend()
          local bufnr = chat.context.bufnr
          if not vc_cache.buf_is_registered(bufnr) then
            return
          end
          codebase_prompt =
            "The following are relevant files from the repository. Use them as extra context."
          local query_result = vc_cache.make_prompt_component(bufnr, component_cb)
          local id = tostring(query_result.count) .. " file(s) from codebase"
          codebase_prompt = codebase_prompt .. query_result.content
          chat:add_message(
            { content = codebase_prompt, role = "user" },
            { visible = false, id = id }
          )
          chat.references:add({
            source = "VectorCode",
            name = "VectorCode",
            id = id,
          })
        end,
      }
    end),

    make_tool = make_tool,
  },
}
