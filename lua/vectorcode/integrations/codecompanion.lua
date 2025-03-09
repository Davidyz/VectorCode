---@alias tool_opts {max_num: integer?, default_num: integer?, include_stderr: boolean?}

local check_cli_wrap = require("vectorcode.config").check_cli_wrap

---@param opts tool_opts?
---@return CodeCompanion.Agent.Tool
local make_tool = check_cli_wrap(function(opts)
  opts = vim.tbl_deep_extend(
    "force",
    { max_num = -1, default_num = 10, include_stderr = false },
    opts or {}
  )
  local capping_message = ""
  if opts.max_num > 0 then
    capping_message = ("  - Request for at most %d documents"):format(opts.max_num)
  end
  return {
    name = "vectorcode",
    cmds = { { "vectorcode", "query", "--pipe" } },
    handlers = {
      ---@param self CodeCompanion.Agent
      setup = function(self)
        local tool = self.tool
        local n_query = tool.request.action.count or opts.default_num
        local keywords = tool.request.action.query
        if type(keywords) == "string" then
          keywords = { keywords }
        end
        vim.list_extend(tool.cmds[1], { "-n", tostring(n_query) })
        vim.list_extend(tool.cmds[1], keywords)
        if not opts.include_stderr then
          vim.list_extend(tool.cmds[1], { "--no_stderr" })
        end
      end,
    },
    schema = {
      {
        tool = {
          _attr = { name = "vectorcode" },
          action = {
            query = { "keyword1", "keyword2" },
            count = 5,
          },
        },
      },
      {
        tool = {
          _attr = { name = "vectorcode" },
          action = {
            query = { "keyword1" },
            count = 2,
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
  - seperate phrases into distinct keywords when appropriate
  - If a class, type or function has been imported from another file, this tool may be able to find its source. Add the name of the imported symbol to the query
  - The embeddings are mostly generated from source code, so using keywords that may be present in source code may help with the retrieval
  - The path of a retrieved file will be wrapped in `<path>` and `</path>` tags. Its content will be right after the `</path>` tag, wrapped by `<content>` and `</content>` tags
  - If you used the tool, tell users that they may need to wait for the results and there will be a virtual text indicator showing the tool is still running
  - Avoid retrieving one single file because the retrieval mechanism may not be very accurate
  - When providing answers based on VectorCode results, try to give references such as paths to files and line ranges, unless you're told otherwise
  - Include one single command call for VectorCode each time. You may include multiple keywords in the command
  - VectorCode is the name of this tool. Do not include it in the query unless the user explicitly asks
  - If the retrieval results do not contain the needed context, increase the number of files to retrieve so that the result will more likely contain the desired files
  - If the returned paths are relative, they are relative to the root of the project directory
  - Do not suggest edits to retrieved files that are outside of the current working directory, unless the user instructed otherwise
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

Remember:
- Minimize explanations unless prompted. Focus on generating correct XML.]],
        capping_message,
        ("  - If the user did not specify how many documents to retrieve, **start with %d documents**"):format(
          opts.default_num
        ),
        xml2lua.toXml({ tools = { schema[1] } }),
        xml2lua.toXml({ tools = { schema[2] } })
      )
    end,
    output = {
      ---@param agent CodeCompanion.Agent
      ---@param cmd table
      ---@param stderr table
      ---@param stdout? table
      error = function(agent, cmd, stderr, stdout)
        if type(stderr) == "table" then
          stderr = table.concat(vim.iter(stderr):flatten(math.huge):totable(), "\n")
        end

        if opts.include_stderr then
          vim.notify(
            stderr,
            vim.log.levels.ERROR,
            require("vectorcode.config").notify_opts
          )
          agent.chat:add_message({
            role = "user",
            content = string.format(
              [[After the VectorCode tool completed, there was an error:
<error>
%s
</error>
]],
              stderr
            ),
          }, { visible = false })

          agent.chat:add_buf_message({
            role = "user",
            content = "I've shared the error message from the VectorCode tool with you.\n",
          })
        else
          agent.chat:add_buf_message({
            role = "user",
            content = "There was an error in the execution of the tool, but the user chose to ignore it.",
          })
        end
      end,
      ---@param agent CodeCompanion.Agent
      ---@param cmd table The command that was executed
      ---@param stdout table
      success = function(agent, cmd, stdout)
        local retrievals = {}
        if type(stdout) == "table" then
          retrievals = vim.json.decode(
            vim.iter(stdout):flatten(math.huge):totable()[1],
            { array = true, object = true }
          )
        end

        for i, file in ipairs(retrievals) do
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

        agent.chat:add_buf_message({
          role = "user",
          content = "I've shared the content from the VectorCode tool with you.\n",
        })
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
          local ok, vc_config = pcall(require, "vectorcode.config")
          if ok then
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
          end
        end,
      }
    end),

    make_tool = make_tool,
  },
}
