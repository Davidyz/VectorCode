---@module "codecompanion"

local vc_config = require("vectorcode.config")
local check_cli_wrap = vc_config.check_cli_wrap

return {
  chat = {
    ---@param component_cb (fun(result:VectorCode.QueryResult):string)?
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

    ---@param subcommand sub_cmd
    ---@param opts VectorCode.CodeCompanion.ToolOpts
    ---@return CodeCompanion.Agent.Tool
    make_tool = function(subcommand, opts)
      local has = require("codecompanion").has
      if has ~= nil and has("function-calling") then
        return require(
          string.format("vectorcode.integrations.codecompanion.%s_tool", subcommand)
        )(opts)
      else
        error("Unsupported version of codecompanion!")
      end
    end,
  },
}
