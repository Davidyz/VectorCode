local check_cli_wrap = require("vectorcode.config").check_cli_wrap

---@class VectorCode.CopilotChatOpts

---@param opts? {prompt_header:string?, prompt_footer:string?, skip_empty:boolean?, format_file:fun(file:VectorCode.Result):string?}
---@return function
local make_context_provider = check_cli_wrap(function(opts)
  opts = vim.tbl_deep_extend("force", {
    prompt_header = "The following are relevant files from the repository. Use them as extra context for helping with code completion and understanding:",
    prompt_footer = "\nExplain and provide a strategy with examples about: \n",
    skip_empty = true,
    format_file = function(file)
      local utils = require("CopilotChat.utils")
      return string.format(
        [[
### File: %s
```%s
%s
```

---
]],
        file.path,
        utils.filetype(file.path),
        file.document
      )
    end,
  }, opts or {})

  return function()
    ---@return {content:string, filename:string, filetype:string}[]|{} Context data for CopilotChat or empty table
    local log = require("plenary.log")
    local copilot_utils = require("CopilotChat.utils")
    local vectorcode_cacher = require("vectorcode.config").get_cacher_backend()

    -- Initialize cache in the closure
    if not rawget(_G, "__vectorcode_copilotchat_cache") then
      _G.__vectorcode_copilotchat_cache = {
        content = "",
        checksum = "",
      }
    end
    local cache = _G.__vectorcode_copilotchat_cache

    -- Get all valid listed buffers
    local listed_buffers = vim.tbl_filter(function(b)
      return copilot_utils.buf_valid(b)
        and vim.fn.buflisted(b) == 1
        and #vim.fn.win_findbuf(b) > 0
    end, vim.api.nvim_list_bufs())

    -- Calculate a simple checksum based on buffer paths and modification times
    local buffers_checksum = ""
    for _, bufnr in ipairs(listed_buffers) do
      if vectorcode_cacher.buf_is_registered(bufnr) then
        local buf_path = vim.api.nvim_buf_get_name(bufnr)
        local modified = vim.fn.getbufvar(bufnr, "&modified")
        buffers_checksum = buffers_checksum .. buf_path .. tostring(modified)
      end
    end

    -- Use cached result if checksum matches
    if buffers_checksum == cache.checksum and cache.content ~= "" then
      return {
        {
          content = cache.content,
          filename = "vectorcode_context",
          filetype = "markdown",
        },
      }
    end

    local all_content = ""
    local total_files = 0
    local processed_paths = {}

    -- Process each buffer with registered VectorCode cache
    for _, bufnr in ipairs(listed_buffers) do
      local buf_path = vim.api.nvim_buf_get_name(bufnr)
      log.debug("Current buffer name", buf_path)

      -- Skip if already processed paths to avoid duplicates
      if
        not processed_paths[buf_path] and vectorcode_cacher.buf_is_registered(bufnr)
      then
        processed_paths[buf_path] = true
        log.debug("Current registered buffer name", buf_path)

        local cache_result =
          vectorcode_cacher.make_prompt_component(bufnr, opts.format_file)

        log.debug("VectorCode context", cache_result)
        if cache_result and cache_result.content and cache_result.content ~= "" then
          all_content = all_content .. "\n" .. cache_result.content
          total_files = total_files + cache_result.count
        end
      end
    end

    if total_files > 0 or not opts.skip_empty then
      local prompt_message = opts.prompt_header .. all_content .. opts.prompt_footer
      log.debug("VectorCode context result", prompt_message)
      return {
        {
          content = prompt_message,
          filename = "vectorcode_context",
          filetype = "markdown",
        },
      }
    end

    log.debug("VectorCode context when no success", opts.prompt_footer)
    -- If VectorCode is not available or has no results
    if not opts.skip_empty then
      return {
        {
          content = opts.prompt_footer,
          filename = "error",
          filetype = "markdown",
        },
      }
    end

    return {}
  end
end)

-- Update the integrations/init.lua file to include copilotchat
return {
  ---Creates a context provider for CopilotChat
  ---@param opts? {prompt_header:string?, prompt_footer:string?, skip_empty:boolean?, format_file:fun(file:VectorCode.Result):string?}
  ---@return function Function that can be used in CopilotChat's contextual prompt
  make_context_provider = make_context_provider,
}
