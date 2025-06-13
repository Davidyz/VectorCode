---@module "codecompanion"

local job_runner
local vc_config = require("vectorcode.config")
local notify_opts = vc_config.notify_opts
local logger = vc_config.logger

---@type VectorCode.CodeCompanion.ToolOpts
local default_options = {
  max_num = { chunk = -1, document = -1 },
  default_num = { chunk = 50, document = 10 },
  include_stderr = false,
  use_lsp = false,
  auto_submit = { ls = false, query = false },
  ls_on_start = false,
  no_duplicate = true,
  chunk_mode = false,
}

local TOOL_RESULT_SOURCE = "VectorCodeToolResult"

return {
  tool_result_source = TOOL_RESULT_SOURCE,
  ---@param t table|string
  ---@return string
  flatten_table_to_string = function(t)
    if type(t) == "string" then
      return t
    end
    return table.concat(vim.iter(t):flatten(math.huge):totable(), "\n")
  end,

  ---@param opts VectorCode.CodeCompanion.ToolOpts|{}|nil
  ---@return VectorCode.CodeCompanion.ToolOpts
  get_tool_opts = function(opts)
    if opts == nil or opts.use_lsp == nil then
      opts = vim.tbl_deep_extend(
        "force",
        opts or {},
        { use_lsp = vc_config.get_user_config().async_backend == "lsp" }
      )
    end
    opts = vim.tbl_deep_extend("force", default_options, opts)
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
    return opts
  end,

  ---@param result VectorCode.Result
  ---@return string
  process_result = function(result)
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
  end,

  ---@param use_lsp boolean
  ---@return VectorCode.JobRunner
  initialise_runner = function(use_lsp)
    if job_runner == nil then
      if use_lsp then
        job_runner = require("vectorcode.jobrunner.lsp")
      end
      if job_runner == nil then
        job_runner = require("vectorcode.jobrunner.cmd")
        logger.info("Using cmd runner for CodeCompanion tool.")
        if use_lsp then
          vim.schedule_wrap(vim.notify)(
            "Failed to initialise the LSP runner. Falling back to cmd runner.",
            vim.log.levels.WARN,
            notify_opts
          )
        end
      else
        logger.info("Using LSP runner for CodeCompanion tool.")
      end
    end
    return job_runner
  end,

  ---@param results VectorCode.Result[]
  ---@param chat CodeCompanion.Chat
  ---@return VectorCode.Result[]
  filter_results = function(results, chat)
    local existing_refs = chat.refs
    if existing_refs == nil then
      return results
    end
    existing_refs = vim
      .iter(existing_refs)
      :filter(
        ---@param ref CodeCompanion.Chat.Ref
        function(ref)
          return ref.source == TOOL_RESULT_SOURCE or ref.path or ref.bufnr
        end
      )
      :map(
        ---@param ref CodeCompanion.Chat.Ref
        function(ref)
          if ref.source == TOOL_RESULT_SOURCE then
            return ref.id
          elseif ref.path then
            return ref.path
          elseif ref.bufnr then
            return vim.api.nvim_buf_get_name(ref.bufnr)
          end
        end
      )
      :totable()

    return vim
      .iter(results)
      :filter(
        ---@param res VectorCode.Result
        function(res)
          -- return true if res is not in refs
          if res.chunk then
            if res.chunk_id == nil then
              return true
            end
            return not vim.tbl_contains(existing_refs, res.chunk_id)
          else
            return not vim.tbl_contains(existing_refs, res.path)
          end
        end
      )
      :totable()
  end,
}
