local M = {}

local function traverse(node, cb)
  if node == nil then
    return
  end
  if node.result ~= nil then
    traverse(node.result, cb)
  end
  if vim.isarray(node) then
    for _, v in pairs(node) do
      traverse(v, cb)
    end
    return
  end
  if vim.isarray(node.children) then
    for _, v in pairs(node.children) do
      traverse(v, cb)
    end
  end
  if not vim.list_contains({ 15, 16, 20, 21, 25 }, node.kind) then
    -- exclude certain kinds.
    if cb then
      cb(node)
    end
  end
end

---@alias VectorCode.QueryCallback fun(bufnr:integer?):string|string[]

---Retrieves all LSP document symbols from the current buffer, and use the symbols
---as query messages. Fallbacks to `make_surrounding_lines_cb` if
---`textDocument_documentSymbol` is not accessible.
---@return VectorCode.QueryCallback
function M.make_lsp_document_symbol_cb()
  return function(bufnr)
    if bufnr == 0 or bufnr == nil then
      bufnr = vim.api.nvim_get_current_buf()
    end
    local has_documentSymbol = false
    for _, client in ipairs(vim.lsp.get_clients({ bufnr = bufnr })) do
      if client.server_capabilities.documentSymbolProvider then
        has_documentSymbol = true
      end
    end
    if not has_documentSymbol then
      return M.make_surrounding_lines_cb(-1)(bufnr)
    end

    local result, _ = vim.lsp.buf_request_sync(
      0,
      vim.lsp.protocol.Methods.textDocument_documentSymbol,
      { textDocument = vim.lsp.util.make_text_document_params(bufnr) }
    )
    if result ~= nil then
      local symbols = {}
      traverse(result, function(node)
        if node.name ~= nil then
          vim.list_extend(symbols, { node.name })
        end
      end)
      return symbols
    else
      return M.make_surrounding_lines_cb(20)(bufnr)
    end
  end
end

---Use the lines above and below the current line as the query messages.
---@param num_of_lines integer The number of lines to include in the query.
---@return VectorCode.QueryCallback
function M.make_surrounding_lines_cb(num_of_lines)
  return function(bufnr)
    if bufnr == 0 or bufnr == nil then
      bufnr = vim.api.nvim_get_current_buf()
    end
    if num_of_lines <= 0 then
      return table.concat(vim.api.nvim_buf_get_lines(bufnr, 0, -1, false), "\n")
    end
    local cursor_line = vim.api.nvim_win_get_cursor(0)[1]
    local start_line = cursor_line - math.floor(num_of_lines / 2)
    if start_line < 1 then
      start_line = 1
    end
    return table.concat(
      vim.api.nvim_buf_get_lines(
        bufnr,
        start_line - 1,
        start_line + num_of_lines - 1,
        false
      ),
      "\n"
    )
  end
end

---@param path string|integer
---@return string?
function M.find_root(path)
  return vim.fs.root(path, ".vectorcode") or vim.fs.root(path, ".git")
end

---@param str string
---@param sep string?
---@return string[]
local function split(str, sep)
  if sep == nil then
    sep = " "
  end
  local result = {}
  local pattern = "([^" .. sep .. "]+)"
  for part in string.gmatch(str, pattern) do
    table.insert(result, part)
  end
  return result
end

--- This function build a `VectorCode.QueryCallback` by extracting recent changes from the `:changes` command.
---@param max_num integer? Default is 50
---@return VectorCode.QueryCallback
function M.make_changes_cb(max_num)
  if max_num == nil then
    max_num = 50
  end
  return function(bufnr)
    ---@type string?
    local raw_changes = vim.api.nvim_exec2("changes", { output = true }).output
    if raw_changes == nil then
      -- fallback to other cb
      return M.make_surrounding_lines_cb(-1)(bufnr)
    end
    local lines = vim.tbl_map(function(s)
      local res = string.gsub(s, "^[%d%s]+", "")
      return res
    end, split(raw_changes, "\n"))
    local results = {}
    local seen = {} -- deduplicate
    for i = #lines - 1, 2, -1 do
      if #results <= max_num then
        if not seen[lines[i]] then
          table.insert(results, lines[i])
          seen[lines[i]] = true
        end
      else
        break
      end
    end
    if #results == 0 then
      -- fallback to other cb
      return M.make_surrounding_lines_cb(-1)(bufnr)
    end
    return results
  end
end

---@param f string
---@return boolean
function M.is_file(f)
  if type(f) ~= "string" then
    return false
  end
  local stats = vim.uv.fs_stat(f)
  return stats and (stats.type == "file") or false
end

---@param f string
---@return boolean
function M.is_directory(f)
  if type(f) ~= "string" then
    return false
  end
  local stats = vim.uv.fs_stat(f)
  return stats and (stats.type == "directory") or false
end

---@param t table|string|nil
---@param fallback string?
---@return string
M.flatten_table_to_string = function(t, fallback)
  fallback = fallback or ""
  if t == nil then
    return fallback
  end
  if type(t) == "string" then
    return t
  end

  -- Handle empty tables or tables with empty strings
  local flattened = vim
    .iter(t)
    :flatten(math.huge)
    :filter(function(item)
      return type(item) == "string" and vim.trim(item) ~= ""
    end)
    :totable()

  if #flattened == 0 then
    return fallback
  end

  return table.concat(flattened, "\n")
end

---Convert any `vim.NIL` instances to `nil` in lua.
---@generic Obj: any
---@param obj Obj
---@return Obj
function M.fix_nil(obj)
  if obj == vim.NIL then
    return nil
  end
  if type(obj) == "table" then
    for k, v in pairs(obj) do
      obj[k] = M.fix_nil(v)
    end
  end
  return obj
end

return M
