local M = {}

local function traverse(node)
  if vim.isarray(node) then
    for k, v in pairs(node) do
      node[k] = traverse(v)
    end
  end
  if vim.isarray(node.children) then
    for k, v in pairs(node.children) do
      node.children[k] = traverse(v)
    end
  end
  node.selectionRange = nil
  if not vim.list_contains({ 15, 16, 21, 25 }, node.kind) then
    -- exclude certain kinds.
    return node
  end
end

---@alias VectorCodeQueryCallback fun(bufnr:integer?):string

---@return VectorCodeQueryCallback
function M.lsp_document_symbol_cb()
  return function(bufnr)
    if bufnr == 0 or bufnr == nil then
      bufnr = vim.api.nvim_get_current_buf()
    end
    local ok, result = pcall(
      vim.lsp.buf_request_sync,
      0,
      vim.lsp.protocol.Methods.textDocument_documentSymbol,
      { textDocument = vim.lsp.util.make_text_document_params(bufnr) }
    )
    if ok then
      return vim.json.encode(traverse(result))
    else
      return M.surrounding_lines_cb(-1)(bufnr)
    end
  end
end

---@param num_of_lines integer
---@return VectorCodeQueryCallback
function M.surrounding_lines_cb(num_of_lines)
  assert(num_of_lines > 0)
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

return M
