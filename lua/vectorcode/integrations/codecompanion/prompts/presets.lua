---@type table<string, VectorCode.CodeCompanion.PromptFactory.Opts>
local M = {}

M["Neovim Tutor"] = {
  project_root = vim.fs.normalize(vim.env.VIMRUNTIME),
  file_patterns = { "lua/**/*.lua", "doc/**/*.txt" },
}

return M
