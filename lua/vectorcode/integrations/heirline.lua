---@class VectorCode.Heirline.Opts: VectorCode.Lualine.Opts
--- Other heirline component fields (like `hl`, `on_click`, `update`, etc.)
---@field component_opts table

---@type VectorCode.Heirline.Opts
local default_opts = { show_job_count = false, component_opts = {} }

---@param opts VectorCode.Heirline.Opts?
return function(opts)
  opts = vim.tbl_deep_extend("force", default_opts, opts or {}) --[[@as VectorCode.Heirline.Opts]]
  local lualine_comp = require("vectorcode.integrations").lualine(opts)
  local heirline_component = {
    provider = function(_)
      return lualine_comp[1]()
    end,
    condition = function(_)
      return lualine_comp.cond()
    end,
  }

  return vim.tbl_deep_extend("force", heirline_component, opts.component_opts)
end
