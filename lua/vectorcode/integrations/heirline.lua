---@class VectorCode.Heirline.Opts: VectorCode.Lualine.Opts
local default_opts = { show_job_count = false }

---@param opts VectorCode.Heirline.Opts?
return function(opts)
  opts = vim.tbl_deep_extend("force", default_opts, opts or {}) --[[@as VectorCode.Heirline.Opts]]
  local lualine_comp = require("vectorcode.integrations").lualine(opts)
  return {
    provider = function(_)
      return lualine_comp[1]()
    end,
    condition = function(_)
      return lualine_comp.cond()
    end,
  }
end
