local vc_config = require("vectorcode.config")

---@class VectorCode.Lualine.Opts
---Whether to show the number of running async jobs.
---@field show_job_count boolean

---@param opts VectorCode.Lualine.Opts?
return function(opts)
  opts = vim.tbl_deep_extend("force", { show_job_count = false }, opts or {}) --[[@as VectorCode.Lualine.Opts]]
  local cacher = vc_config.get_cacher_backend()
  return {
    function()
      local message = "VectorCode: "
      if cacher.buf_is_enabled(0) then
        local retrieval = cacher.query_from_cache(0, { notify = false })
        if retrieval then
          message = message .. tostring(#retrieval)
        end
        local job_count = cacher.buf_job_count(0)
        if job_count > 0 then
          if opts.show_job_count then
            message = message .. (" (%d) "):format(job_count)
          else
            message = message .. "  "
          end
        else
          message = message .. "  "
        end
      else
        message = message .. " "
      end
      return message
    end,
    cond = function()
      return cacher.buf_is_registered()
    end,
  }
end
