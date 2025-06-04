local vc_config = require("vectorcode.config")
local notify_opts = vc_config.notify_opts

---@param args string[]?
---@return table<string, any>
local function process_args(args)
  if args == nil then
    return {}
  end
  local result = {}
  for _, str in pairs(args) do
    local equal = string.find(str, "=")
    if equal then
      local key = string.sub(str, 1, equal - 1)
      local value = string.sub(str, equal + 1) --[[@as any]]
      result[key] = value
    end
  end
  return result
end

vim.api.nvim_create_user_command("VectorCode", function(args)
  local cacher = vc_config.get_cacher_backend()
  local splitted_args = vim.tbl_filter(function(str)
    return str ~= nil and str ~= ""
  end, vim.split(args.args, " "))
  local action = table.remove(splitted_args, 1)
  if action == "register" then
    local bufnr = vim.api.nvim_get_current_buf()
    cacher.register_buffer(bufnr, {
      run_on_register = true,
      project_root = process_args(splitted_args).project_root,
    })
    vim.notify(
      ("Buffer %d has been registered for VectorCode."):format(bufnr),
      vim.log.levels.INFO,
      notify_opts
    )
  elseif action == "deregister" then
    local buf_nr = vim.api.nvim_get_current_buf()
    cacher.deregister_buffer(buf_nr, { notify = true })
  else
    vim.notify(
      ([[Command "VectorCode %s" was not recognised.]]):format(args.args),
      vim.log.levels.ERROR,
      notify_opts
    )
  end
end, {
  nargs = 1,
  complete = function(arglead, cmd, _)
    local cacher = vc_config.get_cacher_backend()
    local splitted_cmd = vim.tbl_filter(function(str)
      return str ~= nil and str ~= ""
    end, vim.split(cmd, " "))

    if #splitted_cmd < 2 then
      if cacher.buf_is_registered(0) then
        return { "register", "deregister" }
      else
        return { "register" }
      end
    elseif #splitted_cmd == 2 and splitted_cmd[2] == "register" then
      return { "project_root=" }
    elseif splitted_cmd[2] == "register" and #splitted_cmd == 3 then
      local prefix = "project_root="
      if string.find(splitted_cmd[3], prefix) == 1 then
        local partial = arglead:sub(#prefix + 1)
        local dirs = vim.fn.getcompletion(partial, "dir")
        for i = 1, #dirs do
          dirs[i] = prefix .. dirs[i]
        end
        return dirs
      end
    end
  end,
})
