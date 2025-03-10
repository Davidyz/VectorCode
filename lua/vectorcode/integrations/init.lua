return {
  codecompanion = require("vectorcode.integrations.codecompanion"),
  copilotchat = require("vectorcode.integrations.copilotchat"),
  lualine = require("vectorcode.integrations.lualine"),
}
-- local copilot_chat_config = {
--   -- other config...
--
--   context_provider = {
--     -- your other context providers...
--
--     vectorcode = {
--       description = "Inject VectorCode context",
--       resolve = require("vectorcode.integrations.copilotchat").make_context_provider({
--         -- Optional: customize the integration
--         prompt_header = "Here's some relevant code from the repository:",
--         prompt_footer = "\nBased on this context, please: \n",
--         -- skip_empty = true, -- Skip when there are no results
--       })
--     }
--   }
-- }
