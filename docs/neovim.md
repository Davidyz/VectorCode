# NeoVim Plugin
> [!NOTE]
> This plugin depends on the CLI tool. Please go through 
> [the CLI documentation](./cli.md) and make sure the VectorCode CLI is working
> before proceeding.

> [!NOTE]
> When the neovim plugin doesn't work properly, please try upgrading the CLI
> tool to the latest version before opening an issue.


<!-- mtoc-start -->

* [Installation](#installation)
* [Quick Start](#quick-start)
* [User Command](#user-command)
  * [`VectorCode register`](#vectorcode-register)
  * [`VectorCode deregister`](#vectorcode-deregister)
* [Configuration](#configuration)
  * [`setup(opts?)`](#setupopts)
* [API Usage](#api-usage)
  * [Synchronous API](#synchronous-api)
    * [`query(query_message, opts?)`](#queryquery_message-opts)
    * [`check(check_item?)`](#checkcheck_item)
  * [Cached Asynchronous API](#cached-asynchronous-api)
    * [`register_buffer(bufnr?, opts?)`](#register_bufferbufnr-opts)
    * [`query_from_cache(bufnr?)`](#query_from_cachebufnr)
    * [`async_check(check_item?, on_success?, on_failure?)`](#async_checkcheck_item-on_success-on_failure)
    * [`buf_is_registered(bufnr?)`](#buf_is_registeredbufnr)
* [Integrations](#integrations)
  * [olimorris/codecompanion.nvim](#olimorriscodecompanionnvim)
  * [nvim-lualine/lualine.nvim:](#nvim-lualinelualinenvim)

<!-- mtoc-end -->

## Installation
Use your favourite plugin manager. 

```lua 
{
  "Davidyz/VectorCode",
  version = "*", -- optional, depending on whether you're on nightly or release
  dependencies = { "nvim-lua/plenary.nvim" },
}
```
The VectorCode CLI and neovim plugin share the same release scheme (version
numbers). In other words, CLI 0.1.3 is guaranteed to work with neovim plugin
0.1.3, but if you use CLI 0.1.0 with neovim plugin 0.1.3, they may not work
together because the neovim plugin is built for a newer CLI release.

To ensure maximum compatibility, please either:
1. Use release build for VectorCode CLI and pin to the release tags for the
   neovim plugin;

**OR**

2. Use the latest commit for the neovim plugin with VectorCode installed from
   the latest GitHub commit.

It may be helpful to use a `build` hook to automatically upgrade the CLI when
the neovim plugin updates. For example, if you're using lazy.nvim and `pipx`,
you can use the following plugin spec:

```lua
{
  "Davidyz/VectorCode",
  version = "*", -- optional, depending on whether you're on nightly or release
  build = "pipx upgrade vectorcode",
  dependencies = { "nvim-lua/plenary.nvim" },
}
```

## Quick Start

For VectorCode to work with a LLM plugin, you need to somehow integrate the
query results into the prompt.

Here's how VectorCode may be used with 
[minuet-ai.nvim](https://github.com/milanglacier/minuet-ai.nvim):
```lua
{
  "milanglacier/minuet-ai.nvim",
  config = function()
    -- This uses the async cache to accelerate the prompt construction.
    -- There's also the require('vectorcode').query API, which provides 
    -- more up-to-date information, but at the cost of blocking the main UI.
    local vectorcode_cacher = require("vectorcode.cacher")
    require("minuet").setup({
      add_single_line_entry = true,
      n_completions = 1,
      -- I recommend you start with a small context window firstly, and gradually
      -- increase it based on your local computing power.
      context_window = 4096,
      after_cursor_filter_length = 30,
      notify = "debug",
      provider = "openai_fim_compatible",
      provider_options = {
        openai_fim_compatible = {
          api_key = "TERM",
          name = "Ollama",
          stream = true,
          end_point = "http://127.0.0.1:11434/v1/completions",
          model = "qwen2.5-coder:7b-base-q4_1",
          template = {
            prompt = function(pref, suff)
              local prompt_message = ""
              for _, file in ipairs(vectorcode_cacher.query_from_cache(0)) do
                prompt_message = prompt_message .. "<|file_sep|>" .. file.path .. "\n" .. file.document
              end
              return prompt_message
                .. "<|fim_prefix|>"
                .. pref
                .. "<|fim_suffix|>"
                .. suff
                .. "<|fim_middle|>"
            end,
            suffix = false,
          },
        },
      },
    })
  end,
}
```

> [!NOTE]
> It can be challenging to find the best way to incorporate the project-level
> context into the prompt. The template above works well for [`Qwen2.5-Coder`](https://github.com/QwenLM/Qwen2.5-Coder),
> but may not work as intended for your LLM. I compiled the prompt construction 
> code snippets for some other LLMs
[in the wiki](https://github.com/Davidyz/VectorCode/wiki/Configuration-Gallery).
> Please check it out, because if your model was trained with project-level
> context support, you'll have to modify the prompt structure accordingly to maximise
> its potential.

To use [async cache](#cached-asynchronous-api), you need to register the buffer.
You may either manually register a buffer using the [user command](#vectorcode-register)
`VectorCode register`, or set up an autocommand:
```lua
vim.api.nvim_create_autocmd("LspAttach", {
  callback = function()
    local bufnr = vim.api.nvim_get_current_buf()
    cacher.async_check("config", function()
      cacher.register_buffer(
        bufnr,
        { 
          notify = false, 
          n_query = 10, 
          query_cb = require("vectorcode.utils").lsp_document_symbol_cb(),
          events = { "BufWritePost" }
        }
      )
    end, nil)
  end,
  desc = "Register buffer for VectorCode",
})
```
to automatically register a new buffer. The autocommand is more convenient, but
if you open a lot of buffers at the same time your system may be overloaded by
the query commands. Using the user command allows you to choose what buffer to register. 
Carefully choose how you register your buffers according to your system specs and setup.

## User Command
### `VectorCode register`

Register the current buffer for async caching.

### `VectorCode deregister`

Deregister the current buffer. Any running jobs will continue to run, but no
further jobs will be scheduled.

## Configuration

### `setup(opts?)`
This function initialises the VectorCode client and sets up some default

```lua
require("vectorcode").setup({
    n_query = 1,
})
```

The following are the available options for this function:
- `n_query`: number of retrieved documents. Default: `1`;
- `notify`: whether to show notifications when a query is completed.
  Default: `true`;
- `timeout_ms`: timeout in milliseconds for the query operation. Default: 
  `5000` (5 seconds);
- `exclude_this`: whether to exclude the file you're editing. Setting this to
  `false` may lead to an outdated version of the current file being sent to the
  LLM as the prompt, and can lead to generations with outdated information.

## API Usage
This plugin provides 2 sets of APIs that provides similar functionalities. The
synchronous APIs provide more up-to-date retrieval results at the cost of
blocking the main neovim UI, while the async APIs use a caching mechanism to 
provide asynchronous retrieval results almost instantaneously, but the result
may be slightly out-of-date. For some tasks like chat, the main UI being
blocked/frozen doesn't hurt much because you spend the time waiting for response
anyway, and you can use the synchronous API in this case. For other tasks like 
completion, the async API will minimise the interruption to your workflow.


### Synchronous API
#### `query(query_message, opts?)`
This function queries VectorCode and returns an array of results.

```lua
require("vectorcode").query("some query message", {
    n_query = 5,
})
```

The following are the available options for this function:
- `n_query`: number of retrieved documents. Default: `1`;
- `notify`: whether to show notifications when a query is completed.
  Default: `true`;
- `timeout_ms`: timeout in milliseconds for the query operation. Default: 
  `5000` (5 seconds).
The return value of this function is an array of results in the format of
`{path="path/to/your/code.lua", document="document content"}`. 

For example, in [cmp-ai](https://github.com/tzachar/cmp-ai), you can add 
the path/document content to the prompt like this:
```lua
prompt = function(prefix, suffix)
    local retrieval_results = require("vectorcode").query("some query message", {
        n_query = 5,
    })
    for _, source in pairs(retrieval_results) do
        -- This works for qwen2.5-coder.
        file_context = file_context
            .. "<|file_sep|>"
            .. "path/to/your/code.lua"
            .. "\n"
            .. source.document
            .. "\n"
    end
    return file_context
        .. "<|fim_prefix|>" 
        .. prefix 
        .. "<|fim_suffix|>" 
        .. suffix 
        .. "<|fim_middle|>"
end
```


#### `check(check_item?)`
This function checks if VectorCode has been configured properly for your project. See the [CLI manual for details](./cli.md).

```lua 
require("vectorcode").check()
```

The following are the available options for this function:
- `check_item`: Only supports `"config"` at the moment. Checks if a project-local 
  config is present.
  Return value: `true` if passed, `false` if failed.

This involves the `check` command of the CLI that checks the status of the
VectorCode project setup. Use this as a pre-condition of any subsequent
use of other VectorCode APIs that may be more expensive (if this fails,
VectorCode hasn't been properly set up for the project, and you should not use
VectorCode APIs).

The use of this API is entirely optional. You can totally ignore this and call
`query` anyway, but if `check` fails, you might be spending the waiting time for
nothing.

### Cached Asynchronous API

The async cache mechanism helps mitigate the issue where the `query` API may
take too long and block the main thread.

#### `register_buffer(bufnr?, opts?)`
This function registers a buffer to be cached by VectorCode.

```lua
require("vectorcode.cacher").register_buffer(0, {
    n_query = 1,
})
```

The following are the available options for this function:
- `bufnr`: buffer number. Default: current buffer;
- `opts`: a table that accepts all options supported by `setup` and the following:
  - `query_cb`: a callback function that takes the buffer number as the only
    argument and returns the query message. Some examples are bundled in the
    plugin, accessible in `require("vectorcode.utils")` Default: 
    `require("vectorcode.utils").surrounding_lines_cb(-1)`, which queries the full buffer;
  - `events`: a list of events to trigger the query. Default:
    `{"BufWritePost", "InsertEnter", "BufReadPost"}`;
  - `debounce`: debounce time in seconds for the query. Default: `10`;
  - `run_on_register`: boolean, whether to trigger the query immediately after
    the registration.


#### `query_from_cache(bufnr?)`
This function queries VectorCode from cache.

```lua
require("vectorcode.cacher").query_from_cache()
```

The following are the available options for this function:
- `bufnr`: buffer number. Default: current buffer.

Return value: an array of results. Each item of the array is in the format of 
`{path="path/to/your/code.lua", document="document content"}`.

#### `async_check(check_item?, on_success?, on_failure?)`
This function checks if VectorCode has been configured properly for your project.

```lua 
require("vectorcode.cacher").async_check(
    "config", 
    do_something(),
    do_something_else()
)
```

The following are the available options for this function:
- `check_item`: any check that works with `vectorcode check` command. If not set, 
  it defaults to `"config"`;
- `on_success`: a callback function that is called when the check passes;
- `on_failure`: a callback function that is called when the check fails.

#### `buf_is_registered(bufnr?)`
This function checks if a buffer has been registered with VectorCode.

```lua 
require("vectorcode.cacher").buf_is_registered()
```

The following are the available options for this function:
- `bufnr`: buffer number. Default: current buffer.
Return value: `true` if registered, `false` otherwise.

## Integrations

`require("vectorcode.integrations")` provides integration utilities for some
other neovim plugins.

### [olimorris/codecompanion.nvim](https://github.com/olimorris/codecompanion.nvim)
A [slash command](https://codecompanion.olimorris.dev/usage/chat-buffer/slash-commands.html#using-slash-commands)
that adds VectorCode retrieval results to the prompt sent to the LLM to enhance
its understanding about your repository.

```lua
opts =
  {
    -- your other codecompanion configs
    strategies = {
      chat = {
        adapter = "your adapter",
        slash_commands = {
          -- add the vectorcode command here.
          codebase = require("vectorcode.integrations").codecompanion.chat.slash_command,
        },
      },
    },
  }  
```

### [nvim-lualine/lualine.nvim](https://github.com/nvim-lualine/lualine.nvim):
A `lualine` component that shows the status of the async job and the number of
cached retrieval results.
```lua 
tabline = {
  lualine_y = {
    require("vectorcode.integrations").lualine()
  }
}
```
