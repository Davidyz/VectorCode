---@class VectorCode.JobRunner
---@field run_async fun(args: string[], callback:fun(result: table, error: table)?, bufnr: integer):(job_handle:integer?)
---@field run fun(args: string[], timeout_ms: integer?, bufnr: integer):(result:string[], error:string[])
---@field is_job_running fun(job_handle: integer):boolean
---@field stop_job fun(job_handle: integer)

return {}
