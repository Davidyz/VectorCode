---
name: Bug report
about: Create a report to help us improve
title: "[BUG]"
labels: bug
assignees: ''

---

**Describe the bug**
A clear and concise description of what the bug is.

**To Reproduce**
Steps to reproduce the behavior:
1. ...
2. ...

**Expected behavior**
A clear and concise description of what you expected to happen.

**VectorCode Configuration**
Please attach your `<project_root>/.vectorcode/config.json` or 
  `~/.config/vectorcode/config.json` here.
```json

```

For issues with the Neovim plugin, please also attach your `setup` options:
```lua

```
If it only occurs when you use VectorCode with a particular plugin, please
attach the relevant config here:
```lua

```

**Platform information:**
 - If the issue is about the CLI, attach a list of packages in the Python virtual environment:
   - for `pipx`, run `pipx runpip vectorcode freeze`;
   - for `uv`, run `uv tool run --from=vectorcode python -m ensurepip && uv tool run --from=vectorcode python -m pip freeze`.
```

```
 - If the issue is about the neovim plugin, attach the neovim version you're using:


**System Information:**

> For Mac users, please also mention whether you're using intel or apple silicon devices.

 - OS: Linux, MacOS, Windows...

**Additional context**
Add any other context about the problem here. Please attach 
[CLI logs](https://github.com/Davidyz/VectorCode/blob/main/docs/cli.md#debugging-and-diagnosing) 
or 
[nvim plugin logs](https://github.com/Davidyz/VectorCode/blob/main/docs/neovim.md#debugging-and-logging) 
if applicable.
