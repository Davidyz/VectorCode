This project uses [pre-commit](https://pre-commit.com/) to perform some
formatting and linting. If you're
contributing to this project, having it on your system will help you write code
that passes the CI. 
You can also see 
[.pre-commit-config.yaml](https://github.com/Davidyz/VectorCode/blob/main/.pre-commit-config.yaml) 
for a list of hooks enabled for the repo.

# Python CLI

The development and publication of this tool is managed by 
[pdm](https://pdm-project.org/en/latest/).

Once you've cloned and `cd`ed into the repo, run `make deps`. This will call
some `pdm` commands to install development dependencies. Some of them are
actually optional, but for convenience I decided to leave them here. This will
include [pytest](https://docs.pytest.org/en/stable/), the testing framework, 
and [coverage.py](https://coverage.readthedocs.io/en/7.7.1/), the coverage
report tool. If you're not familiar with pytest or coverage.py, you can run `make test` to
run tests on all python code, and `make coverage` to generate a coverage report. 
The testing and coverage report are also in the CI configuration, but it might 
still help to run them locally before you open the PR.

This project also runs static analysis with
[basedpyright](https://docs.basedpyright.com). GitHub Action will also run the
check when a PR is submitted. This, as well as `ruff check`, are both included
in `make lint`.

You may also find it helpful to 
[enable logging](https://github.com/Davidyz/VectorCode/blob/main/docs/cli.md#debugging-and-diagnosing) 
for the CLI when developing new features or working on fixes.

# Neovim Plugin

At the moment, there isn't much to cover on here. As long as the code is 
formatted (stylua) and appropriately type-annotated, you're good. I do have 
plans to write some tests, but before that happens, formatting and type 
annotations are the only things that you need to take special care of.

The lua codebase is linted by [selene](https://github.com/Kampfkarren/selene).
You may run `make lint` or call `selene` from the CLI to lint the code.

You may find it useful to 
[enable logging](https://github.com/Davidyz/VectorCode/blob/main/docs/cli.md#debugging-and-diagnosing) 
when you're poking around the codebase.
