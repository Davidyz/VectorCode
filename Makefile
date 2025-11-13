EXTRA_LOCK_ARGS?=
EXTRA_DEPS?=
EXTRA_COVERAGEPY_ARGS?=

LOADED_DOT_ENV=@if [ -f .env ] ; then source .env; fi;

DEFAULT_GROUPS=--group dev --group lsp --group mcp --group debug $(EXTRA_LOCK_ARGS)

.PHONY: multitest

deps:
	pdm lock $(DEFAULT_GROUPS) || pdm lock $(DEFAULT_GROUPS) --group legacy; \
	pdm install
	[ -z "$(EXTRA_DEPS)" ] || (pdm run python -m ensurepip && pdm run python -m pip install $(EXTRA_DEPS))
	
test:
	make deps; \
	pdm run pytest --enable-coredumpy --coredumpy-dir dumps

multitest:
	@for i in {11..13}; do \
		pdm use python3.$$i; \
		make test; \
	done

coverage:
	make deps; \
	pdm run coverage run $(EXTRA_COVERAGEPY_ARGS) -m pytest --enable-coredumpy --coredumpy-dir dumps; \
	pdm run coverage html; \
	pdm run coverage report -m

lint:
	pdm run ruff check src/**/*.py; \
	pdm run basedpyright src/**/*.py; \
	selene lua/**/*.lua plugin/*.lua
