.PHONY: multitest

DEFAULT_GROUPS=--group dev --group lsp --group mcp

deps:
	pdm lock $(DEFAULT_GROUPS) || pdm lock $(DEFAULT_GROUPS) --group legacy; \
	pdm install
	
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
	pdm run coverage run -m pytest; \
	pdm run coverage html; \
	pdm run coverage report -m

lint:
	pdm run ruff check src/**/*.py; \
	pdm run basedpyright src/**/*.py; \
	selene lua/**/*.lua plugin/*.lua
