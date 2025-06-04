.PHONY: multitest

deps:
	pdm lock --group dev --group lsp --group mcp; \
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
