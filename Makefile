.PHONY: multitest

deps:
	uv sync --group dev --extra lsp --extra mcp
	
test:
	make deps; \
	uv run pytest --enable-coredumpy --coredumpy-dir dumps

multitest:
	@for i in {11..13}; do \
		uv venv -p python3.$$i; \
		make test || exit 1; \
	done

coverage:
	make deps; \
	uv run coverage run -m pytest; \
	uv run coverage html; \
	uv run coverage report -m
