# VectorCode

VectorCode is a code repository indexing tool. It helps you write better prompt
for your coding LLMs by indexing and providing information about the code
repository you're working on. This repository also contains the corresponding
neovim plugin because that's what I used to write this tool.

> [!NOTE]
> This project is in beta quality and only implements very basic retrieval and
> embedding functionalities. There are plenty of rooms for improvements and any
> help is welcomed.

> [!NOTE]
> [Chromadb](https://www.trychroma.com/), the vector database backend behind
> this project, supports multiple embedding engines. I developed this tool using
> SentenceTransformer, but if you encounter any issues with a different embedding 
> function, please open an issue (or even better, a pull request :D).

<!-- mtoc-start -->

* [Why VectorCode?](#why-vectorcode)
* [Documentation](#documentation)
* [TODOs](#todos)

<!-- mtoc-end -->

## Why VectorCode?
LLMs usually have very limited understanding about close-source and/or infamous 
projects, as well as cutting edge developments that have not made it into the
releases. Their capabilities on these projects are quite limited. Take my little
toy sudoku-solving project as an example: When I wrote the first few lines and
want the LLM to fill in the list of solvers that I implemented in
`solver_candidates`, without project context, the completions are simply random 
guesses that *might* be part of another sudoku project:
![](./images/sudoku_no_rag.png)
But with RAG context provided by VectorCode, my completion LLM was able to
provide completions that I actually implemented:
![](./images/sudoku_with_rag.png)
This makes the completion results far more usable. 
[A similar strategy](https://docs.continue.dev/customize/deep-dives/codebase) 
is implemented in [continue](https://www.continue.dev/), a popular AI completion
and chat plugin available on VSCode and JetBrain products.

## Documentation

- For the setup and usage of the command-line tool, see [the CLI documentation](./docs/cli.md);
- For neovim users, after you've gone through the CLI documentation, please refer to 
  [the neovim plugin documentation](./docs/neovim.md) for further instructions.

## TODOs
- [x] query by ~file path~ excluded paths;
- [ ] chunking support;
  - [x] add metadata for files;
  - [x] chunk-size configuration;
  - [ ] smarter chunking (semantics/syntax based);
  - [x] configurable document selection from query results.
- [x] ~NeoVim Lua API with cache to skip the retrieval when a project has not
  been indexed~ Returns empty array instead;
- [x] job pool for async caching;
- [x] [persistent-client](https://docs.trychroma.com/docs/run-chroma/persistent-client);
- [x] proper [remote Chromadb](https://docs.trychroma.com/production/administration/auth) support (with authentication, etc.);
- [x] respect `.gitignore`;
- [x] implement some sort of project-root anchors (such as `.git` or a custom
  `.vectorcode.json`) that enhances automatic project-root detection.
  **Implemented project-level `.vectorcode/config.json` as root anchor**
