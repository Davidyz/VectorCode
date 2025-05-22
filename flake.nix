{
  description = "VectorCode";
  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };
  outputs =
    {
      nixpkgs,
      self,
      ...
    }@inputs:
    inputs.flake-utils.lib.eachDefaultSystem (
      system:
      let
        pkgs = import nixpkgs { inherit system; };

        # INFO: This is a workaround until newer versions of the `chromadb` package are available
        chromadb = pkgs.python312Packages.buildPythonPackage rec {
          pname = "chromadb";
          version = "0.6.3";
          pyproject = true;

          src = pkgs.fetchFromGitHub {
            owner = "chroma-core";
            repo = "chroma";
            tag = version;
            hash = "sha256-yvAX8buETsdPvMQmRK5+WFz4fVaGIdNlfhSadtHwU5U=";
          };

          cargoDeps = pkgs.rustPlatform.fetchCargoVendor {
            inherit src;
            name = "${pname}-${version}";
            hash = "sha256-lHRBXJa/OFNf4x7afEJw9XcuDveTBIy3XpQ3+19JXn4=";
          };

          pythonRelaxDeps = [
            "chroma-hnswlib"
            "orjson"
          ];

          build-system = with pkgs.python312Packages; [
            setuptools
            setuptools-scm
          ];

          nativeBuildInputs = with pkgs; [
            cargo
            pkg-config
            protobuf
            rustc
            rustPlatform.cargoSetupHook
          ];

          buildInputs = with pkgs; [
            openssl
            zstd
          ];

          dependencies = with pkgs.python312Packages; [
            bcrypt
            build
            chroma-hnswlib
            fastapi
            grpcio
            httpx
            importlib-resources
            kubernetes
            mmh3
            numpy
            onnxruntime
            opentelemetry-api
            opentelemetry-exporter-otlp-proto-grpc
            opentelemetry-instrumentation-fastapi
            opentelemetry-sdk
            orjson
            overrides
            posthog
            pulsar-client
            pydantic
            pypika
            pyyaml
            requests
            tenacity
            tokenizers
            tqdm
            typer
            typing-extensions
            uvicorn
          ];

          nativeCheckInputs = with pkgs.python312Packages; [
            hypothesis
            psutil
            pytest-asyncio
            pytestCheckHook
          ];

          pythonImportsCheck = [ "chromadb" ];

          env = {
            ZSTD_SYS_USE_PKG_CONFIG = true;
          };

          pytestFlagsArray = [ "-x" ];

          preCheck = ''
            (($(ulimit -n) < 1024)) && ulimit -n 1024
            export HOME=$(mktemp -d)
          '';

          doCheck = false;

          __darwinAllowLocalNetworking = true;

          meta = with pkgs.lib; {
            description = "AI-native open-source embedding database";
            homepage = "https://github.com/chroma-core/chroma";
            changelog = "https://github.com/chroma-core/chroma/releases/tag/${version}";
            license = licenses.asl20;
            maintainers = with maintainers; [ fab ];
            mainProgram = "chroma";
            broken = pkgs.stdenv.hostPlatform.isLinux && pkgs.stdenv.hostPlatform.isAarch64;
          };
        };

        pkgVersion = "0.5.6";

        vectorcode = pkgs.python312Packages.buildPythonApplication rec {
          pname = "vectorcode";
          version = pkgVersion;
          format = "pyproject";

          src = self;

          nativeBuildInputs = with pkgs; [
            python312Packages.pdm-backend
            makeWrapper
            installShellFiles
          ];

          propagatedBuildInputs = with pkgs.python312Packages; [
            chromadb
            httpx
            numpy
            pathspec
            psutil
            pygments
            sentence-transformers
            shtab
            tabulate
            transformers
            tree-sitter
            tree-sitter-language-pack
            google-api-python-client
            colorlog
            json5
            lsprotocol
            pygls
          ];

          optional-dependencies = with pkgs.python312Packages; {
            intel = [
              openvino
              optimum
            ];
            legacy = [
              numpy
              torch
              transformers
            ];
            lsp = [
              lsprotocol
              pygls
            ];
            mcp = [
              mcp
              pydantic
            ];
          };

          pythonImportsCheck = [ "vectorcode" ];

          nativeCheckInputs = with pkgs.python312Packages; [
            mcp
            pygls
            pytestCheckHook
            pytest-asyncio
          ];
          versionCheckProgramArg = "version";

          postFixup = ''
            # INFO: Workaround for subprocesses
            wrapProgram $out/bin/vectorcode \
              --prefix PYTHONPATH : "$PYTHONPATH"
            wrapProgram $out/bin/vectorcode-server \
              --prefix PYTHONPATH : "$PYTHONPATH"
            wrapProgram $out/bin/vectorcode-mcp-server \
              --prefix PYTHONPATH : "$PYTHONPATH"
          '';

          postInstall = ''
            installShellCompletion --cmd vectorcode \
              --bash <($out/bin/vectorcode --print-completion bash) \
              --zsh <($out/bin/vectorcode --print-completion zsh)
            installShellCompletion --cmd vectorcode-server \
              --bash <($out/bin/vectorcode --print-completion bash) \
              --zsh <($out/bin/vectorcode --print-completion zsh)
            installShellCompletion --cmd vectorcode-mcp-server \
              --bash <($out/bin/vectorcode --print-completion bash) \
              --zsh <($out/bin/vectorcode --print-completion zsh)
          '';

          disabledTests = [
            # Require internet access
            "test_get_embedding_function"
            "test_get_embedding_function_fallback"
            "test_reranker"
            "test_common"
          ];

          meta = {
            description = "Code repository indexing tool to supercharge your LLM experience";
            homepage = "https://github.com/Davidyz/VectorCode";
            changelog = "https://github.com/Davidyz/VectorCode/releases/tag/${version}";
            license = pkgs.lib.licenses.mit;
            mainProgram = "vectorcode";
          };
        };
      in
      {
        packages = {
          default = vectorcode;

          vimPlugin = pkgs.vimUtils.buildVimPlugin {
            pname = "VectorCode";
            version = pkgVersion;
            src = self;
            dependencies = [
              pkgs.vimPlugins.plenary-nvim
            ];
            patches = pkgs.replaceVars ./nix/vim-plugin.patch {
              inherit vectorcode;
            };
          };
        };
      }
    );
}
