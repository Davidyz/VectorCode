#!/bin/sh

export EXTRA_COVERAGEPY_ARGS='--append'

make deps

pdm run coverage erase

# chroma 0.6.3
EXTRA_LOCK_ARGS="--group chroma0" make coverage
# default install (chroma 1.x)
make coverage

pdm run coverage report -m
