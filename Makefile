PYTHON ?= python3

.PHONY: install run test

install:
$(PYTHON) -m pip install -e .[dev]

run:
$(PYTHON) -m streamdeck_tui

test:
$(PYTHON) -m pytest
