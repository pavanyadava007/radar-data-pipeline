# Each target mirrors one pipeline stage. `make all` reproduces every result from the raw file.
PY ?= $(shell [ -x .venv/bin/python ] && echo .venv/bin/python || echo python)
RDP = $(PY) -m rdp
RAW = data/raw/data_SAAB_SIRS_77GHz_FMCW.npy

.PHONY: all data ingest dsp quality export train sql report test lint clean-derived docker

all: ingest dsp quality export train sql report

data: $(RAW)
$(RAW):
	bash scripts/download_data.sh data/raw

ingest: $(RAW)
	$(RDP) ingest

dsp:
	$(RDP) dsp

quality:
	$(RDP) quality

export:
	$(RDP) export

train:
	$(RDP) train

sql:
	$(RDP) sql-all

report:
	$(PY) scripts/report.py

test:
	$(PY) -m pytest -q

lint:
	$(PY) -m ruff check .
	$(PY) -m ruff format --check .

clean-derived:
	rm -rf data/derived data/exports data/models data/catalog.sqlite

docker:
	docker build -t rdp:latest .
	docker run --rm rdp:latest
