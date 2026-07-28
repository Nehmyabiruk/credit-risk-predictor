.PHONY: install preprocess train evaluate predict clean all help

help:
	@echo "Available targets:"
	@echo "  make install     - install dependencies"
	@echo "  make preprocess  - clean + feature engineer"
	@echo "  make train       - train + hyperparameter search"
	@echo "  make evaluate    - evaluate saved model on hold-out"
	@echo "  make all         - preprocess → train → evaluate"
	@echo "  make clean       - remove processed data, models, reports"

install:
	pip install -r requirements.txt

preprocess:
	python -m src.preprocess

train:
	python -m src.train

evaluate:
	python -m src.evaluate

all: preprocess train evaluate

clean:
	rm -f data/processed/*.csv
	rm -f models/*.pkl
	rm -f reports/*.png reports/*.json
	@echo "Cleaned processed data, models and reports."
