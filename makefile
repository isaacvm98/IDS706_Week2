install:
	pip install --upgrade pip &&\
		pip install -r requirements.txt

format:
	black *.py

lint:
	flake8 basic_data_analysis.py

test:
	python -m pytest -vv --cov=basic_data_analysis test_*.py

run:
	python basic_data_analysis.py

notebook:
	jupyter lab --ip=0.0.0.0 --port=8888 --no-browser --allow-root

clean:
	rm -rf __pycache__ .pytest_cache .coverage
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete

docker-build:
	docker build -t mexican-stock-analysis .

docker-run:
	docker run --rm -v $(PWD):/app mexican-stock-analysis

all: install format lint test

.PHONY: install format lint test run notebook clean docker-build docker-run all