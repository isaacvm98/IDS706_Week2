install:
	pip install --upgrade pip &&\
		pip install -r requirements.txt

format:
	black *.py

format-check:
	black --check --diff *.py

lint:
	# Stop build if there are Python syntax errors or undefined names
	flake8 . --count --select=E9,F63,F7,F82 --show-source --statistics
	# Check specific files with relaxed settings for line length and Black compatibility
	flake8 basic_data_analysis.py test_stock_analysis.py --count --max-complexity=10 --max-line-length=100 --statistics --ignore=E203,W503,E501

lint-strict:
	flake8 basic_data_analysis.py test_stock_analysis.py --max-line-length=100 --ignore=E203,W503

test:
	python -m pytest -vv --cov=basic_data_analysis --cov-report=term --cov-report=html test_stock_analysis.py

test-coverage:
	python -m pytest -vv --cov=basic_data_analysis --cov-report=xml --cov-report=term --cov-report=html test_stock_analysis.py

run:
	python basic_data_analysis.py

notebook:
	jupyter lab --ip=0.0.0.0 --port=8888 --no-browser --allow-root

clean:
	rm -rf __pycache__ .pytest_cache .coverage htmlcov coverage.xml
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	rm -rf visualizations/*.png

docker-build:
	docker build -t mexican-stock-analysis .

docker-run:
	docker run --rm -v $(PWD):/app mexican-stock-analysis

docker-test:
	docker run --rm -v $(PWD):/app mexican-stock-analysis python -m pytest -vv test_stock_analysis.py

docker-lint:
	docker run --rm -v $(PWD):/app mexican-stock-analysis sh -c "black --check *.py && flake8 basic_data_analysis.py test_stock_analysis.py --max-line-length=100 --ignore=E203,W503"

# Quality checks - run all formatting, linting, and testing
quality: format-check lint test

# CI simulation - run the same checks as CI
ci: install quality

all: install format lint test

.PHONY: install format format-check lint lint-strict test test-coverage run notebook clean docker-build docker-run docker-test docker-lint quality ci all