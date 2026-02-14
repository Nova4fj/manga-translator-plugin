.PHONY: test lint coverage ci clean

test:
	python -m pytest manga_translator/tests/ -v

lint:
	ruff check manga_translator/

coverage:
	python -m pytest manga_translator/tests/ --cov=manga_translator --cov-report=term-missing --cov-report=html

ci: lint test coverage

clean:
	rm -rf .pytest_cache htmlcov .coverage __pycache__
