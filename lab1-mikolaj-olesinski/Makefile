build:
	docker build --tag pdiow .

run_main_1:
	docker run --rm -v $(shell pwd):/app -e PYTHONPATH=/app pdiow python scripts/main_1.py

run_main_2:
	docker run --rm -v $(shell pwd):/app -e PYTHONPATH=/app pdiow python scripts/main_2.py

format:
	docker run --rm -v $(shell pwd):/app pdiow ruff format

run_jupyter:
	docker run --rm -v $(shell pwd):/app -e PYTHONPATH=/app -p 8888:8888 pdiow jupyter notebook --ip=0.0.0.0 --allow-root --no-browser
