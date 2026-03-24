build:
	docker build --tag pwdiow-lab2 .

jupyter:
	docker run --rm -v $(shell pwd):/app -p 8888:8888 pwdiow-lab2 jupyter notebook --ip=0.0.0.0 --allow-root --no-browser

format:
	docker run --rm -v $(shell pwd):/app pwdiow-lab2 ruff format