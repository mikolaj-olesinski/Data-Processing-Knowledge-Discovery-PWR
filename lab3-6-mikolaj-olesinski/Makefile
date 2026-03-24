build:
	docker build --tag pwdiow-lab3 .

setup-remote:
	docker run --rm --network lab3-6-mikolaj-olesinski_default -v $(shell pwd):/app pwdiow-lab3 bash scripts/shell/setup_dvc_remote.sh

shell:
	docker run --rm -it --network lab3-6-mikolaj-olesinski_default -v $(shell pwd):/app pwdiow-lab3 bash