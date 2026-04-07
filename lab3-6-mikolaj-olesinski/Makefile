build:
	docker build --tag pwdiow-lab3 .

setup-remote:
	docker run --rm --network lab3-6-mikolaj-olesinski_default -v $(shell pwd):/app pwdiow-lab3 bash scripts/shell/setup_dvc_remote.sh

shell:
	docker run --rm -it --network lab3-6-mikolaj-olesinski_default -e MLFLOW_TRACKING_URI=http://mlflow:5000 -v $(shell pwd):/app -v gensim-data:/root/gensim-data pwdiow-lab3 bash