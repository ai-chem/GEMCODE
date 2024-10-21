### Build in container:
``` bash
docker build -t llama_local:latest -f pytorch.Dockerfile .

docker run --gpus all --rm -it -p 8888:8888 llama_local jupyter lab --no-browser --port 8888 --ServerApp.token='' --ip='*' --allow-root
```

#### ! Don't forget to mount a volume with scripts and datasets using [-v arg](https://docs.docker.com/reference/cli/docker/container/run/#volume) on docker run.