## Torchserve-Helm deployment

Convert `.pt` to `.mar` execute:
```bash
torch-model-archiver --model-name [model-name] --version 1.0 --serialized-file ml_app_[ver]/yolo_weights/best.pt --export-path model_store --handler handler.py
```

## Containerize
In both apps run
```bash
docker build --platform linux/amd64 -f Dockerfile -t [tag] .
```

```bash
docker run --rm -it -p 127.0.0.1:8080:8080 -p 127.0.0.1:8081:8081 -p 127.0.0.1:8082:8082 -p 127.0.0.1:7070:7070 -p 127.0.0.1:7071:7071 [tag]
```

## Helm

To install:
```bash
helm install torchserve-app ./torchserve-app
```

To uninstall:
```bash
helm uninstall torchserve-app
```

Workflow:
1. Two images are loaded to minikube cluster (`coco-model`, `wildlife-model`)
2. Helm deploys two models to two pods `kubectl get pods`
3. Ingress reverse proxies port `8081` to point to `/predictions/[model_tag]`
4. Server is accessible via `http://torchserve.local`. Had to map minikube ip to it in `sudo nano /etc/hosts`

## Locust
Activate virtual env
```bash
conda activate torchserve-helm
```

Run locust
```bash
locust
```