# Generic Classifier CNN

- **docker**
- **python**
- **tensorflow**
- **GPU (NVIDIA)**

## Steps to run

### Project compile

 ```
docker image build -t img_gccnn_app .
```

### Run project

 ```
docker run --name container_gccnn_app --gpus all img_gccnn_app
```