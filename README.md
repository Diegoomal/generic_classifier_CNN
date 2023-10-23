# Generic Classifier Image (CNN)

- **docker**
- **python**
- **tensorflow**
- **GPU (NVIDIA)**

## Dataset creation

Create the dataset where each folder in the directory will represent a class of the object for recognition, and within it, the JSON file named "metadata" will contain the information about this object. After creation, host the dataset on your preferred version control system.

## Steps to run

### Project compile

 ```
docker image build -t img_gcicnn .
```

### Run project

 ```
docker run --name container_gcicnn --gpus all img_gcicnn
```