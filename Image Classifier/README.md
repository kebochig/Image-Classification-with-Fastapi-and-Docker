# DEFECTIVE BOX IMAGE CLASSIFIER

In this project, a model was built to classify defective and non defective boxes by applying transfer learning on MobileNwtV2, an off-the-shelf CNN model. Image Classifier 1.ipynb describes the training and testing process, while Image Classifier2.ipynb allows for predicting an image either through a local file path or URL. The model is then exposed through an API for consumption. FastAPI was used to expose the model through a UI interface. Finally, Docker is used to contenarize and deploy the application

## Environment Setup

Run docker-compose build in the directory to build the environment, after which, run docker-compose build to start the app. Once the app is up in docker, the APIs will be ready to be called. You can interact with the APIs through the UI (http://0.0.0.0:8000/docs)

```bash
docker-compose build
```
```bash
docker-compose up
```

## Usage

Once the app is started, the documentation can be viewed using the Swagger UI url:
 http://{HOST}:{PORT}/docs  for example  http://0.0.0.0:8000/docs

