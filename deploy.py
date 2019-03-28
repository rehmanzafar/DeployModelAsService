from azureml.core import Workspace
from azureml.core.model import Model
from azureml.core.webservice import AciWebservice
from azureml.core.webservice import Webservice
from azureml.core.image import ContainerImage
from azureml.core.conda_dependencies import CondaDependencies

ws = Workspace.from_config()

myenv = CondaDependencies()
myenv.add_pip_package("tensorflow==1.12.0")
myenv.add_pip_package("keras==2.2.4")
myenv.add_pip_package("numpy")

with open("dlenv.yml", "w") as f:
    f.write(myenv.serialize_to_string())

model = Model.register(model_path = "tf_mnist_model.h5",
                       model_name = "tf_mnist_model",
                       tags = {"key": "1"},
                       description = "MNIST Prediction",
                       workspace = ws)

aciconfig = AciWebservice.deploy_configuration(cpu_cores=1,
                                               memory_gb=1,
                                               tags={"data": "MNIST", "method": "tf"},
                                               description='Predict MNIST with tf')
# configure the image
image_config = ContainerImage.image_configuration(execution_script="score.py",
                                                  runtime="python",
                                                  conda_file="dlenv.yml")

service = Webservice.deploy_from_model(workspace=ws,
                                       name='tf-mnist-svc',
                                       deployment_config=aciconfig,
                                       models=[model],
                                       image_config=image_config)

service.wait_for_deployment(show_output=True)

print(service.scoring_uri)
