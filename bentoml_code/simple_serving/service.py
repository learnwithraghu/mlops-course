import bentoml
from bentoml.io import JSON

# Define a simple model
def predict(input_data):
    return {"message": f"Hello, {input_data['name']}!"}

# Create a BentoML service
svc = bentoml.Service("hello_world_service", runners=[])

# Define an API endpoint
@svc.api(input=JSON(), output=JSON())
def say_hello(input_data):
    return predict(input_data)
