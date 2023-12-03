import mlflow

if __name__ == "__main__":
    mlflow.set_tracking_uri("http://localhost:5000")
    
    model_name = "MLProject"
    stage = "Staging"
    
    
    model = mlflow.pyfunc.load(
        model_uri=f"models:/{model_name}/{stage}"
    )
    
    output = model.predict(['model'])
    print(output)
    
