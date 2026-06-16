import requests

url = "http://localhost:11434/api/generate"

def ask_model(prompt):
    data = {
        "model":"mistral",
        "prompt":prompt,
        "stream":False
    }

    res = requests.post(url,json=data)
    return res.json()["response"]

print(ask_model("Write a GitHub Actions YAML pipeline that runs on push to main, sets up Python 3.11, installs requirements, runs pytest with coverage, builds a Docker image with the commit SHA as the tag, and pushes it to Docker Hub."))