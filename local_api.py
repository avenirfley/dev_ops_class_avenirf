"""Exercise the locally running FastAPI application."""

import requests

BASE_URL = "http://127.0.0.1:8000"


def main():
    """Send the required GET and POST requests and display their results."""
    response = requests.get(BASE_URL, timeout=10)
    print(f"GET status: {response.status_code}")
    print(f"GET message: {response.json()['message']}")

    data = {
        "age": 37,
        "workclass": "Private",
        "fnlgt": 178356,
        "education": "HS-grad",
        "education-num": 10,
        "marital-status": "Married-civ-spouse",
        "occupation": "Prof-specialty",
        "relationship": "Husband",
        "race": "White",
        "sex": "Male",
        "capital-gain": 0,
        "capital-loss": 0,
        "hours-per-week": 40,
        "native-country": "United-States",
    }
    response = requests.post(f"{BASE_URL}/data/", json=data, timeout=10)
    print(f"POST status: {response.status_code}")
    print(f"POST result: {response.json()['result']}")


if __name__ == "__main__":
    main()