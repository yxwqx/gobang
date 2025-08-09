import requests

params = {
    "name": "name",
    "student_number": "1234"
}
response = requests.get("http://183.175.12.27:8004/step_02", params=params)
print(response.json())