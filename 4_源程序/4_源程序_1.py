import requests

response = requests.get("http://183.175.12.27:8004/step_01", proxies={"http": None, "https": None})
print(response.text)