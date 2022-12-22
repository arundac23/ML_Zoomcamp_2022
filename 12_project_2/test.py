import requests

url = 'http://localhost:8080/2015-03-31/functions/function/invocations'

data = {"url": "https://upload.wikimedia.org/wikipedia/commons/d/d4/Upsala_Glacier_3.jpg"}

result = requests.post(url, json=data).json()
print(result)
