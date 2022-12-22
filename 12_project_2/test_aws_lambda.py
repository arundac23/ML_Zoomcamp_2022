import requests

url = ' https://buknx4xya8.execute-api.us-east-2.amazonaws.com/test'

data = {'url': 'https://upload.wikimedia.org/wikipedia/commons/d/d4/Upsala_Glacier_3.jpg'}

result = requests.post(url, json=data).json()
print(result)
