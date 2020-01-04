import requests

API_KEY = 'nGiInmfELjTiMJkqZgG80ffgINVxK61P'
URL = 'https://api.nytimes.com/svc/topstories/v2/home.json'

def get_current_news():
    req = requests.get(URL, params=[('api-key', API_KEY)])
    
    return req.json()['results'][0]['title']