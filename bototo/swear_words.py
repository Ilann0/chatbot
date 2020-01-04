import requests

URL = 'https://www.purgomalum.com/service/containsprofanity'

def has_swear_words(user_input):
    req = requests.get(URL, params=[('text', user_input)])
    
    return req.json()