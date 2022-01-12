import requests

url_items = 'https://dsr02ie4h4.execute-api.us-east-2.amazonaws.com/post_stage/test'
headers = {'x-api-key': "nES4cK9iq69m18dEC1VBq88Ah7zWi2ks1cA4vpnA"}
item_data = {
    'key1': 'ほんげー',
    'key2': 'zunba',
    'key3': "ふんぬらば"
}

r_post = requests.post(url_items, headers=headers, json=item_data)
print(r_post.status_code)
print(r_post.content)
