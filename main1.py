import requests
from bs4 import BeautifulSoup
import numpy as np

urls = 'https://www.basketball-reference.com/players/a/'
grab = requests.get(urls)
soup = BeautifulSoup(grab.content, 'lxml')

table = soup.find('table')

trs = table.find_all('tr')
links = []
for tr in trs:
    if tr.find("a") is not None:
        links.append(tr.find("a"))  ## finds the first link
links1 = [tr.text for tr in links]
links2 = ['https://www.basketball-reference.com' + tr.get('href') for tr in links]

print(links1)
print(links2)
