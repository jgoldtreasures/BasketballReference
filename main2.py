import requests
from bs4 import BeautifulSoup
import pandas as pd


url = 'https://www.basketball-reference.com/players/a/abdulka01.html'
html = requests.get(url).content
df_list = pd.read_html(html)
df = df_list[-1]
print(df)
