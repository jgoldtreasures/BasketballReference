import re
import requests
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np

url = 'https://www.basketball-reference.com/players/a/abdulka01.html'


html = requests.get(url).content
df_list = pd.read_html(html)
df = df_list[-1]
df = df.loc[df['Season'] == 'Career']
df = df.reset_index()


grab = requests.get(url)
soup = BeautifulSoup(grab.content, 'lxml')

table = soup.find(id='bling')

arr = ["All Star", "MVP", "DPOY", "All-NBA", "All-Defensive", "Sixth Man", "ROY", "Scoring Champ", "AST Champ",
       "TRB Champ", "STL Champ", "BLK Champ", "NBA Champ", 'Hall of Fame']
Dict = {k: v for v, k in enumerate(arr)}

awards = np.zeros(len(arr))
if table is not None:
    for entry in table.find_all('a'):
        entry = str(entry)
        if entry == '<a>Hall of Fame</a>':
            awards[len(awards) - 1] = 1
        val = re.findall(
            "(?:(\d+)x|\d+-\d+) (All Star|MVP|DPOY|All-NBA|All-Defensive|Sixth Man|ROY|Scoring Champ|AST Champ|TRB Champ|STL Champ|BLK Champ|NBA Champ)",
            entry)
        if len(val) > 0:
            tup = val[0]
            num = int(tup[0]) if tup[0] != '' else 1
            awards[Dict[tup[1]]] = num

df2 = pd.DataFrame([awards], columns=arr)
df = df.join(df2)

print(df)

