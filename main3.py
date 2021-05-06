import re
import requests
from bs4 import BeautifulSoup
import pandas as pd
import urllib
import urllib.request
import numpy as np

url = 'https://www.basketball-reference.com/players/a/abdulka01.html'
# grab = requests.get(url)
# soup = BeautifulSoup(grab.content, 'lxml')
# # soup = BeautifulSoup(open('https://www.basketball-reference.com/players/a/abdulka01.html'))
# hof = soup.find_all(re.compile('\bKareem\b'))

grab = requests.get(url)
soup = BeautifulSoup(grab.content, 'lxml')

table = soup.find(id='bling')

table1 = np.array(table)

print(table)

arr = ["All Star", "MVP", "DPOY", "All-NBA", "All-Defensive", "Sixth Man", "ROY", "Scoring Champ", "AST Champ",
       "TRB Champ", "STL Champ", "BLK Champ", "NBA Champ", 'Hall of Fame']
Dict = {k: v for v, k in enumerate(arr)}

awards = np.zeros(len(arr))
if table is not None:
    for entry in table.find_all('a'):
        entry = str(entry)
        print(entry)
        if entry == '<a>Hall of Fame</a>':
            awards[len(awards) - 1] = 1
        val = re.findall(
            "(?:(\d+)x|\d+-\d+) (All Star|MVP|DPOY|All-NBA|All-Defensive|Sixth Man|ROY|Scoring Champ|AST Champ|TRB Champ|STL Champ|BLK Champ|NBA Champ)",
            entry)
        if len(val) > 0:
            tup = val[0]
            num = int(tup[0]) if tup[0] != '' else 1
            awards[Dict[tup[1]]] = num

print(awards)
