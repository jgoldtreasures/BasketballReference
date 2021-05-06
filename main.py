import re
from collections import Counter

import lxml
import requests
import lxml.html as lh
import pandas as pd
from bs4 import BeautifulSoup
from string import ascii_lowercase
import numpy as np
import urllib3
from lxml import etree

from sklearn.model_selection import train_test_split, cross_val_predict
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, precision_score


def get_stats(url):
    #     page = requests.get(url)
    #     doc = lh.fromstring(page.content)
    #     tr_elements = doc.xpath('//tr')
    #
    #     col = []
    #     i = 0
    #
    #     for t in tr_elements[0]:
    #         i += 1
    #         name = t.text_content()
    #         col.append((name, []))
    #
    #     for j in range(1, len(tr_elements)):
    #         T = tr_elements[j]
    #
    #         i = 0
    #         for t in T.iterchildren():
    #             data = t.text_content()
    #             if i > 0:
    #                 try:
    #                     data = int(data)
    #                 except:
    #                     pass
    #             col[i][1].append(data)
    #             i += 1
    #     Dict = {title: column for (title, column) in col}
    #     print(Dict)
    #     df = pd.DataFrame(Dict)

    html = requests.get(url).content
    df_list = pd.read_html(html)
    df = df_list[-1]

    return df.loc[df['Season'] == 'Career']


def get_stats1(url):
    # print(url)
    html = requests.get(url).content
    try:
        df_list = pd.read_html(html)
        df = df_list[-1]
        # print(df)
        # if '2020-21' in df['Season']:
        #     df['']
        # print((df[df['Season'] == 'Career'].index[0] - 1))
        last_yr = df.at[df[df['Season'] == 'Career'].index[0] - 1, 'Season']
        in_el = ['2016-17', '2017-18', '2018-19', '2019-20', '2020-21']
        if last_yr in in_el:
            df['eligible'] = 0
        else:
            df['eligible'] = 1
        df = df.loc[df['Season'] == 'Career']
        df = df.reset_index()
        if int(df['G']) >= 400:
            grab = requests.get(url)
            soup = BeautifulSoup(grab.content, 'lxml')

            table = soup.find(id='bling')

            arr = ["All Star", "MVP", "Def. POY", "All-NBA", "All-Defensive", "Sixth Man", "ROY", "Scoring Champ", "AST Champ",
                   "TRB Champ", "STL Champ", "BLK Champ", "NBA Champ", 'Hall of Fame']
            Dict = {k: v for v, k in enumerate(arr)}

            awards = np.zeros(len(arr))
            if table is not None:
                for entry in table.find_all('a'):
                    entry = str(entry)
                    if entry == '<a>Hall of Fame</a>':
                        awards[len(awards) - 1] = 1
                    val = re.findall(
                        "(?:(\d+)x|\d+-\d+) (All Star|MVP|Def. POY|All-NBA|All-Defensive|Sixth Man|ROY|Scoring Champ|AST Champ|TRB Champ|STL Champ|BLK Champ|NBA Champ)",
                        entry)
                    if len(val) > 0:
                        tup = val[0]
                        num = int(tup[0]) if tup[0] != '' else 1
                        awards[Dict[tup[1]]] = num

            df2 = pd.DataFrame([awards], columns=arr)
            df = df.join(df2)

            table2 = soup.find(id='all_leaderboard')
            val = re.findall("<a href=\"/leaders/hof_prob.html\">(.*\S.*)%<", str(table2))
            # print(val)
            if len(val) == 1:
                df['hof_prob'] = val[0]

            return df
        return None
    except etree.ParserError:
        return None


def get_urls():
    names = []
    links1 = []
    # for c in ['a']:
    for c in ascii_lowercase:
        urls = 'https://www.basketball-reference.com/players/' + c + '/'
        grab = requests.get(urls)
        soup = BeautifulSoup(grab.content, 'lxml')

        table = soup.find('table')

        trs = table.find_all('tr')
        links = []
        for tr in trs:
            if tr.find("a") is not None:
                links.append(tr.find("a"))
        name = [tr.text for tr in links]
        link = ['https://www.basketball-reference.com' + tr.get('href') for tr in links]
        names.extend(name)
        links1.extend(link)
    return names, links1


def pretty_print(matrix):
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    pd.set_option('display.max_colwidth', None)
    print(pd.DataFrame(matrix))


def get_all_stats():
    names, links = get_urls()

    tup = [(names[i], links[i]) for i in range(0, len(names))]

    df = pd.DataFrame()
    for name, link in tup:
        stats = get_stats1(link)
        if stats is not None:
            stats.insert(0, 'Player', name)
            df = df.append(stats)

    # df = df.drop(columns=['Age', 'Tm', 'Pos'])
    df.to_csv('data/career_stats9.csv', index=False)


def remove_nan():
    df = pd.read_csv('data/career_stats9.csv')
    # df = df.drop(columns=['index'])
    # print(df[df['Lg'] == 'NBA'])
    df = df[df['Lg'] == 'NBA']
    df = df.drop(columns=['Age', 'Tm', 'Pos'])
    df = df.dropna()
    df = df.filter(['Player', 'TRB', 'AST', 'STL', 'BLK', 'PTS', "All Star", "MVP", "Def. POY", "All-NBA", "All-Defensive",
                    "Sixth Man", "ROY", "Scoring Champ", "AST Champ",
                    "TRB Champ", "STL Champ", "BLK Champ", "NBA Champ", 'Hall of Fame', 'eligible', 'hof_prob'])
    df.to_csv('data/career_stats10.csv', index=False)

    pretty_print(df)


def main():
    get_all_stats()
    # df = pd.read_csv('data/career_stats1.csv')
    # df = pd.read_csv('data/career_stats1.csv')
    # df = df.filter(['Player', 'TRB', 'AST', 'STL', 'BLK', 'PTS'])
    # df.to_csv('data/career_stats2.csv', index=False)
    #
    # # players = list(df['Player'])
    # # index = list(df.index)
    # #
    # # print(len(players))
    # # print(len(index))
    # # zip_iter = zip(players, index)
    # # Dict = dict(zip_iter)
    # #
    # # print(Counter(players))
    # # print(len(set(players)))
    # # asg = np.zeros(len(Dict))
    #
    # url = 'https://www.basketball-reference.com/awards/all_star_by_player.html'
    # html = requests.get(url).content
    # df_list = pd.read_html(html)
    # df2 = df_list[-1]
    # print(df2)
    #
    # for ind, row in df2.iterrows():
    #     name = row['Player']

    # xtrain, xtest, ytrain, ytest = train_test_split(df.drop(columns=['Player', 'Hall of Fame'], axis=1), df['Hall of Fame'], test_size=0.20)
    # logmodel = LogisticRegression()
    # logmodel.fit(xtrain, ytrain)
    # predictions = logmodel.predict(xtest)
    #
    # print(classification_report(ytest, predictions))
    # print(confusion_matrix(ytest, predictions))

    # logreg = LogisticRegression()
    # ypred = cross_val_predict(logreg, df.drop(columns=['Player', 'Hall of Fame'], axis=1), df['Hall of Fame'], cv=5)
    # print(confusion_matrix(df['Hall of Fame'], ypred))
    # print(precision_score(df['Hall of Fame'], ypred))


# main()
remove_nan()
