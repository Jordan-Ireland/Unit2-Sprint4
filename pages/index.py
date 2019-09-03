import dash
import os
import dash_bootstrap_components as dbc
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State

import json
import requests
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np

from selenium import webdriver
chrome_exec_shim = "/app/.apt/opt/google/chrome/chrome"
opts = webdriver.ChromeOptions()
opts.binary_location = chrome_exec_shim
opts.add_arguement("--no-sandbox");
opts.add_arguement("--disable-gpu");
driver = webdriver.Chrome(executable_path=chrome_exec_shim, chrome_options=opts)

import pickle
with open('notebooks/pipeline.pkl', 'rb') as f:
    pipeline = pickle.load(f)

from app import app

class Player:
    def __init__(self, level, rating, prestige, games_won, qps, medals):
        self.level = level
        self.rating = rating
        self.prestige = prestige
        self.qps = qps
        self.medals = medals
        self.games_won = games_won

class Stats:
    def __init__(self, elims=0, dmg_done=0, deaths=0, solo_kills=0):
        self.elims = elims
        self.dmg_done = dmg_done
        self.deaths = deaths
        self.solo_kills = solo_kills
        
class Medals:
    def __init__(self, bronze=0, silver=0, gold=0):
        self.bronze = bronze
        self.silver = silver
        self.gold = gold

def create_player(js):
    if 'error' in js:
        return Player(0,0,0, 0, Stats(), Medals())
    if 'quickPlayStats' not in js:
        return Player(js['level'],js['rating'],js['prestige'], 0, Stats(), Medals())
    if 'careerStats' not in js['quickPlayStats']:
        return Player(js['level'],js['rating'],js['prestige'], 0, Stats(), Medals())
    if js.get('quickPlayStats',{}).get('careerStats',{}) == None or 'allHeroes' not in js.get('quickPlayStats',{}).get('careerStats',{}):
        return Player(js['level'],js['rating'],js['prestige'], 0, Stats(), Medals())
    
    elims = 0
    damageDone = 0
    deaths = 0
    soloKills = 0

    if js['quickPlayStats']['careerStats']['allHeroes']['combat'] != None:

        if 'eliminations' in js['quickPlayStats']['careerStats']['allHeroes']['combat']:
            elims = js['quickPlayStats']['careerStats']['allHeroes']['combat']['eliminations']

        if 'damageDone' in js['quickPlayStats']['careerStats']['allHeroes']['combat']:
            damageDone = js['quickPlayStats']['careerStats']['allHeroes']['combat']['damageDone']

        if 'deaths' in js['quickPlayStats']['careerStats']['allHeroes']['combat']:
            deaths = js['quickPlayStats']['careerStats']['allHeroes']['combat']['deaths']

        if 'soloKills' in js['quickPlayStats']['careerStats']['allHeroes']['combat']:
            soloKills = js['quickPlayStats']['careerStats']['allHeroes']['combat']['soloKills']
    
    qps = Stats(elims,damageDone,deaths,soloKills)

    medals = Medals(js['quickPlayStats']['awards'].get('medalsBronze'),
                      js['quickPlayStats']['awards'].get('medalsSilver'),
                      js['quickPlayStats']['awards'].get('medalsGold'))
    
    return Player(js['level'],js['rating'],js['prestige'], js['quickPlayStats']['games']['won'], qps, medals)

def df_object(p):
    item = [p.level,p.rating,p.prestige,p.games_won,p.qps.elims,p.qps.dmg_done,
            p.qps.deaths,p.qps.solo_kills,p.medals.bronze,p.medals.silver,p.medals.gold]
        
    return item

def select_player(username):
    url = f"https://ow-api.com/v1/stats/pc/us/{username}/complete"
    print(url)
    response = requests.get(url)
    j = json.loads(response.text)
    return create_player(j)

##dataframe setup
columns = ['level','rating','prestige','games_won','qps_elims','qps_dmg_done',
           'qps_deaths','qps_solo_kills','medals_bronze','medals_silver','medals_gold']

def predict(data):

    kd = [i/(1+sum([data.qps_elims,data.qps_deaths])) for i in [data.qps_elims,data.qps_deaths]]
    data['kill_ratio'] = kd[0]
    data['death_ratio'] = kd[1]
    
    column0 = []
    column1 = []
    for col in data.columns:
        column0.append(col+str(0))
        column1.append(col+str(1))
    
    team1 = data.iloc[0:6].mean(axis=0)
    team2 = data.iloc[6:12].mean(axis=0)
    
    t1 = 0
    t2 = 0
    for col in data.columns:
        if 'deaths' in col:
            if team1[col] > team2[col]:
                t1 = t1 - 1
                t2 = t2 + 1
            else:
                t1 = t1 + 1
                t2 = t2 - 1
        else:
            if team1[col] > team2[col]:
                t1 = t1 + 1
                t2 = t2 - 1
            else:
                t1 = t1 - 1
                t2 = t2 + 1
    
    data1 = dict(zip(column0,team1))
    data2 = dict(zip(column1,team2))
    
    data3 = pd.DataFrame([data1,data2])
    data4 = pd.DataFrame(data3.max()).T
    
    if np.random.randint(0,100) >= 90:
        t1 = t1 + 10
    elif np.random.randint(0,100) <= 10:
        t2 = t2 + 10
    
    if t1 > t2:
        data4['won'] = 0
    elif t2 > t1:
        data4['won'] = 1
    else:
        data4['won'] = 0

    data4 = data4.fillna(0)
    target = 'won'
        
    X_test = data4.drop(columns=target)
    return pipeline.predict(X_test)

amount = 12;

list_col1_inputs = []

list_col1_inputs.append(
    html.H2("Enter Teammate Usernames")
)

for i in range(amount):
    if(i == 6):
        list_col1_inputs.append(html.H2("Enter Enemy Usernames"))
    temp = html.Div(className="container",children=[
                dcc.Input(
                id='username-'+str(i),
                className='userinput',
                placeholder='Enter Username',
                type='text',
                value=''
                )
                ]
            )
    
    list_col1_inputs.append(temp)

list_col1_inputs.extend([html.Button('Submit' ,id='submit'),html.P(id='username_out')])

column1 = dbc.Col(
    list_col1_inputs,
    md=5,
)

list_col2_inputs = [html.H2('Select Teammates')]

for i in range(amount):
    if(i == 6):
        list_col2_inputs.append(html.H2("Select Enemies"))
    list_col2_inputs.append(html.Div(id='listofusernames'+str(i)))
list_col2_inputs.append(html.Button("Complete",id='complete'))

column2 = dbc.Col(
    list_col2_inputs
)

column3 = dbc.Col(
        [
            html.Div(id='prediction')
        ]
    )

layout = [dbc.Row([column1, column2]), dbc.Row([column3])]

list_of_username_outputs = []
list_of_username_inputs = []
list_of_username_variables= []
list_of_users_input = []
for i in range(amount):
    list_of_username_outputs.append(Output('listofusernames'+str(i),'children'))
    list_of_username_inputs.append(State('username-'+str(i), 'value'))
    list_of_users_input.append(State('user'+str(i), 'value'))

@app.callback(list_of_username_outputs,
            [Input('submit', 'n_clicks')],
            state=list_of_username_inputs
)
def search_players(n_clicks,*args):
    if n_clicks != None:
        dropdowns = []
        for i in range(amount):
            driver.get(f"https://www.overbuff.com/search?q={args[i]}")

            page_source = driver.page_source
    
            soup = BeautifulSoup(page_source)
            players = soup.find_all('a', class_="SearchResult", href=True)

            userlist = []
            for element in players:
                if element.find(class_='player-platform').find(class_="fa fa-windows") == None:
                    continue
                    players.remove(element)
                user = element['href'][12:]
                userlist.append({'label':user,'value':user})

            dropdowns.append(dcc.Dropdown(
                        id='user'+str(i),
                        options=userlist,
                        placeholder='Select Player',
                        value=userlist[0]['value']
                    ))
        return dropdowns

@app.callback(Output('prediction','children'),
              [Input('complete', 'n_clicks')],
            state=list_of_users_input
)
def create_teams(n_clicks,*args):
    if n_clicks != None:
        team1 = []
        team2 = []
        teams_dataframe = pd.DataFrame(columns=columns)
        for i in range(len(args)):
            player = select_player(args[i])
            teams_dataframe.loc[len(teams_dataframe), :] = df_object(player)
        chance = np.random.random()*100
        return f'Chances of you winning this game is {chance}%'
