import pandas as pd
import numpy as np
import requests
import random
import scipy.stats as ss
import pdb
from sklearn import preprocessing
from collections import defaultdict


# functions
def get_position(player_id):
    """Use the MLB API to look up a players primary position"""
    url = ("http://lookup-service-prod.mlb.com/json/named.player_info.bam?"
           "sport_code='mlb'&player_id='{}'").format(player_id)

    r = requests.get(url)

    pos = r.json()[
        'player_info']['queryResults']['row']['primary_position_txt']

    return(pos)

def load_projections(roto_stats):
    """Load player projections"""
    # set up data
    idmap = pd.read_csv('https://www.smartfantasybaseball.com/PLAYERIDMAPCSV',
                      usecols=['IDFANGRAPHS', 'MLBID']
    )
    df = pd.read_csv('steamer_batter.csv')
    df = df.merge(idmap, left_on='playerid', right_on='IDFANGRAPHS')
    df['position'] = df.apply(lambda x: get_position(x['MLBID']), axis=1)

    
    scaler = preprocessing.MinMaxScaler(feature_range=(0, 1))
    for rs in roto_stats:
        df['scaled_' + rs] = scaler.fit_transform(df[[rs]].values)
    
    return(df)

def init_weights(total, num):
    weights = []
    remaining = total
    for i in list(range(num)):
        if i < num - 1:
            new = random.randint(0, remaining)
            weights.append(new)
            remaining -= new
        else:
            weights.append(remaining)

    return(weights)



# classes
class Player(object):

    def __init__(self, name, mlbid, **kwargs):
        self.name = name
        self.mlbid = mlbid
        if kwargs:
            self.__dict__.update(kwargs)            
        
    def __str__(self):
        return(self.name)

    __repr__ = __str__   


class ValueNet(object):

    def __init__(self, weights):
        self.weights = weights

        

class Team(object):

    def __init__(self, name, valnet, roto_stats):
        self.name = name
        self.valnet = valnet
        self.roto_stats = roto_stats
        self.roster = []

    def choose(self, players):

        # refresh current roster stats
        self.get_totals()    

        # assign value to each remaining player
        pvals = [np.dot(np.array([p.s_R, p.s_HR, p.s_RBI, p.s_SB]),
                        self.valnet.weights) for p in players]
        

        choice = players[pvals.index(max(pvals))]

        return(choice)

    def add_player(self, player):
        self.roster.append(player)

    def get_totals(self):
        for rs in self.roto_stats:
            self.__dict__[rs] = sum([player.__dict__[rs]
                                     for player in self.roster])

            
    def __str__(self):
        return(self.name)
    __repr__ = __str__


class League(object):

    def __init__(self, teams, players, roto_stats):
        self.teams = teams
        self.players = players
        self.roto_stats = roto_stats

    def snake(self, num_rounds):
        for dround in list(range(num_rounds)):
            order = self.teams
            if dround % 2 == 1:
                order = self.teams[::-1]

            for team in order:
                choice = team.choose(self.players)
                print("Round: {} -- {} -- {}".format(
                    dround, team, choice))
                team.add_player(choice)
                self.players.remove(choice)
                
    def roto(self):
        for team in self.teams:
            team.get_totals()

        # calculate roto points
        points = {}
        for rs in self.roto_stats:
            points[rs] = ss.rankdata([i.__dict__[rs] for i in league.teams])

        for ind, team in enumerate(self.teams):
            team.fitness = 0
            for rs in points.keys():
                team.fitness += points[rs][ind]
            
        standings = sorted(self.teams, key=lambda x: x.fitness, reverse=True)

        for i in standings:
            print('{}: {}'.format(i, i.fitness))
            


### To Run

# choose stats and load projections
roto_stats = ['R', 'HR', 'RBI', 'SB']
projections = load_projections(roto_stats)

# generate players
players = []
for index, row in projections.iterrows():
    stats = {}
    for rs in roto_stats:
        stats[rs] = row[rs]
        stats['s_'+rs] = row['scaled_'+rs]
    players.append(Player(mlbid=row['MLBID'],
                          name=row['Name'],
                          **stats
                          ))

# generate teams
teams = []
for i in list(range(10)):
    vn = ValueNet(init_weights(100, 4))
    teams.append(Team('Team: {}'.format(str(i)), vn, roto_stats))
    
# generate the league
league = League(teams, players, roto_stats)

# draft
league.snake(10)

# results
league.roto()
    

