import pandas as pd
import numpy as np
import requests
import random
import scipy.stats as ss
import pdb
import matplotlib.pyplot as plt
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
    weights = np.array([])
    remaining = total
    for i in list(range(num)):
        if i < num - 1:
            new = random.randint(0, remaining)
            weights = np.append(weights, new)
            remaining -= new
        else:
            weights = np.append(weights, remaining)

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

    def mutate(self):
        # generate a random mutation
        new = np.random.randint(
            -5, 5, size=len(self.weights) - 1)

        # make sure weights sum to 100
        if abs(sum(new)) < 5:
            new = np.append(new, 0 - sum(new))
            # make sure all weights are > 0
            if not ((new + self.weights) > 0).all():
                return(self.mutate())

            return(self.weights + new)
        else:
            return(self.mutate())

    def __str__(self):
        return('Net: {}'.format(', '.join([str(i) for i in self.weights])))

    __repr__ = __str__


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
        self.draft = []

    def snake(self, num_rounds):
        for dround in list(range(num_rounds)):
            order = self.teams
            if dround % 2 == 1:
                order = self.teams[::-1]

            for team in order:
                choice = team.choose(self.players)

                self.draft.append("Round: {} -- {} -- {}".format(
                    dround, team, choice))
                team.add_player(choice)
                self.players.remove(choice)

    def roto(self):
        for team in self.teams:
            team.get_totals()

        # calculate roto points
        points = {}
        for rs in self.roto_stats:
            points[rs] = ss.rankdata([i.__dict__[rs] for i in self.teams])

        for ind, team in enumerate(self.teams):
            team.fitness = 0
            for rs in points.keys():
                team.fitness += points[rs][ind]

        standings = sorted(self.teams, key=lambda x: x.fitness, reverse=True)

        return(standings)


def sim(players, roto_stats, new_nets=None):
    # initial sim
    if not new_nets:
        teams = []
        for i in list(range(10)):
            vn = ValueNet(init_weights(100, len(roto_stats)))
            teams.append(Team('Team: {}'.format(str(i)), vn, roto_stats))

    else:
        teams = []
        for i, vn in enumerate(new_nets):
            vn = ValueNet(vn)
            teams.append(Team('Team: {}'.format(str(i)), vn, roto_stats))

    # generate the initial league
    league = League(teams, players.copy(), roto_stats)

    # draft
    league.snake(10)

    # results
    standings = league.roto()

    # evolve
    winners = standings[:3]
    winning_nets = [i.valnet for i in winners]

    # next generation of winners: top 3 + two children of each top3 + rando
    new_nets = []
    for wn in winning_nets:
        new_nets.append(wn.weights)
        for i in list(range(2)):
            new_nets.append(wn.mutate())

    new_nets.append(init_weights(100, 4))

    return(winning_nets[0], new_nets, standings, league.draft)


# To Run

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

# initial sim
winning_nets = []
print('Simming Round 1')
first_winner, new_nets, first_standings, first_draft = sim(players, roto_stats)

winning_nets.append(first_winner)

stop = False
counter = 2
last_winner = first_winner
last_nets = new_nets
while not stop:
    print('Simming Round {}'.format(counter))
    new_winner, new_nets, new_standings, new_draft = sim(
        players, roto_stats, new_nets=last_nets)

    # check stopping conditions
    counter += 1
    if counter > 100:
        stop = True
        print("Stop on count")
    if new_winner in winning_nets:
        stop = True
        print("Stop on back to back winner")
    else:
        # update
        winning_nets.append(new_winner)
        last_nets = random.sample(new_nets, len(new_nets))
        last_winner = new_winner


# plotting
w_R = [vn.weights[0] for vn in winning_nets]
w_HR = [vn.weights[1] for vn in winning_nets]
w_RBI = [vn.weights[2] for vn in winning_nets]
w_SB = [vn.weights[3] for vn in winning_nets]
rounds = np.arange(1, len(w_R)+1, 1)

ax1 = plt.subplot(411)
ax1.set(title="R")
ax1.set_ylim([0, 100])
plt.plot(rounds, w_R)
plt.setp(ax1.get_xticklabels(), visible=False)

ax2 = plt.subplot(412, sharex=ax1)
ax2.set(title="HR")
ax2.set_ylim([0, 100])
plt.plot(rounds, w_HR)
plt.setp(ax2.get_xticklabels(), visible=False)

ax3 = plt.subplot(413, sharex=ax1)
ax3.set(title="RBI")
ax3.set_ylim([0, 100])
plt.plot(rounds, w_RBI)
plt.setp(ax3.get_xticklabels(), visible=False)

ax4 = plt.subplot(414, sharex=ax1)
ax4.set(title="SB")
ax4.set_ylim([0, 100])
plt.plot(rounds, w_SB)

plt.show()
