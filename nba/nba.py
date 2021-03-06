#!/usr/bin/env python2
"""
Predicting NBA games with Trueskill
"""
from __future__ import division
import numpy as np
import pylab as pl
import trueskill as ts
from trueskill.mathematics import cdf

def parse(filename, homeaway=False):
    scoredata = np.loadtxt(filename,
                           delimiter=',',
                           comments="#",
                           dtype=({'names':
                                       ['date', 'type', 'team1', 'score1', 'team2', 'score2', 'comment', 'empty'],
                                   'formats':
                                       ['S100', 'S100', 'S100', 'i4', 'S100', 'i4', 'S100', 'S100', ]}),
                           )
    teams = {}
    for match in scoredata:
        t1, t2 = match['team1'], match['team2']
        if homeaway:
            t1 += "H"
            t2 += "A"
        if t1 not in teams:
            teams[t1] = ts.Rating()
        if t2 not in teams:
            teams[t2] = ts.Rating()
    return teams, scoredata

def updateRating(team1, team2):
    r1, s1 = team1
    r2, s2 = team2
    if s1 > s2:
        new_r1, new_r2 = ts.rate_1vs1(r1, r2, drawn=False)
    elif s2 < s1:
        new_r2, new_r1 = ts.rate_1vs1(r2, r1, drawn=False)
    else:  # Draw
        new_r1, new_r2 = ts.rate_1vs1(r1, r2, drawn=True)
    return new_r1, new_r2


def train(teams, scores, homeaway=False):
    for match in scores:
        t1, t2 = match['team1'], match['team2']
        if homeaway:
            t1 += "H"
            t2 += "A"
        s1, s2 = match['score1'], match['score2']
        teams[t1], teams[t2] = updateRating((teams[t1], s1), (teams[t2], s2))
    return teams

def test(teams, scores, homeaway=False, showmissed=False):
    correct, count = 0, 0
    for match in scores:
        count += 1
        t1, t2 = match['team1'], match['team2']
        if homeaway:
            t1 += "H"
            t2 += "A"
        s1, s2 = match['score1'], match['score2']
        r1, r2 = teams[t1], teams[t2]
        w = Pwin(r1, r2)
        # if 0.3 <= w <= 0.7:  # skip
        #     count -= 1
        if ((w > 0.5) and (s1 > s2)) or ((w < 0.5) and (s1 < s2)):
            correct += 1
        elif showmissed:
            if  w < 0.5:
                w = 1-w
            print "%s (%.1f, %.2f) %d : %d %s (%.1f, %.2f) [winP: %.1f%%]" %(t1, r1.mu, r1.sigma, s1, s2, t2, r2.mu, r2.sigma, w*100)
    predict = correct/count
    diff = 1.96 * np.sqrt(predict*(1-predict)/count)
    print "\nPre-trained:\n"+"="*10
    print "Predictions: %d / %d (%.1f +- %.1f%%: %.1f-%.1f)" %(correct, count, predict*100, diff*100, (predict-diff)*100, (predict+diff)*100)

def testTrain(teams, scores, homeaway=False):
    correct, count = 0, 0
    for match in scores:
        count += 1
        t1, t2 = match['team1'], match['team2']
        if homeaway:
            t1 += "H"
            t2 += "A"
        s1, s2 = match['score1'], match['score2']
        r1, r2 = teams[t1], teams[t2]
        w = Pwin(r1, r2)
        if ((w > 0.5) and (s1 > s2)) or ((w < 0.5) and (s1 < s2)):
            correct += 1
        teams[t1], teams[t2] = updateRating((r1, s1), (r2, s2))
    predict = correct/count
    diff = 1.96 * np.sqrt(predict*(1-predict)/count)
    print "\nKeep training:\n"+"="*10
    print "Predictions: %d / %d (%.1f +- %.1f%%: %.1f-%.1f)" %(correct, count, predict*100, diff*100, (predict-diff)*100, (predict+diff)*100)


# From: https://github.com/sublee/trueskill/issues/1
def Pwin(rA, rB):
    """ Calculate winning probability """
    deltaMu = rA.mu - rB.mu
    rsss = np.sqrt(rA.sigma**2 + rB.sigma**2)
    return cdf(deltaMu/rsss)

if __name__ == "__main__":
    import sys
    datafile = sys.argv[1]
    teams, scores = parse(datafile)
    machno = len(scores)
    stopno = int(machno * 0.9)
    trainset = scores[0:stopno]
    testset = scores[stopno:]
    trained_teams = train(teams, trainset)
    testres = test(trained_teams, testset, showmissed=False)
    testres2 = testTrain(trained_teams, testset)

    # teams, scores = parse(datafile, homeaway=True)
    # machno = len(scores)
    # stopno = int(machno * 0.9)
    # trainset = scores[0:stopno]
    # testset = scores[stopno:]
    # trained_teams = train(teams, trainset, homeaway=True)
    # testres = test(trained_teams, testset, homeaway=True)
    # testres2 = testTrain(trained_teams, testset, homeaway=True)

    # print Pwin(trained_teams['Portland Trail Blazers'], trained_teams['Miami Heat'])
    # print Pwin(trained_teams['Oklahoma City Thunder'], trained_teams['Utah Jazz'])
    # print Pwin(trained_teams['Phoenix Suns'], trained_teams['Los Angeles Lakers'])

    # # Double train
    # datafile = '2011_01.csv'
    # teams, scores = parse(datafile)
    # trained_teams = train(teams, scores)
    # datafile = '2012_01.csv'
    # teams, scores = parse(datafile)
    # machno = len(scores)
    # stopno = int(machno * 0.9)
    # trainset = scores[0:stopno]
    # testset = scores[stopno:]
    # trained_teams = train(trained_teams, trainset)
    # testres = test(trained_teams, testset)

