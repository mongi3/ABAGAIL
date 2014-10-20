import sys
import os
import time

import java.io.FileReader as FileReader
import java.io.File as File
import java.lang.String as String
import java.lang.StringBuffer as StringBuffer
import java.lang.Boolean as Boolean
import java.util.Random as Random

import dist.DiscreteDependencyTree as DiscreteDependencyTree
import dist.DiscreteUniformDistribution as DiscreteUniformDistribution
import dist.Distribution as Distribution
import opt.DiscreteChangeOneNeighbor as DiscreteChangeOneNeighbor
import opt.EvaluationFunction as EvaluationFunction
import opt.GenericHillClimbingProblem as GenericHillClimbingProblem
import opt.HillClimbingProblem as HillClimbingProblem
import opt.NeighborFunction as NeighborFunction
import opt.RandomizedHillClimbing as RandomizedHillClimbing
import opt.SimulatedAnnealing as SimulatedAnnealing
import opt.example.FourPeaksEvaluationFunction as FourPeaksEvaluationFunction
import opt.ga.CrossoverFunction as CrossoverFunction
import opt.ga.SingleCrossOver as SingleCrossOver
import opt.ga.DiscreteChangeOneMutation as DiscreteChangeOneMutation
import opt.ga.GenericGeneticAlgorithmProblem as GenericGeneticAlgorithmProblem
import opt.ga.GeneticAlgorithmProblem as GeneticAlgorithmProblem
import opt.ga.MutationFunction as MutationFunction
import opt.ga.StandardGeneticAlgorithm as StandardGeneticAlgorithm
import opt.ga.UniformCrossOver as UniformCrossOver
import opt.prob.GenericProbabilisticOptimizationProblem as GenericProbabilisticOptimizationProblem
import opt.prob.MIMIC as MIMIC
import opt.prob.ProbabilisticOptimizationProblem as ProbabilisticOptimizationProblem
import shared.FixedIterationTrainer as FixedIterationTrainer
import shared.OccasionalPrinter as OccasionalPrinter
import opt.example.CallCountingEvaluationFunction as CallCountingEvaluationFunction
from array import array
import timeit

import opt.example.CountOnesEvaluationFunction as CountOnesEvaluationFunction
import opt.example.FourPeaksEvaluationFunction as FourPeaksEvaluationFunction
from opt.example import FlipFlopEvaluationFunction




"""
Commandline parameter(s):
   none
"""

def train(trainer, iterations, iterPrint, ef, perfect, method, problemKey):
    iterCnt = 0
    ef.callCnt = 0; # reset call cnt
    start = timeit.default_timer()
    while iterCnt < iterations:
        trainer.train()
        reachedPerfect = abs(ef.valueNoCount(trainer.getOptimal())-perfect) < 0.1
        if reachedPerfect:
            break
        if (iterCnt % iterPrint) == 0 and iterCnt != 0:
            print problemKey,method,N,iterCnt,str(ef.valueNoCount(trainer.getOptimal())),(timeit.default_timer()-start),ef.callCnt
        iterCnt += 1
    print problemKey,method,N,iterCnt,str(ef.valueNoCount(trainer.getOptimal())),(timeit.default_timer()-start),ef.callCnt
    print >> outFile,problemKey,method,N,iterCnt,str(ef.valueNoCount(trainer.getOptimal())),(timeit.default_timer()-start),ef.callCnt
    
            
def runSims(N,problemKey):
    fill = [2] * N
    ranges = array('i', fill)
    
    if problemKey is 'fourpeaks':
        T = N/10
        ef = CallCountingEvaluationFunction(efmap[problemKey](T))
        perfectAns = N + (N-(T+1))
    elif problemKey is 'countones':
        ef = CallCountingEvaluationFunction(efmap[problemKey]())
        perfectAns = N
    elif problemKey is 'flipflop':
        ef = CallCountingEvaluationFunction(efmap[problemKey]())
        perfectAns = N-1
    else:
        raise "invalid problem key!"
    
    odd = DiscreteUniformDistribution(ranges)
    nf = DiscreteChangeOneNeighbor(ranges)
    mf = DiscreteChangeOneMutation(ranges)
    cf = SingleCrossOver()
    df = DiscreteDependencyTree(.1, ranges)
    hcp = GenericHillClimbingProblem(ef, odd, nf)
    gap = GenericGeneticAlgorithmProblem(ef, odd, mf, cf)
    pop = GenericProbabilisticOptimizationProblem(ef, odd, df)
    
#    rhc = RandomizedHillClimbing(hcp)
#    method = 'RHC'
#    train(rhc,1001,100,ef)
    
    sa = SimulatedAnnealing(100, .95, hcp)
    method = 'SA'
    train(sa,maxIter,iterPrint,ef,perfectAns,method,problemKey)
    
    ga = StandardGeneticAlgorithm(100, 50, 5, gap)
    method = 'GA'
    train(ga,maxIter,iterPrint,ef,perfectAns,method,problemKey)
    
    mimic = MIMIC(100, 20, pop)
    method = 'MIMIC'
    train(ga,maxIter,iterPrint,ef,perfectAns,method,problemKey)
    
maxIter = 99999
iterPrint = maxIter
numRepeats = 100

efmap = {'countones': CountOnesEvaluationFunction,
         'flipflop': FlipFlopEvaluationFunction,
         'fourpeaks': FourPeaksEvaluationFunction,
         }

import time
timestr = time.strftime("%Y%m%d-%H%M%S")
outFile = open('%s.txt' % timestr,'w')

for k in efmap:
    for N in range(10,81,10):
        for runCnt in range(0,numRepeats):
            runSims(N,k)

outFile.close()