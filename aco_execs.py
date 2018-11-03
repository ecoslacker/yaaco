#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Nov  3 01:28:26 2018

@author: eduardo
"""
import numpy as np
from datetime import datetime
from yaaco import ACO

def aco_execs(data_dir, prob, res_dir, **kwargs):
    """ Execute ACO for the Symmetric TSP 
    
    Execute the Ant Colony Optimization several times to search for good
    solutions to the indicated instance problem of symmetric TSP problem.

    :param str data_dir: path where the input TSP file is located.
    :param str prob: problem TSP data.
    :param str res_dir: path to save the results.
    :param int execs: number of executions of the algorithm.
    :param int ants: ant colony size or number of ants.
    :param int iters: number of iterations.
    :param str flag: algorithm to use: AS, EAS, RAS or MMAS
    """
    
    start = datetime.now()
    
    # Get the optional parameters
    _execs = kwargs.get('execs', 10)
    _ants = kwargs.get('ants', 20)
    _iters = kwargs.get('iters', 500)
    _flag = kwargs.get('flag', 'MMAS')
    
    df = '%Y_%m_%d_%H_%M_%S'  # Date format
    instance = data_dir + prob

    # Save best tour & solution, WARNING: directories should exist!
    f_plot = res_dir + prob + '/' + datetime.strftime(start, df) + '.png'
    f_best = res_dir + prob + '/' + datetime.strftime(start, df) + '.txt'
    f_stat = res_dir + prob + '/' + datetime.strftime(start, df) + '_stat.txt'

    # Initialize the colony stats file
    with open(f_stat, 'w') as f:
        f.write("Exec\tIter\tMax\tMin\tAvg\tStd\n")

    min_length = np.inf
    best_sol = None
    best_aco = None
    for i in range(1, _execs+1):
        print('*** RUNNING EXECUTION: {0} of {1} ***'.format(i, _execs))

        # **** Problem instance data (TSP coordinates file) ****
        # Create the ACO object & run
        tsp_aco = ACO(instance, ants=_ants, max_iters=_iters, flag=_flag)
        best, stats = tsp_aco.run(i)
        
        # Find the best solution with minimal tour length
        if best.tour_length < min_length:
            min_length = best.tour_length
            best_sol = best.clone()
            best_aco = tsp_aco
        
        # Write the colony stats to a file
        with open(f_stat, 'a') as f:
            for row in stats:
                f.write('{0}\t{1}\t{2}\t{3}\t{4}\t{5}\n'.format(row[0],
                        row[1], row[2], row[3], row[4], row[5]))

    # Save the results
    with open(f_best, 'w') as f:
        f.write('Best overall solution:\n{0}\n'.format(best_sol))
    best_aco.plot_best_tour(f_plot)
    
    # Show the results
    print("\nBest overall solution:")
    print(best_sol)
    
    end = datetime.now() - start
    print ('Runtime: {0} executions in {1}'.format(_execs, end))


if __name__ == "__main__":
    
    data = 'test_data/'
    prob = 'eil51.tsp'
    res = 'results/'
    
    aco_execs(data, prob, res, execs=3)
