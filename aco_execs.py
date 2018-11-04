#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Nov  3 01:28:26 2018

@author: eduardo
"""
import csv
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
from yaaco import ACO

def aco_execs(data_dir, prob, res_dir, **kwargs):
    """ Execute ACO for the Symmetric TSP 
    
    Execute the Ant Colony Optimization algorithm several times in search for
    good solutions to the specified instance of the symmetric Traveling
    Salesman Problem.
    
    This function will save files with the best solution and execution
    statistics to the results directory, also best tour and convergence plots
    are saved.

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
        f.write(str(tsp_aco))
        f.write('\n')
        f.write('Best overall solution:\n{0}\n'.format(best_sol))
    best_aco.plot_best_tour(f_plot)
    
    # Show the results
    print("\nBest overall solution:")
    print(best_sol)
    
    # Save convergence plot
    plot_convergence(_execs, _iters, f_stat)
    
    end = datetime.now() - start
    print ('Runtime: {0} executions in {1}'.format(_execs, end))

def plot_convergence(execs, gens, file_stats):
    """ Creates a convergence chart

    Plot the fitness of each generation to create a convergence chart, using
    the data saved in the file created by the evolution.

    :param execs, number of executions or runs
    :param gens, generations of each execution
    :param file_stats, CSV file containing data from evolution
    """

    HEADER = 1   # Header rows
    COL_GEN = 1  # Exec
    COL_MIN = 3  # Min

    # Create the figure
    figConvergence = plt.figure()

    # Read all the data from the file
    gen = []
    rmin = []

    with open(file_stats, 'r') as f:
        reader = csv.reader(f, delimiter='\t')
        i = 0
        for row in reader:
            # Convert to numeric values, ignore headers
            if i >= HEADER:
                gen.append(int(row[COL_GEN]))      # Generation
                rmin.append(float(row[COL_MIN]))   # Raw minimum
            i += 1

    # Get the data slicing through executions, each run contains all its
    # generations and fitness
    ini = 0
    end = gens
    best = rmin[ini:end]
    g = gen[ini:end]

    # Get the best of each generation
    for r in range(execs):
        y = rmin[ini:end]
        for i in range(len(y)):
            if y[i] < best[i]:
                best[i] = y[i]
        ini += gens
        end += gens

    # Plot the data: generation vs best fitness
    plt.plot(g, best, c='black')

    # Create a new file with PNG extension to save the chart
    file_fig = file_stats[:-4]
    file_fig = file_fig + ".png"

    plt.title('Convergence chart for {0} executions.'.format(execs))
    plt.xlabel('Generation')
    plt.ylabel('Cost')
    plt.savefig(file_fig, bbox_inches='tight', dpi=300, transparent=True)
    figConvergence.show()


if __name__ == "__main__":
    
    data = 'test_data/'
    prob = 'eil51.tsp'
    res = 'results/'
    
    aco_execs(data, prob, res, execs=3)

