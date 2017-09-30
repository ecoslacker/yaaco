#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Tree-like networks layout (topology) optimization using ACO metaheuristic

Created on Tue Sep 26 13:38:06 2017
@author: ecoslacker
"""
from datetime import datetime
from yaaco import ACO


def optim_layout_aco(instance, executions, **kwargs):
    """ Optimize layout with ACO

    Optimization of tree-like networks layout (topology)

    :param instance: file name of the problem instance
    :param int executions: times the ACO algorithm will be executed
    :param str savedir: path to save the results and statistics
    :param int n_ants: number of ants in the colony
    :param float rho: the pheromone evaporation parameter
    :param float alpha: the pheromone trail influence
    :param float beta: the heuristic information influence
    :param int max_iters: maximum number of iterations of the algorithm
    :param str ptype: Description of problem instance (Default 'TSP')
    :param bool base: True to use data from a base graph file
    """
    # Get the parameters
    savedir = kwargs.get('savedir', 'results/')
    n_ants = kwargs.get('n_ants', 10)
    _rho = kwargs.get('rho', 0.02)
    _alpha = kwargs.get('alpha', 1.0)
    _beta = kwargs.get('beta', 2.0)
    _iters = kwargs.get('max_iters', 20)
    _ptype = kwargs.get('ptype', 'TSP')
    _base = kwargs.get('base', False)

    best = None
    t = datetime.now()       # File names will be identified by its exec time
    f = '%Y_%m_%d-%H_%M_%S'  # Date format
    # File names to save the statistics, network plot and solution
    stats = savedir + 'aco_layout_' + datetime.strftime(t, f) + '.csv'
    net_img = stats[:-4] + '_network.png'
    sol_file = stats[:-4] + '.txt'

    # Execute the ACO algorithm the specified times
    for i in range(executions):
        print('Execution {0}'.format(i))
        print('-' * 80)

        # Get the best solution from the current execution
        aco1 = ACO(n_ants, instance, rho=_rho, alpha=_alpha, beta=_beta,
                   max_iters=_iters, use_base_graph=_base,
                   instance_type=_ptype)
        b = aco1.run()

        # Initialize the best individual
        if (i == 0) or (b.tour_length < best.tour_length):
            best = b

    # Print best ant (solution)
    print('*** OVERALL BEST ***')
    print(best)

    # Plot the best solution
    if _ptype == 'TSP':
        aco1.plot_best_tour()
    else:
        aco1.plot_tree(best)

    # Save the best solution to a text file
    with open(sol_file, 'w') as f:
        f.write('{0}'.format(best))

    return best

if __name__ == "__main__":

    # Problem instance
    file_name = "../networks_design/data/network09.csv"

    # Parameters for Ant System
    m = 9
    r = 0.5
    aco_iters = 10
    problem = 'TSP'

    execs = 20

    optim_layout_aco(file_name, execs, n_ants=m, rho=r, max_iters=aco_iters,
                     ptype=problem, base=False)
