#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
yaaco.py

Yet Another Ant Colony Optimization Python Implementation

An attempt to code the Ant Colony Optimization (ACO) metaheuristic in
Python 2.7 language.

IMPORTANT: This code only includes AS and MMAS algorithms.

To understand what this code does you should probably read the book:

  Dorigo, M., & Stützle, T. (2004). Ant colony optimization. Massachusetts,
  United States of America: Massachusetts Institute of Technology.

@author: ecoslacker
"""
# import matplotlib
# matplotlib.use('TkAgg')
import csv
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from datetime import datetime
from math import sqrt, pow
from matplotlib.path import Path

# To print complete arrays
np.set_printoptions(threshold='nan')

MAXFACTOR = 3


#def euclidean(x1, y1, x2, y2):
#    """ Distance
#
#    Computes the Euclidean distance between two points (x1, y1) and (x2, y2)
#
#    :param float x1: first point x-axis coordinate
#    :param float y1: first point y-axis coordinate
#    :param float x2: second point x-axis coordinate
#    :param float y2: second point y-axis coordinate
#    :return: distance
#    """
#    assert type(x1) is float, "Coordinates should be float type"
#    assert type(y1) is float, "Coordinates should be float type"
#    assert type(x2) is float, "Coordinates should be float type"
#    assert type(y2) is float, "Coordinates should be float type"
#
#    return sqrt(pow(x2 - x1, 2) + pow(y2 - y1, 2))


class Ant:
    """ Single ant

    Creates a single ant with its properties

    :param int size: the dimension or length of the ant
    :param str atype: the type of problem instance
    """

    uid = 0

    def __init__(self, size, atype='TSP'):
        """ Initialize the Ant object """
        assert type(size) is int, "The Ant size should be integer type"
        self.size = size
        self.ant_type = atype
        self.uid = self.__class__.uid
        self.__class__.uid += 1
        self.tour_length = np.inf

        self.tour = np.ones(self.size, dtype=np.int64) * -1
        self.visited = np.zeros(self.size, dtype=np.int64)
        self.tree = []

        if self.ant_type == 'TSP':
            self.tour = np.ones(self.size+1, dtype=np.int64) * -1

    def __str__(self):
        """ String representation of the Ant object """
        text = "Ant:\n"
        text += " UID:     " + str(self.uid) + "\n"
        text += " Type:    " + str(self.ant_type) + "\n"
        text += " Tour:    " + str(self.tour) + "\n"
        text += " Visited: " + str(self.visited) + "\n"
        text += " Network: " + str(self.tree) + " (tree-like)\n"
        text += " Tour length: " + str(self.tour_length) + "\n"
        return text

    def clone(self):
        """ Returns a deep copy of the current Ant instance with a new UID

        :return Ant ant: instance with same properties
        """
        ant = Ant(len(self.tour), self.ant_type)
        ant.tour_length = self.tour_length
        ant.tour = self.tour.copy()
        ant.visited = self.visited.copy()
        ant.tree = self.tree[:]
        return ant


class Problem:
    """ Problem instance

    :param str filename: a tex file with the data of the problem instance
    :param callable func: a function to calculate the distance
    :param str name: the name of the problem (default is "Problem#")
    """

    uid = 0

    def __init__(self, filename, func, **kwargs):
        """ Initialize the problem instance """
        if func is None:
            print("No distance function provided, assuming Euclidean 2D")
            self.func = self.euclidean_2D
        else:
            self.func = func
        self.file_instance = filename
        self.name = kwargs.get('name', 'Problem{0}'.format(self.uid))
        self.type = kwargs.get('ptype', 'TSP')
        self.base = kwargs.get('base', False)

        # Initialize some variables
        self.file_graph = ''
        self.base_graph = []

        # WARNING! This will overwrite the name of the problem
        self.x, self.y, self.name = self.read_instance(self.base)

        assert len(self.x) > 0, "Coordinates list is empty"
        assert len(self.y) > 0, "Coordinates list is empty"
        assert len(self.x) == len(self.y), "Coordinates must have same length"

        # The dimension of the problem, this is the number of cities for the
        # TSP problem
        self.dimension = len(self.x)
        self.distance_matrix = np.zeros((self.dimension, self.dimension))

        self.compute_distances()  # Compute distance matrix
        self.compute_nearest_neighbor()  # nn_list
        self.nn_tour_tsp()  # Nearest-neighbor tour initialization

        # If not base graph specified, then create one
        if not self.base:
            self.create_base_graph()

    def read_instance(self, base_graph=False):
        """ Read instance

        Reads the problem instance from a text delimited file, the file must
        be formatted as the *.tsp type as described in TSPLIB

        :param base_graph, a boolean indicating if a base graph is required
        :return x, the x-axis coordinates of the instance
        :return y, the y-axis coordinates of the instance
        :return name, the name of the instance
        """

        x = []
        y = []
        name = ''
        try:
            if base_graph:
                self.file_graph = self.file_instance[:-4] + '_base.csv'
                print("Using base graph from: {0}".format(self.file_graph))
                with open(self.file_graph, 'r') as f:
                    self.base_graph = f.read().split('\n')
                    # Remove empty lines
                    if '' in self.base_graph:
                        self.base_graph.remove('')
            read = False  # Token that indicates when to read the coordinates
            with open(self.file_instance, 'r') as f:
                reader = csv.reader(f, delimiter=' ')
                for row in reader:
                    if row[0] == 'NAME':
                        name = row[2]
                    if row[0] == 'NODE_COORD_SECTION':
                        read = True
                    elif row[0] == 'EOF':
                        read = False
                    elif read is True:
                        x.append(float(row[1]))
                        y.append(float(row[2]))
        except IOError as e:
            z = e
            print(z)
        return x, y, name

    def compute_distances(self):
        """ Compute distances

        Computes the distance matrix for the problem
        """

        for i in range(self.dimension):
            for j in range(self.dimension):
                # Calculate the distance between the points
                d = self.func(self.x[j], self.y[j], self.x[i], self.y[i])
                self.distance_matrix[i][j] = d

        # Assign huge values to diagonal in distance matrix, making the
        # distance from a point (city) to itself greater than the maximum
        # Get max value from each row
        row_max = np.amax(self.distance_matrix, axis=1)
        row_max *= MAXFACTOR

        # Diagonal identity matrix
        diag = np.eye(self.dimension)
        dmax = diag * row_max
        self.distance_matrix = self.distance_matrix + dmax

        return self.distance_matrix

    def compute_nearest_neighbor(self):
        """ Compute nearest-neighbor

        Get the nearest-neighbor list for each city, row "i" is a list of
        the nearest cities to the city "i".

        The nearest-neighbor list, nn_list[i][r] gives the identifier (index)
        of the r-th nearest city to city i (i.e. nn_list[i][r] = j).

        WARNING! For this to work the cities should be labeled as 0,1,...,N
        """

        nn = []
        for row in self.distance_matrix:
            n = range(self.dimension)  # Number of cities
            d = row.tolist()           # Distances from "i" to each city
            # Sort by distance, then by number of city
            indexes = np.lexsort((n, d))
            nn.append(indexes)

        self.nn_list = np.array(nn)

        return self.nn_list

    def create_base_graph(self):
        """ Create base graph

        Creates a base graph using every possible combination of the nodes
        in the problem.
        WARNING! This could be time consuming!
        """
        self.base_graph = []
        for i in range(self.dimension):
            for j in range(self.dimension):
                if i != j:
                    edge1 = '{0}-{1}'.format(i, j)
                    edge2 = '{0}-{1}'.format(j, i)
                    if (edge1 not in self.base_graph) and \
                       (edge2 not in self.base_graph):
                        self.base_graph.append(edge1)
        return self.base_graph

    def nn_tour_tsp(self, ttype='TSP'):
        """ A TSP tour generated by the nearest-neighbor heuristic

        :param ttype, tour type could be 'TSP' or 'NET'
        :return length: the length of the generated tour
        """
        self.ant = Ant(self.dimension, ttype)
        phase = 0
        # Initialize ant to a random city
        r = np.random.randint(0, self.dimension)
        self.ant.tour[phase] = r
        self.ant.visited[r] = 1  # True

        while (phase < self.dimension - 1):
            phase += 1
            # Choose closest next
            current_city = self.ant.tour[phase - 1]  # current city
            next_city = self.dimension-1  # next city
            min_dist = np.inf
            for city in range(self.dimension):
                if not self.ant.visited[city]:
                    if self.distance_matrix[current_city][city] < min_dist:
                        next_city = city  # next city
                        min_dist = self.distance_matrix[current_city][city]
            # Once all cities had been tested, set the next one
            self.ant.tour[phase] = next_city
            self.ant.visited[next_city] = 1  # True
        # Set the last city
        if ttype == 'TSP':
            self.ant.tour[-1] = self.ant.tour[0]

    def plot_tour(self):
        """ Plot tour

        Plot the complete tour of an Ant instance

        :Ant ant, an instance of the Ant class
        """
        figTour = plt.figure()
        ax = figTour.add_subplot(1, 1, 1)
        plt.scatter(self.x, self.y)

        labels = ['{0}'.format(l) for l in range(len(self.x))]
        for label, lx, ly in zip(labels, self.x, self.y):
            plt.annotate(label, xy=(lx, ly), xytext=(-5, 5),
                         textcoords='offset points')

        plt.title("Nearest-Neighbor TSP tour")
        for j in range(self.dimension):
            p1 = self.ant.tour[j]    # Initial point
            p2 = self.ant.tour[j+1]  # Final point
            # Draw a line from (x1, y1) to (x2, y2)
            x1 = self.x[p1]
            y1 = self.y[p1]
            x2 = self.x[p2]
            y2 = self.y[p2]
            verts = [(x1, y1), (x2, y2)]
            codes = [Path.MOVETO, Path.LINETO]
            path = Path(verts, codes)
            ax.add_patch(patches.PathPatch(path, lw=0.5))
#            plt.xlim([min(self.x) - 50, max(self.x) + 50])
#            plt.ylim([min(self.y) - 50, max(self.y) + 50])
        plt.show()

    def euclidean_2D(self, x1, y1, x2, y2):
        """ Distance

        Computes the Euclidean distance between two points (x1, y1)
        and (x2, y2)

        :param float x1: first point x-axis coordinate
        :param float y1: first point y-axis coordinate
        :param float x2: second point x-axis coordinate
        :param float y2: second point y-axis coordinate
        :return: distance
        """
        assert type(x1) is float, "Coordinates should be float type"
        assert type(y1) is float, "Coordinates should be float type"
        assert type(x2) is float, "Coordinates should be float type"
        assert type(y2) is float, "Coordinates should be float type"

        return sqrt(pow(x2 - x1, 2) + pow(y2 - y1, 2))


class ACO(Problem):
    """ The Ant Colony Optimization metaheuristic

    Creates and initializes a colony of ants to run the algorithm

    :param int ants: number of ants in the colony
    :param str filename: a tex file with the data of the problem instance
    :param float rho: the pheromone evaporation parameter
    :param float alpha: the pheromone trail influence
    :param float beta: the heuristic information influence
    :param int max_iters: maximum number of iterations of the algorithm
    :param str flag: type of algorithm to be used in the ACO metaheuristic
    :param bool use_base_graph: True to use data from a base graph file
    :param str instance_type: Description of problem instance (Default 'TSP')

    Possible flags include:
        AS: Ant System (Default)
        EAS: Elitist Ant System (Not yet implemented)
        RAS: Rank-Based Ant System (Not yet implemented)
        MMAS: Max-Min Ant System
    """

    TSP = 0  # Symmetric TSP problem
    TREE_NET = 1  # Tree-like network layout problem
    SIZE_NET = 2  # Network pipe sizing problem
    LOOP_NET = 3  # Looped network layout problem

    def __init__(self, ants, filename, rho=0.5, alpha=1.0, beta=2.0,
                 nn_ants=20, max_iters=100, **kwargs):
        """ Initialize

        Set parameters and initializes pheromone trails
        """

        self.start = datetime.now()

        assert type(ants) is int, "The number of ants should be integer"
        # Initialize class variables from arguments
        self.ants = ants
        self.alpha = alpha
        self.beta = beta
        self.rho = rho
        self.max_iters = max_iters
        self.flag = kwargs.get('flag', 'AS')           # Flag: Ant System
        self.bg = kwargs.get('use_base_graph', False)  # Use base graph?
        function = kwargs.get('function', None)        # Distance function
        instance_type = kwargs.get('instance_type', 'TSP')  # Problem instance

        # Initialize the Problem instance
        Problem.__init__(self, filename, function, ptype=instance_type,
                         name="TSP 0", base=self.bg)
        self.n = self.dimension       # dimension of the problem
        self.nn_ants = nn_ants        # nearest-neighbors in tour construction
        if self.n < self.nn_ants:     # check if this is a small instance
            self.nn_ants = self.n
        self.Cnn = self.compute_tour_length(self.ant)

        # Initial pheromone trail
        self.tau_0 = self.ants / self.Cnn
        if self.flag == 'MMAS':
            self.tau_0 = 1.0 / (self.rho * self.Cnn)

        self.create_colony(instance_type)

        # Initialize pheromone trail matrix
        self.reset_pheromone(self.tau_0)

        # Heuristic information (eta)
        self.eta = 1.0 / (self.distance_matrix + 0.1)
        # Add a small constant to avoid division by zero if a distance is zero

        # Initialize the choice information matrix & compute it
        self.choice_info = np.empty_like(self.pheromone)
        self.choice_info[:] = self.pheromone
        self.compute_choice_information()

        # Pheromone trails are initialized to upper pheromone trail limit
        self.tau_max = 1.0 / (self.rho * self.Cnn)
        self.tau_min = self.tau_max / (2.0 * self.n)

        # Initialize the variables concerning statistics
        self.iteration = 0
        self.best_so_far_ant = Ant(self.n)
        self.best_so_far_ant.tour_length = np.inf
        self.show_initialization(False, False)

        self.plot_base_graph()

        # Measure initialization time
        self.exec_time = 0
        self.init_time = datetime.now() - self.start
        print("Initialization time: {0}".format(self.init_time))

    def __str__(self):
        """ str

        A string representation of the ant colony
        """

        text = "Colony size: " + str(self.ants) + "\n"
        text += "Problem dimension : " + str(self.n) + "\n"
        i = 0
        for ant in self.colony:
            print(ant)
            i += 1
        return text

    def show_initialization(self, matrices=True, plot=False):
        """ Show initialization

        Shows the default parameters
        :return: None
        """
        print("Initialization parameters:")
        print("  Problem:  {0}".format(self.type))
        print("  Ants:     {0}".format(self.ants))
        print("  alpha:    {0}".format(self.alpha))
        print("  beta:     {0}".format(self.beta))
        print("  rho:      {0}".format(self.rho))
        print("  max_iters:{0}".format(self.max_iters))
        print("  flag:     {0}".format(self.flag))
        print("  Cnn:      {0} (initial tour length)".format(self.Cnn))
        print("  Tau max:  {0}".format(self.tau_max))
        print("  Tau min:  {0}".format(self.tau_min))
        print("  Tau 0:    {0}".format(self.tau_0))
        if plot:
            print("  Initial tour (Using nearest-neighbor heuristic):")
            self.plot_tour()
        if matrices:
            print("  Distance matrix:")
            print(self.distance_matrix)
            print("  Pheromone matrix:")
            print(self.pheromone)
            print("  Choice information matrix")
            print(self.choice_info)
        else:
            # Nothing to show
            return

    def create_colony(self, ant_type):
        """ Create colony

        Creates a colony of ants (Ant instances) according the number of ants
        specified.
        """
        colony = []
        for i in range(self.ants):
            colony.append(Ant(self.n, ant_type))
        self.colony = np.array(colony)

    def reset_pheromone(self, level=0.1):
        """ Reset the pheromone to a default level

        :param float level: default level to reset the pheromone
        """
        self.pheromone = np.ones((self.n, self.n), dtype=np.float) * level

    def plot_nodes(self):
        """ Plot nodes

        Plots the nodes of the network
        """
        fig1 = plt.figure()
        fig1.clear()
        plt.title("Network: " + self.name)
        plt.scatter(self.x, self.y)
        labels = ['{0}'.format(l) for l in range(len(self.x))]
        for label, lx, ly in zip(labels, self.x, self.y):
            plt.annotate(label, xy=(lx, ly), xytext=(-5, 5),
                         textcoords='offset points')
        plt.show()

    def plot_base_graph(self):
        """ Plot base graph

        Plot the base graph from the data file
        """
        figBase = plt.figure()
        figBase.clear()
        plt.title("Base graph for: " + self.name)
        plt.scatter(self.x, self.y)
        labels = ['{0}'.format(l) for l in range(len(self.x))]
        for label, lx, ly in zip(labels, self.x, self.y):
            plt.annotate(label, xy=(lx, ly), xytext=(-5, 5),
                         textcoords='offset points')
        for arc in range(len(self.base_graph)):
            n = self.get_nodes(self.base_graph[arc])
            plt.plot([self.x[n[0]], self.x[n[1]]],
                     [self.y[n[0]], self.y[n[1]]])
        plt.show()

    def plot_tree(self, ant):
        """ Plot tree

        Plot the tree-like network from the ant

        :param ant, Ant object which tree will be plotted
        """
        figTree = plt.figure()
        figTree.clear()
        plt.scatter(self.x, self.y)
        labels = ['{0}'.format(l) for l in range(len(self.x))]
        for label, lx, ly in zip(labels, self.x, self.y):
            plt.annotate(label, xy=(lx, ly), xytext=(-5, 5),
                         textcoords='offset points')
        for i in ant.tree:
            n = self.get_nodes(self.base_graph[i])
            plt.plot([self.x[n[0]], self.x[n[1]]],
                     [self.y[n[0]], self.y[n[1]]])
        plt.show()

    def plot_all_tours(self):
        """ Plot all tours

        Plots all the tours generated by a TSP instance
        WARNING! Use with caution (see labels and colours list)
        :return:
        """
        fig2 = plt.figure()
        ax = fig2.add_subplot(1, 1, 1)

        plt.title("TSP tours")
        # Plot the nodes
        labels = ['{0}'.format(l) for l in range(len(self.x))]
        for label, lx, ly in zip(labels, self.x, self.y):
            plt.annotate(label, xy=(lx, ly), xytext=(-5, 5),
                         textcoords='offset points')
        # Plot the tours of all ants
        colours = ['black', 'green', 'blue', 'red', 'yellow', 'cyan']
        for i in range(self.ants):
            for j in range(self.n):
                p1 = self.colony[i].tour[j]    # Initial point
                p2 = self.colony[i].tour[j+1]  # Final point
                # Draw a line from (x1, y1) to (x2, y2)
                x1 = self.x[p1]
                y1 = self.y[p1]
                x2 = self.x[p2]
                y2 = self.y[p2]
                verts = [(x1, y1), (x2, y2)]
                codes = [Path.MOVETO, Path.LINETO]
                path = Path(verts, codes)
                ax.add_patch(patches.PathPatch(path, color=colours[i], lw=0.5))
                plt.xlim([-50, 250])
                plt.ylim([-50, 250])
        plt.show()

    def plot_best(self):
        """ Plot best

        Plot the best solution to the problem from the Ant colony.
        """
        if self.itype == 'TSP':
            self.plot_best_tour()
        else:
            self.plot_best_tree()

    def plot_best_tour(self):
        """ Plot tour

        Plot the complete tour of an Ant instance

        :Ant ant, an instance of the Ant class
        """
        figBestTour = plt.figure()
        ax = figBestTour.add_subplot(1, 1, 1)
        plt.scatter(self.x, self.y)

        labels = ['{0}'.format(l) for l in range(len(self.x))]
        for label, lx, ly in zip(labels, self.x, self.y):
            plt.annotate(label, xy=(lx, ly), xytext=(-5, 5),
                         textcoords='offset points')

        plt.title("Best TSP tour")
        for j in range(self.dimension):
            p1 = self.best_so_far_ant.tour[j]    # Initial point
            p2 = self.best_so_far_ant.tour[j+1]  # Final point
            # Draw a line from (x1, y1) to (x2, y2)
            x1 = self.x[p1]
            y1 = self.y[p1]
            x2 = self.x[p2]
            y2 = self.y[p2]
            verts = [(x1, y1), (x2, y2)]
            codes = [Path.MOVETO, Path.LINETO]
            path = Path(verts, codes)
            ax.add_patch(patches.PathPatch(path, lw=0.5))
#            plt.xlim([min(self.x) - 50, max(self.x) + 50])
#            plt.ylim([min(self.y) - 50, max(self.y) + 50])
        plt.show()

    def plot_best_tree(self):
        """ Plot best tree

        Plot the best tree-like network solution from the Ant colony
        """
        figBestTour = plt.figure()
        ax = figBestTour.add_subplot(1, 1, 1)
        plt.scatter(self.x, self.y)

        labels = ['{0}'.format(l) for l in range(len(self.x))]
        for label, lx, ly in zip(labels, self.x, self.y):
            plt.annotate(label, xy=(lx, ly), xytext=(-5, 5),
                         textcoords='offset points')

        plt.title("Best tree network")
        plt.show()

    def run(self):
        """ Run

        Call this function to actually run the ACO metaheuristic. All the ants
        (Ant instances) will construct solutions and update pheromone while
        the termination criteria aren't satisfied.
        :return best_so_far_ant: Ant instance with the best tour
        """
        print("*** Running Ant Colony Optimization ***")
        # 1. Initialize data: this was already done in the __init__ function

        # Selet the proper function to the problem
        construct_solutions = self.tsp_construct_solutions
        if self.type != 'TSP':
            construct_solutions = self.net_construct_solutions
            print("Problem type: {0}".format(self.type))

        # 2. Loop
        while not self.termination_criteria():
            print("\nIteration {0}".format(self.iteration))
            construct_solutions()

            # Local search (optional)

            # Update statistics
            iter_best_ant = self.find_best()

            if iter_best_ant.tour_length < self.best_so_far_ant.tour_length:
                self.best_so_far_ant = iter_best_ant.clone()
                print("Best so far ant:")
                print(self.best_so_far_ant)

            # Update pheromone trails
            self.as_pheromone_update()
            self.iteration += 1

        # Measure execution time
        self.exec_time = datetime.now() - self.start
        print("Execution time: {0}".format(self.exec_time))
        return self.best_so_far_ant

    def termination_criteria(self):
        """ Termination criteria

        Terminate when at least one of the following conditions are True:
        1) The current iteration is greather than the maximum allowed
           iterations

        :return bool: True if the termination criteria are satisfied
        """
        return self.iteration > self.max_iters

    def init_try(self):
        """ Initialize try

        Initialize the parameters for each iteration of the algorithm
        """
        self.reset_pheromone(self.tau_0)
        self.compute_choice_information()

    def tsp_construct_solutions(self):
        """ Construct solutions for TSP

        Construct valid solutions for the Taveling Salesman Problem (TSP)
        """
        step = 0
        # 1. Clear ants memory
        for ant in self.colony:
            for i in range(len(ant.visited)):
                ant.visited[i] = 0  # False
        # 2. Assigns an initial random city to each ant
        for ant in self.colony:
            # Random initial city
            r = np.random.randint(0, self.n)
            ant.tour[step] = r
            ant.visited[r] = 1  # True
        # 3. Each ant constructs a complete tour
        while step < self.n:
            # print(" Step: {0}".format(step))
            step += 1
            for k in range(self.ants):
                self.as_decision_rule(k, step)
        # 4. Move to initial city and compute each ant's tour length
        for ant in self.colony:
            ant.tour[-1] = ant.tour[0]
            self.compute_tour_length(ant)
        return

    def net_construct_solutions(self):
        """ Construct network solutions

        Construct valid sollutions for the tree network optimization problem
        """
        # 1. Clear ants memory, reset the ant colony
        self.create_colony(self.type)

        # 2. Assigns an initial random city to each ant
        # 3. Each ant constructs a complete tour with the
        #    tree growing algorithm
        for ant in self.colony:
            self.tree_growing(ant)

        # 4. Compute ant's network length
        # (This also could be done inside the Tree Growing Algorithm)
        for ant in self.colony:
            self.compute_network_length(ant)

        return

    def get_nodes(self, arc):
        """ Get nodes

        Get the initial and final nodes from an arc

        :param arc, a string arc with the nodes delimited by an "-"
        :return, an integer list with the arc nodes
        """
        assert type(arc) is str, 'Argument should be string'
        chars = arc.split('-')
        nodes = [int(x) for x in chars]
        return nodes

    def adjacent(self, node, arcs):
        """ Adjacent

        Generates a list of adjacent arcs to the given node

        :param node, an integer node
        :param arcs, a list of integer nodes forming a network
        """
        adj = []  # Adjacent arcs
        for i in range(len(arcs)):
            nodes = self.get_nodes(arcs[i])
            if node in nodes:
                adj.append(i)
        return adj

    def adjust_probabilities(self, probabilities):
        """ Adjust probabilities

        Adjust probabilities in the case they don't sum to 1. This is required
        by the Tree Growing Algorithm selection phase.

        :param probabilities, a list with the probabilities
        """
        if sum(probabilities) != 1.0:
            prob = []
            for p in probabilities:
                prob.append(p / sum(probabilities))
        else:
            prob = probabilities
        return prob

    def tree_growing(self, ant, verbose=False):
        """ Tree Growing Algorithm

        Tree Growing Algorithm from Walters & Smith (1995)

        :param Ant ant: the Ant object that will construct a tree-like tour
        :param bool verbose: if True will show usefull information
        """
        step = 0
        AA = []  # set of arcs adjacent to the growing tree

        # Identify the root node Nr
        Nr = np.random.randint(0, self.n)  # Total nodes of the network
        ant.tour[step] = Nr  # set of nodes contained within the growing tree
        ant.visited[Nr] = 1  # Mark as visited=True

        # Initialise AA = [arcs in base graph connected to root'node]
        AA = self.adjacent(Nr, self.base_graph)
        if verbose:
            print("Step {0}".format(step))
            print("  Random root node: {0}".format(Nr))
            print("  AA={0} ({1})".format(AA, len(AA)))

        # while len(ant.tree) != (self.n - 2):
        while step < (self.n - 1):
            step += 1
            print("Step {0}".format(step))

            if verbose:
                print("  AA={0} ({1})".format(AA, len(AA)))

            # Get the probabilities for the adjacent arcs
            probabilities = []
            for _arc in AA:
                _nodes = self.get_nodes(self.base_graph[_arc])
                i = _nodes[0]
                j = _nodes[1]
                probabilities.append(self.choice_info[i][j])

            if verbose:
                print("  Node\tProbability")
                for aa, pp in zip(AA, probabilities):
                    print("  {0}:\t{1}".format(aa, pp))

            # Choose an arc, a, at random from the adjacent arcs
            probabilities = self.adjust_probabilities(probabilities)
            a = int(np.random.choice(AA, 1, p=probabilities))
            ant.tree.append(a)  # Add arc to tree-like network

            # Identify newly connected node, N
            nodes = self.get_nodes(self.base_graph[a])
#            N = None  # Just initialize with something dummy
#            for node in nodes:
#                if node not in ant.tour:
#                    # Add note to tour
#                    N = node
#                    ant.tour[step] = N
#                    ant.visited[N] = 1  # Mark as visited=True
#
#                    if verbose:
#                        print(" Newly connected node: {0}".format(N))
#
#            if N is None:
#                print("WARNING! Something is really fucked...")
#                return
            if verbose:
                print("  Selected arc={0} ({1})".format(a, self.base_graph[a]))

            N = None
            if nodes[0] in ant.tour:
                N = nodes[1]
                # Just check if the other node is in tour
#                if nodes[1] in ant.tour:
#                    print("ERROR! Both nodes in tour!")
#                else:
#                    print("OK!")
            else:
                N = nodes[0]
                # Just check if the other node is in tour
#                if nodes[1] in ant.tour:
#                    print("OK!")
#                else:
#                    print("ERROR! Neither node in tour!")

            if N is None:
                print("  WARNING! Something is really fucked...")
                return

            ant.tour[step] = N
            ant.visited[N] = 1  # Mark as visited=True

            # Identify adjacent arcs to node N in base graph
            arcs_connected = self.adjacent(N, self.base_graph)

            if verbose:
                print("  Newly node N={0}".format(N))
                print("  Adj arcs to N={0}: {1}".format(N, arcs_connected))

            if a in arcs_connected:
                # Excluding the previous choosen arc, a
                arcs_connected.remove(a)

            # Update AA, by removing arc a and any newly infeasible arcs

            # Remove the newly connected arc from list
            AA.remove(a)
            if verbose:
                print("  Removing selected arc: {0}".format(a))

            dummy = 0
            for arc in arcs_connected:
                # Is this arc in the adjacent arcs (AA) already?
                if arc in AA:
                    # Remove arc from list, as tree is now such that adding
                    # this arc would cause a loop
                    if verbose:
                        print("  Remove infeasible arc: {0} ({1})".format(arc,
                              self.base_graph[arc]))
                    AA.remove(arc)
                else:
                    # Are both end nodes of this arc in the ant tree?
                    arc_nodes = self.get_nodes(self.base_graph[arc])
                    if arc_nodes[0] in ant.tour and arc_nodes[1] in ant.tour:
                        # Do nothing, leave list unaltered, as adding this
                        # arc would cause a loop in the tree
                        dummy += 1
                    else:
                        # As there are not other criteria (such as direction)
                        # add the arc to the list of adjacent arcs to the
                        # current tree
                        AA.append(arc)
                        if verbose:
                            print("  Add feasible arc: {0} ({1})".format(arc,
                                  self.base_graph[arc]))
            # End of step-by-step tree construction
        # For testing purposes only, this is not the right place for computing
        # the network length, please do this outside
#        self.compute_network_length(ant)
#        print(ant)
#        self.plot_tree(ant)
        return

    def compute_tour_length(self, ant):
        """ Compute tour length

        Compute the length of the Ant tour using the node information

        :param ant: Ant object to compute tour length
        :return: tour length
        """
        length = 0.0
        for i in range(self.n):
            # This works because tour length = n + 1
            c1 = ant.tour[i]    # city 1
            c2 = ant.tour[i+1]  # city 2
            x1 = self.x[c1]
            y1 = self.y[c1]
            x2 = self.x[c2]
            y2 = self.y[c2]
            length += self.func(x1, y1, x2, y2)  # Distance function
            # print("  Distance {0}-{1}: {2}".format(c1, c2, length))
        ant.tour_length = length
        return ant.tour_length

    def compute_network_length(self, ant):
        """ Compute network length

        Compute the length of the Ant tour using the arc (edge) information

        :param Ant ant: the Ant object to compute the network length
        :return: float tour length
        """
        length = 0.0
        for a in ant.tree:
            # Get the string representation of the arc
            arc = self.base_graph[a]
            # Get the nodes of the arc
            nodes = self.get_nodes(arc)
            j = nodes[0]  # Initial node
            k = nodes[1]  # End node
            x1 = self.x[j]
            x2 = self.x[k]
            y1 = self.y[j]
            y2 = self.y[k]
            length += self.func(x1, y1, x2, y2)  # Distance function
        ant.tour_length = length
        return length

    def as_decision_rule(self, k, i):
        """ AS decision rule

        The ants apply the Ant System (AS) action choice rule eq. 3.2

        :param int k: ant identifier
        :param int i: counter for construction step
        """
        c = self.colony[k].tour[i-1]  # current city
        # Sum the probabilities of the cities not yet visited
        sum_probabilities = 0.0
        selection_probability = np.zeros(self.n, dtype=np.float)
        for j in range(self.n):
            # Check if city has already been visited
            if self.colony[k].visited[j]:
                selection_probability[j] = 0.0
            else:
                selection_probability[j] = self.choice_info[c][j]
                if self.choice_info[c][j] < 0:
                    print(" Warning: ChoiceInfo[{0}][{1}] is {2}".format(
                            c, j, self.choice_info[c][j]))
                sum_probabilities += selection_probability[j]
        # Random number from the interval [0, sum_probabilities), this number
        # correspond to a Uniform Distribution
        r = (sum_probabilities - 0) * np.random.random_sample() + 0
#        print("\n  Sum prob. not visited: {0}".format(sum_probabilities))
#        print("  Random probability:    {0}".format(r))
#        print("  Select. probabilities: {0}".format(selection_probability))
#        print("  Sum of sel. prob.:  {0}".format(sum(selection_probability)))
        j = 0
        p = selection_probability[j]
        while p < r:
            j += 1
            p += selection_probability[j]
#        print("  p: {0} < r: {1} ({2})".format(p, r, p < r))
        self.colony[k].tour[i] = j
        self.colony[k].visited[j] = 1  # True
        return

    def neighbor_list_as_decision_rule(self, k, i):
        """ Neighbor list AS decision rule

        The ants apply the Ant System (AS) action choice rule eq. 3.2, adapted
        to exploit candidate lists.

        :param k: ant identifier
        :param i: counter for construction step
        """
        c = self.colony[k].tour[i-1]  # current city
        sum_probabilities = 0.0
        selection_probability = np.zeros(self.n, dtype=np.float)
        for j in range(self.nn_ants):
            if self.colony[k].visited[self.nn_list[c][j]]:
                selection_probability[j] = 0.0
            else:
                idx = self.nn_list[c][j]
                selection_probability[j] = self.choice_info[c][idx]
                sum_probabilities += selection_probability[j]
        if sum_probabilities == 0:
            self.choose_best_next(k, i)
        elif sum_probabilities < 0:
            # This should never be reached but still...
            print("Sum probabilities is lower than zero")
            self.choice_best_next(k, i)
        else:
            # Random number (Uniform) from interval [0, sum_probabilities)
            r = (sum_probabilities - 0) * np.random.random_sample() + 0
            j = 0
            p = selection_probability[j]
            while p < r:
                j += 1
                p += selection_probability[j]
            self.colony[k].tour[i] = self.nn_list[c][j]
            self.colony[k].visited[self.nn_list[c][j]] = 1  # True
        return

    def choose_best_next(self, k, i):
        """ Choose best next

        Identify the city with maximum value of tau^alpha * eta^beta

        :param k:
        :param i:
        :return:
        """
        v = 0.0
        c = self.colony[k].tour[i - 1]  # current city
        nc = -1
        for j in range(self.n):
            if not self.colony[k].visited[j]:
                if self.choice_info[c][j] > v:
                    nc = j  # city with maximal tau^alpha * eta^beta
                    v = self.choice_info[c][j]
        # TODO: What happens if nc is -1?
        self.colony[k].tour[i] = nc
        self.colony[k].visited[nc] = 1  # True
        return

    def as_pheromone_update(self):
        """ AS pheromone update

        Update the pheromone trace of the ants
        """
        self.evaporate()
        for ant in self.colony:
            self.deposit_pheromone(ant)

#        for i in range(self.ants):
#            self.deposit_pheromone(i)

        self.compute_choice_information()
        return

    def evaporate(self):
        """ Evaporate

        Evaporate the pheromone trail of the ants
        """
        self.pheromone = self.pheromone * (1.0 - self.rho)
        return

    def deposit_pheromone(self, ant):
        """ Deposit pheromone

        Update the pheromone trail for the cities in the ant's tour

        :param Ant ant: the Ant instance that will be depositing pheromone
        """
        delta = 1.0 / ant.tour_length
        if self.type == 'TSP':
            # This is intended for the symmetric TSP
            for i in range(self.n):
                j = ant.tour[i]
                k = ant.tour[i+1]

                # Deposit pheromone, assumming symmetric problem
                self.pheromone[j][k] = self.pheromone[j][k] + delta
                self.pheromone[k][j] = self.pheromone[j][k]
        else:
            # Get each arc in the tree-like network
            for a in ant.tree:
                # Get the string representation of the arc
                arc = self.base_graph[a]
                # Get the nodes of the arc
                nodes = self.get_nodes(arc)
                j = nodes[0]  # Initial node
                k = nodes[1]  # End node

                # Deposit the pheromone, assuming symmetric problem
                self.pheromone[j][k] = self.pheromone[j][k] + delta
                self.pheromone[k][j] = self.pheromone[j][k]

#    def deposit_pheromone(self, k):
#        """ Deposit pheromone
#
#        Update the pheromone trail for the cities in the ant's tour
#        This is intended for the symmetric TSP.
#
#        :param k: the ant identifier
#        """
#        delta = 1.0 / self.colony[k].tour_length
#        for i in range(self.n):
#            j = self.colony[k].tour[i]
#            l = self.colony[k].tour[i+1]
#            self.pheromone[j][l] = self.pheromone[j][l] + delta
#            self.pheromone[l][j] = self.pheromone[j][l]  # Symmetric problem
#        return

    def compute_choice_information(self):
        """ Compute choice information

        Compute the choice information matriz using the pheromone and
        heuristic information

        """
        self.choice_info = self.pheromone ** self.alpha * self.eta ** self.beta
        return

    def find_best(self):
        """ Find best

        Find the best ant of current iteration
        """
        best_ant = self.colony[0]
        for ant in self.colony:
            if ant.tour_length < best_ant.tour_length:
                best_ant = ant.clone()
        return best_ant

if __name__ == "__main__":

    # Problem instance
    file_name = "../networks_design/data/network12.csv"

    # Parameters for Ant System
    m = 12
    rho = 0.02
    alpha = 1.0
    beta = 2.0
    nn_ants = 20
    max_iters = 30

    # Test a tree-like network layout optimization problem

    problem = 'TREE_NET'
    # Create & run the ACO object
    aco1 = ACO(m, file_name, rho, alpha, beta, nn_ants, max_iters,
               use_base_graph=False, instance_type=problem)
    best = aco1.run()
    print("\nBest overall solution:")
    print(best)
    aco1.plot_tree(best)

    # Test a symmetric TSP problem

#    problem = 'TSP'
#    tsp = ACO(m, file_name, rho, alpha, beta, nn_ants, max_iters)
#    best = tsp.run()
#    print("\nBest overall solution:")
#    print(best)
#    tsp.plot_best_tour()
