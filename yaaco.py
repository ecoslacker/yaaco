#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
yaaco.py

Yet Another Ant Colony Optimization Python Implementation

An attempt to code the Ant Colony Optimization (ACO) metaheuristic to solve
the Traveling Salesman Problem (TSP) in Python 2.7 language.

IMPORTANT: This code only includes AS, EAS, AS-Rank and MAX-MIN AS algorithms.
Others are pending to be included.

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
from operator import attrgetter
from datetime import datetime
from math import sqrt, pow
from matplotlib.path import Path


# To print complete arrays
np.set_printoptions(threshold='nan')

MAXFACTOR = 3
EPSILON = 1e-6


class Ant:
    """ Single ant

    Creates a single ant with its properties

    :param int size: the dimension or length of the ant
    :param str atype: the type of problem instance
    """

    uid = 0

    def __init__(self, size, atype='Symmetric TSP'):
        """ Initialize the Ant object """
        assert type(size) is int, "The Ant size should be integer type"
        self.size = size
        self.ant_type = atype
        self.uid = self.__class__.uid
        self.__class__.uid += 1
        self.tour_length = np.inf

        self.tour = np.ones(self.size, dtype=np.int64) * -1
        self.visited = np.zeros(self.size, dtype=np.int64)

        self.tour = np.ones(self.size+1, dtype=np.int64) * -1

    def __str__(self):
        """ String representation of the Ant object """
        text = "Ant:\n"
        text += " UID:     " + str(self.uid) + "\n"
        text += " Type:    " + str(self.ant_type) + "\n"
        text += " Tour:    " + str(self.tour) + "\n"
        text += " Visited: " + str(self.visited) + "\n"
        text += " Tour length: " + str(self.tour_length) + "\n"
        return text

    def __len__(self):
        """ The ant's tour length """
        return len(self.tour)

    def clone(self):
        """ Returns a deep copy of the current Ant instance with a new UID

        :return Ant ant: instance with same properties
        """
        ant = Ant(len(self.tour), self.ant_type)
        ant.tour_length = self.tour_length
        ant.tour = self.tour.copy()
        ant.visited = self.visited.copy()
        return ant


class Problem:
    """ Problem instance

    :param str filename: a tex file with the data of the problem instance
    :param callable func: a function to calculate the distance
    :param str name: the name of the problem (default is "Problem#")

    Possible instances include:
        TSP:       Symmetric Traveling Salesman Problem
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
        self.type = kwargs.get('ptype', 'Symmetric TSP')

        # Initialize some variables
        self.file_graph = ''

        # WARNING! This will overwrite the name of the problem
        self.x, self.y, self.name = self.read_instance()

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

        self.diameters = []

    def read_instance_tsp(self):
        """ Read instance TSP

        Reads the problem instance from a text delimited file, the file must
        be formatted as the *.tsp type as described in TSPLIB

        :return x, the x-axis coordinates of the instance
        :return y, the y-axis coordinates of the instance
        :return name, the name of the instance
        """

        x = []
        y = []
        name = ''

        try:
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
                        # Remove all the blank ocurrences
                        r = filter(lambda x: x != '', row)
                        assert len(r) == 3, "Incorrect format data"
                        x.append(float(r[1]))
                        y.append(float(r[2]))
        except IOError as e:
            z = e
            print(z)
        return x, y, name

    def read_instance_csv(self):
        """ Read instance

        Reads the problem instance from a text delimited file, the file must
        be formatted as Comma Separated Values text file with extensions *.csv
        or *.txt

        :return x, the x-axis coordinates of the instance
        :return y, the y-axis coordinates of the instance
        :return name, the name of the instance
        """
        x = []
        y = []
        name = ''

        try:
            with open(self.file_instance, 'r') as f:
                reader = csv.reader(f, delimiter=',')
                for row in reader:
                    if len(row) < 2:
                        break
                    if len(row) == 2:
                        x.append(float(row[0]))
                        y.append(float(row[1]))
                    elif len(row) > 2:
                        x.append(float(row[1]))
                        y.append(float(row[2]))
        except IOError as e:
            z = e
            print(z)

        return x, y, name

    def read_instance(self):
        """ Read instance

        Reads the problem instance from a text delimited file

        :return x, the x-axis coordinates of the instance
        :return y, the y-axis coordinates of the instance
        :return name, the name of the instance
        """

        x = []
        y = []
        name = ''
        ext = self.file_instance[-4:]  # File extension

        # Identify the text file
        if ext == '.tsp':
            return self.read_instance_tsp()
        elif ext == '.csv' or ext == '.txt':
            return self.read_instance_csv()
        else:
            print('Cannot indetify file type, trying as CSV')
            return self.read_instance_csv()

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

    def nn_tour_tsp(self, ttype='TSP'):
        """ A TSP tour generated by the nearest-neighbor heuristic

        :param ttype, tour type could be 'TSP' or another special type
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
            # plt.xlim([min(self.x) - 50, max(self.x) + 50])
            # plt.ylim([min(self.y) - 50, max(self.y) + 50])
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

    :param str filename: a text file with the data of the problem instance
    :param int ants: number of ants in the colony
    :param int nn_ants: number of ants in the nearest-neighbor
    :param float rho: the pheromone evaporation parameter
    :param float alpha: the pheromone trail influence
    :param float beta: the heuristic information influence
    :param int max_iters: maximum number of iterations of the algorithm
    :param str flag: type of algorithm to be used in the ACO metaheuristic
    :param bool use_base_graph: True to use data from a base graph file
    :param str instance_type: Description of problem instance (Default 'TSP')

    Possible flags include:
        AS: Ant System (Default)
        EAS: Elitist Ant System
        RAS: Rank-Based Ant System
        MMAS: Max-Min Ant System
    """

    def __init__(self, filename, **kwargs):
        """ Initialize

        Set parameters and initializes pheromone trails
        """

        self.start = datetime.now()

        self.FLAGS = ['AS', 'EAS', 'RAS', 'MMAS']

        assert type(filename) is str, "File name should be a string"
        assert filename != '', "Empty file name!"

        # Initialize class variables from arguments
        self.ants = kwargs.get('ants', -1)
        self.alpha = kwargs.get('alpha', 1.0)
        self.beta = kwargs.get('beta', 2.0)
        self.rho = kwargs.get('rho', 0.5)
        self.nn_ants = kwargs.get('nn_ants', 20)
        self.max_iters = kwargs.get('max_iters', 100)
        self.flag = kwargs.get('flag', 'AS')           # Flag: Ant System
        function = kwargs.get('function', None)        # Distance function
        instance_type = kwargs.get('instance_type', 'Symmetric TSP')

        # Initialize the Problem instance
        #
        # This allows the ACO class to use functions and variables such as
        # dimension, nn_ants, compute_nearest_neighbor(), etc...
        Problem.__init__(self, filename, function, ptype=instance_type,
                         name="TSP 0")
        self.n = self.dimension       # dimension of the problem

        # If not provided, initialize the ants value to dimension problem
        if self.ants == -1:
            self.ants = self.n

        assert type(self.ants) is int, "The number of ants should be integer"
        assert self.ants > 0, "Number of ants should be greater than zero"
        assert self.flag in self.FLAGS, "Unknown flag"

        # Enable the use of nn_list if n > nn_ants
        if self.n > self.nn_ants:
            self.use_nn = True
            print(' n={0} > nn_ants={1} (True)'.format(self.n, self.nn_ants))
        else:
            self.use_nn = False
            print(' n={0} <= nn_ants={1} (False)'.format(self.n, self.nn_ants))

        # Get the initial TSP tour generated by nearest-neighbor heuristic
        # from a special ant for this purpose
        self.Cnn = self.compute_tour_length(self.ant)

        # Initial pheromone trail and other default parameters
        self.note = "Initial iteration"
        self.found_best = 0  # Iteration in which best solution is found
        self.restart_best_ant = Ant(self.dimension, self.type)
        self.restart_found_best = 0  # Iter in which restart_best_ant is found

        self.trail_0 = self.ants / self.Cnn
        if self.flag is 'MMAS':
            self.trail_0 = 1.0 / (self.rho * self.Cnn)
            self.u_gb = 1  # every u_gb iterations update with best-so-far ant
            self.restart_iteration = 1
        elif self.flag is 'EAS':
            self.trail_0 = 1.0 / (self.rho * self.Cnn)
            self.elitist_ants = self.ants
        elif self.flag is 'RAS':
            self.ras_ranks = 6  # This 'magic' value is from literature
            assert self.ras_ranks < self.ants, "ras-ranks >= n_ants"

        self.create_colony(instance_type)

        # Initialize pheromone trail matrix
        self.reset_pheromone(self.trail_0)

        # Heuristic information (eta)
        self.eta = 1.0 / (self.distance_matrix + 0.1)
        # Add a small constant to avoid division by zero if a distance is zero

        # Initialize the choice information matrix & compute it
        self.choice_info = np.empty_like(self.pheromone)
        self.choice_info[:] = self.pheromone
        self.compute_choice_information()

        # Pheromone trails are initialized to upper pheromone trail limit
        self.trail_max = 1.0 / (self.rho * self.Cnn)
        self.trail_min = self.trail_max / (2.0 * self.n)

        # Initialize the variables concerning statistics
        self.iteration = 0
        self.best_so_far_ant = Ant(self.n)
        self.best_so_far_ant.tour_length = np.inf
        self.show_initialization(False, False)

        # Plot the base graph
        # self.plot_base_graph()

        # Measure initialization time
        self.exec_time = 0
        self.init_time = datetime.now() - self.start
        print("Initialization time: {0}".format(self.init_time))

    def __str__(self):
        """ str

        A string representation of the ant colony
        """

        text = "Initialization parameters:\n"
        text += "  n_ants:    {0}".format(self.ants) + "\n"
        text += "  nn_ants:   {0}".format(self.nn_ants) + "\n"
        text += "  alpha:     {0}".format(self.alpha) + "\n"
        text += "  beta:      {0}".format(self.beta) + "\n"
        text += "  rho:       {0}".format(self.rho) + "\n"
        text += "  max_iters: {0}".format(self.max_iters) + "\n"
        text += "  flag:      {0}".format(self.flag) + "\n"
        text += "  init tour: {0} (length)".format(self.Cnn) + "\n"
        if self.flag == 'MMAS':
            text += "  trail max: {0}".format(self.trail_max) + "\n"
            text += "  trail min: {0}".format(self.trail_min) + "\n"
        text += "  trail 0:   {0}".format(self.trail_0) + "\n"
        text += "  dimension: {0}".format(self.n) + "\n"
        text += "Initial ant colony:\n"
        for ant in self.colony:
            text += "{0}".format(ant) + "\n"
        return text

    def show_initialization(self, matrices=True, plot=False):
        """ Show initialization

        Shows the default parameters
        :return: None
        """
        print("Initialization parameters:")
        print("  n_ants:    {0}".format(self.ants))
        print("  nn_ants:   {0}".format(self.nn_ants))
        print("  alpha:     {0}".format(self.alpha))
        print("  beta:      {0}".format(self.beta))
        print("  rho:       {0}".format(self.rho))
        print("  max_iters: {0}".format(self.max_iters))
        print("  flag:      {0}".format(self.flag))
        print("  init tour: {0} (length)".format(self.Cnn))
        if self.flag == "MMAS":
            print("  trail max: {0}".format(self.trail_max))
            print("  trail min: {0}".format(self.trail_min))
        print("  trail 0:   {0}".format(self.trail_0))
        print("  dimension: {0}".format(self.n))
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

    def plot_nodes(self, filename=''):
        """ Plot nodes

        Plots the nodes of the problem

        :param str filename: path and file name to save the plot figure
        """
        fig1 = plt.figure()
        fig1.clear()

        plt.title("Problem: " + self.name)
        plt.scatter(self.x, self.y, c='black')

        labels = ['{0}'.format(l) for l in range(len(self.x))]
        for label, lx, ly in zip(labels, self.x, self.y):
            plt.annotate(label, xy=(lx, ly), xytext=(-5, 5),
                         textcoords='offset points')
        plt.axis('equal')
        # Save the plot to a PNG file
        if filename != '':
            plt.savefig(filename, bbox_inches='tight', dpi=150)
        plt.show()

    def plot_best_tour(self, filename=''):
        """ Plot best tour

        Plot the complete tour of the best_so_far_ant instance. If called at
        the end of run() this will plot the best overall tour.

        :param Ant ant: an instance of the Ant class
        :param str filename: path and file name to save the plot figure
        """
        figBestTour = plt.figure()
        ax = figBestTour.add_subplot(1, 1, 1)

        plt.scatter(self.x, self.y, c='black')
        plt.xlabel('x')
        plt.ylabel('y')

        labels = ['{0}'.format(l) for l in range(len(self.x))]
        for label, lx, ly in zip(labels, self.x, self.y):
            plt.annotate(label, xy=(lx, ly), xytext=(-5, 5),
                         textcoords='offset points')

        length = self.best_so_far_ant.tour_length
        plt.title('Best TSP tour (length={0:.2f})'.format(length))
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
            ax.add_patch(patches.PathPatch(path, color='black', lw=0.5))
            # plt.xlim([min(self.x) - 50, max(self.x) + 50])
            # plt.ylim([min(self.y) - 50, max(self.y) + 50])
        # Save the plot to a file
        if filename != '':
            plt.savefig(filename, bbox_inches='tight', dpi=150)
        plt.show()

    def run(self):
        """ Run

        Call this function to actually run the ACO metaheuristic. All the ants
        (Ant instances) will construct solutions and update pheromone while
        the termination criteria aren't satisfied.

        :return best_so_far_ant: Ant instance with the best tour
        """

        print("*** Running Ant Colony Optimization ***")
        # 1. Initialize data:
        # Data initialization was already done in the __init__ function

        # 2. Loop
        print("Iter\tTour len\tNote")
        while not self.termination_criteria():
            self.tsp_construct_solutions()

            # Local search step is optional, nothing to do here!

            # Update statistics
            self.update_statistics()

            # Update pheromone trails
            self.pheromone_trail_update()

            # Search control and pheromone trail re-initialization
            self.search_control()

            # Console output
            print("{0}\t{1}\t{2}".format(self.iteration,
                  self.best_so_far_ant.tour_length, self.note))

            self.iteration += 1
            self.note = ""

        # Measure execution time
        self.exec_time = datetime.now() - self.start
        print("Execution time: {0}".format(self.exec_time))
        print("Best solution found at iteration: {0}".format(self.found_best))
        return self.best_so_far_ant

    def termination_criteria(self):
        """ Termination criteria

        Terminate when at least one of the following conditions are True:
        1) The current iteration is greather than the maximum allowed
           iterations

        :return bool: True if the termination criteria are satisfied
        """
        return self.iteration > self.max_iters

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
                if self.use_nn:
                    self.neighbor_list_as_decision_rule(k, step)
                else:
                    self.as_decision_rule(k, step)
        # 4. Move to initial city and compute each ant's tour length
        for ant in self.colony:
            ant.tour[-1] = ant.tour[0]
            self.compute_tour_length(ant)
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

    def update_statistics(self):
        """ Update statistics
        Manage some statistical information about the trial, especially
        if a new best solution (best-so-far or restart-best) is found and
        adjust some parameters if a new best solution is found

        Side effects: restart-best and best-so-far ant may be updated;
        trail_min and trail_max used by MMAS may be updated
        """
        iter_best_ant = self.find_best()

        # Update best so far ant
        # if iter_best_ant.tour_length < self.best_so_far_ant.tour_length:
        diff = self.best_so_far_ant.tour_length - iter_best_ant.tour_length
        if diff > EPSILON:
            self.best_so_far_ant = iter_best_ant.clone()
            self.restart_best_ant = iter_best_ant.clone()

            self.found_best = self.iteration
            self.restart_found_best = self.iteration

            # Update min and max pheromone trails
            if self.flag is 'MMAS':
                # Warning! This code doesn't consider local search
                self.trail_max = 1. / (self.rho *
                                       self.best_so_far_ant.tour_length)
                self.trail_min = self.trail_max / (2. * self.n)
                self.trail_0 = self.trail_max

            # print("Best so far ant found:")
            # print(self.best_so_far_ant)
            self.note += "New best ant found. "

        # Update restart best ant
        # if iter_best_ant.tour_length < self.restart_best_ant.tour_length:
        diff = self.restart_best_ant.tour_length - iter_best_ant.tour_length
        if diff > EPSILON:
            self.restart_best_ant = iter_best_ant.clone()
            self.restart_found_best = self.iteration
            self.note += "Restart best ant found (UID:{0}, {1}). ".format(
                    self.restart_best_ant.uid,
                    self.restart_best_ant.tour_length)
            # print("Restart best ant found:")
            # print(self.restart_best_ant)
        return

    def as_decision_rule(self, k, i):
        """ AS decision rule

        The ants apply the Ant System (AS) action choice rule eq. 3.2

        :param int k: ant identifier
        :param int i: counter for construction step
        """
        c = self.colony[k].tour[i-1]  # current city

        # Create a roulette wheel, like in evolutionary computation
        # Sum the probabilities of the cities not yet visited
        sum_probabilities = 0.0
        selection_probability = np.zeros(self.n, dtype=np.float)
        for j in range(self.n):
            if self.colony[k].visited[j]:
                # If city has been visited, its probability is zero
                selection_probability[j] = 0.0
            else:
                # Assign a slice to the roulette wheel, proportional to the
                # weight of the associated choice
                selection_probability[j] = self.choice_info[c][j]
                if self.choice_info[c][j] < 0:
                    print(" Warning: ChoiceInfo[{0}][{1}] is {2}".format(
                            c, j, self.choice_info[c][j]))
                sum_probabilities += selection_probability[j]

        # Spin the roulette wheel
        # Random number from the interval [0, sum_probabilities), this number
        # correspond to a Uniform Distribution
        r = (sum_probabilities - 0) * np.random.random_sample() + 0

        # print("\n  Sum prob. not visited: {0}".format(sum_probabilities))
        # print("  Random probability:    {0}".format(r))
        # print("  Select. probabilities: {0}".format(selection_probability))
        # print("  Sum of sel. prob.:  {0}".format(sum(selection_probability)))
        j = 0
        p = selection_probability[j]
        while p < r:
            j += 1
            p += selection_probability[j]
        # print("  p: {0} < r: {1} ({2})".format(p, r, p < r))

        # The ant moves to the choosen city j
        self.colony[k].tour[i] = j
        self.colony[k].visited[j] = 1  # True
        return

    def neighbor_list_as_decision_rule(self, k, i):
        """ Neighbor list AS decision rule

        The ants apply the Ant System (AS) action choice rule eq. 3.2,
        adapted to exploit candidate lists.

        :param k: ant identifier
        :param i: counter for construction step
        """
        # Choose best city from candidate list nn_list
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

        # All the cities in nn_list have been visited, con sum_probabilities
        # is zero, in this case choose the best city outside nn_list.
        if sum_probabilities <= 0:
            # print("Sum probabilities is equal (or lower) than zero.")
            self.choose_best_next(k, i)
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

        Identify the city with maximum value of tau^alpha * eta^beta, this
        function is called when all the cities in the candidate list nn_list
        are already visited.

        :param k:
        :param i:
        :return:
        """
        v = 0.0
        c = self.colony[k].tour[i - 1]  # current city
        nc = 0

        # Algorithm to find the maximum value
        for j in range(self.n):
            if not self.colony[k].visited[j]:
                if self.choice_info[c][j] > v:
                    nc = j  # city with maximal tau^alpha * eta^beta
                    v = self.choice_info[c][j]

        self.colony[k].tour[i] = nc
        self.colony[k].visited[nc] = 1  # True
        return

    def pheromone_trail_update(self):
        """ Pheromone trail update

        Pheromone trails are evaporated and pheromones are deposited
        according to the rules defined by the various ACO algorithms.

        Comprises two pheromone update procedures: pheromone evaporation and
        pheromone deposit. Additionally, the procedure
        compute_choice_information() computes the matrix choice_info to be
        used in the next algorithm iteration.
        """

        self.evaporate()
        if self.flag is 'AS':
            self.as_pheromone_update()
        elif self.flag is 'EAS':
            self.eas_pheromone_update()
        elif self.flag is 'RAS':
            self.ras_pheromone_update()
        elif self.flag is 'MMAS':
            self.mmas_pheromone_update()
            self.check_pheromone_trail_limits()

        self.compute_choice_information()
        return

    def as_pheromone_update(self):
        """ Ant System pheromone update

        All the ants deposit pheromone to the matrix.
        """
        for ant in self.colony:
            self.deposit_pheromone(ant)
        return

    def eas_pheromone_update(self):
        """ Elitist Ant System pheromone update

        Manages pheromone update for the Elitist Ant System, all ants plus
        elitist ant deposit pheromones on matrix.
        """
        for ant in self.colony:
            self.deposit_pheromone(ant)
        self.deposit_pheromone_weighted(self.best_so_far_ant,
                                        float(self.elitist_ants))
        return

    def ras_pheromone_update(self):
        """ Rank-based Ant System pheromone update

        Manage global pheromone deposit for Rank-based Ant System
        """
        # Sort the Ants by tour length in ascending order
        sorted_ants = sorted(self.colony, key=attrgetter('tour_length'))
        # Only the first 'ras_ranks' ants (including best_so_far_ant) are
        # allowed to deposit pheromone
        for i in range(self.ras_ranks - 1):
            self.deposit_pheromone_weighted(sorted_ants[i],
                                            float(self.ras_ranks - i - 1))
        self.deposit_pheromone_weighted(self.best_so_far_ant,
                                        float(self.ras_ranks))
        return

    def mmas_pheromone_update(self):
        """ MAX-MIN Ant System pheromone update

        Manage global pheromone deposit for MAX-MIN Ant System
        """
        # Update with iteration_best_ant
        if (self.iteration % self.u_gb):
            iter_best_ant = self.find_best()
            self.deposit_pheromone(iter_best_ant)
        else:
            # Every u_gb iteration update with best_so_far_ant or with
            # restart_best_ant, according to next rule:
            no_improv = self.iteration - self.restart_found_best
            if (self.u_gb == 1 and no_improv > 50):
                # print("No improvement by 50 gens.")
                self.note = "No improvement by 50 gens."
                self.deposit_pheromone(self.best_so_far_ant)
            else:
                self.deposit_pheromone(self.restart_best_ant)
        return

    def check_pheromone_trail_limits(self):
        """ Check pheromone trail limits

        Only for MMAS without local search: keeps pheromone trails inside
        trail limits. Pheromones are forced to interval [trail_min,trail_max]
        """
        self.pheromone = np.clip(self.pheromone, self.trail_min,
                                 self.trail_max)
        return

    def evaporate(self):
        """ Evaporate

        Decreases the values of the pheromone trails on all the arcs by a
        constant factor rho. This uses matrix operations
        """
        self.pheromone = self.pheromone * (1.0 - self.rho)
        return

    def deposit_pheromone(self, ant):
        """ Deposit pheromone

        Adds pheromone to the arcs belonging to the tours constructed by
        the ant.

        :param Ant ant: the Ant instance that will be depositing pheromone
        """
        delta = 1.0 / ant.tour_length

        # This is intended for the symmetric TSP
        for i in range(self.n):
            j = ant.tour[i]
            k = ant.tour[i+1]

            # Deposit pheromone, assumming symmetric problem
            self.pheromone[j][k] = self.pheromone[j][k] + delta
            self.pheromone[k][j] = self.pheromone[j][k]
        return

    def deposit_pheromone_weighted(self, ant, weight):
        """ Deposit pheromone weighted

        Adds pheromone to the arcs belonging to the tours constructed by
        the ant, using a weight factor.

        :param Ant ant: the Ant instance that will be depositing pheromone
        :param float weight: the weight factor to update the pheromone
        """
        assert type(weight) is float, "Weight factor should be float"

        delta = weight / ant.tour_length

        # This is intended for the symmetric TSP
        for i in range(self.n):
            j = ant.tour[i]
            k = ant.tour[i+1]

            # Deposit pheromone, assumming symmetric problem
            self.pheromone[j][k] = self.pheromone[j][k] + delta
            self.pheromone[k][j] = self.pheromone[j][k]
        return

    def compute_choice_information(self):
        """ Compute choice information

        Compute the choice information matrix using the pheromone and
        heuristic information

        """
        self.choice_info = self.pheromone ** self.alpha * self.eta ** self.beta
        return

    def find_best(self):
        """ Find best

        Find the best Ant object from the colony in the current iteration

        :return: best ant, Ant object with the shortest tour length
        """
        best_ant = self.colony[0]
        for ant in self.colony:
            if ant.tour_length < best_ant.tour_length:
                best_ant = ant.clone()
        return best_ant

    def search_control(self):
        """ Search control

        Occasionally compute some statistics and check whether or not if the
        algorithm is converged.

        Side effects: restart_best and best_so_far_ant may be updated;
        trail_min and trail_max used by MMAS may be updated.
        """
        if self.flag is 'MMAS':
            if not (self.iteration % 100):
                # MAX-MIN Ant System was the first ACO algorithm to use
                # pheromone trail re-initialisation as implemented here,
                # Other ACO algorithms may also profit from this mechanism.
                # NOTE: remember for MMAS trail_0 == trail_max

                # print("Restart point was reached")
                self.note += "Restart point was reached."
                self.restart_best_ant.tour_length = np.inf
                self.reset_pheromone(self.trail_0)
                self.compute_choice_information()
                self.restart_iteration = self.iteration
        return


if __name__ == "__main__":

    start = datetime.now()
    f = '%Y_%m_%d_%H_%M_%S'  # Date format

    # The name of the problem to solve, should use in *.tsp or *.csv format
    prob = 'eil51.tsp'

    # Save best tour & solution, WARNING: directories should exist!
    save_plot = 'results/' + prob + '/' + datetime.strftime(start, f) + '.png'
    save_best = 'results/' + prob + '/' + datetime.strftime(start, f) + '.txt'

    # **** Problem instance data (TSP coordinates file) ****
    instance = 'test_data/' + prob

    # Create the ACO object & run
    tsp_aco = ACO(instance, max_iters=500, flag='MMAS')
    tsp_aco.plot_nodes()
    best = tsp_aco.run()

    # Show the results
    print("\nBest overall solution:")
    print(best)

    # Save the results
    with open(save_best, 'w') as f:
        f.write('{0}\n'.format(tsp_aco))
        f.write('Best overall solution:\n{0}\n'.format(best))
        f.write('Initialization time: {0}\n'.format(tsp_aco.init_time))
        f.write('Execution time: {0}\n'.format(tsp_aco.exec_time))
    tsp_aco.plot_best_tour(save_plot)
