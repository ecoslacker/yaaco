# YAACO
Yet Another Ant Colony Optimization Python Implementation

An attempt to code the Ant Colony Optimization (ACO) metaheuristic in
Python 2.7 language.

To understand what this code does you should probably read the recommended
references and bibliography.

## Problems

At the moment this framework only works for two kinds of problems:
* Symmetric TSP (Traveling Salesman Problem)
* Tree-like layout networks optimization

## ACO metaheuristic

At the moment this code only includes partial support for AS and MMAS
algorithms, there are more algorithms in the ACO family that are not yet
available. They will be added to this code eventually, just be patient and
check the project change log.

## Usage

To use this code you should create an ACO instance with the proper data and
parameters, then use the function called **run** to actually execute the
metaheuristic algorithms.

The problem instance data should be in a text file formatted as indicated in
the TSPLIB documentation for the symmetric TSP.

## Requirements

This framework requires:
* Python 2.7
* Numpy
* Matplotlib

NOTE: For people new to Python, it is recommended to install a distribution
like **Anaconda** or **Python(x,y)** in order to get the requirements easily.
Be sure to install a Python 2 (not Python 3) instance.

## Installation

This framework does not require installation, just copy the directory and run
the main script from a Python interpreter.

## References

*  Dorigo, M., & Stützle, T. (2004). Ant colony optimization. Massachusetts,
  United States of America: Massachusetts Institute of Technology.

## License

None
