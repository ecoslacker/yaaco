# yaaco.py
Yet Another Ant Colony Optimization Python Implementation

An attempt to code the Ant Colony Optimization (ACO) metaheuristic to solve
the Traveling Salesman Problem (TSP) in Python 2.7 language.

To understand what this code, first you should probably read the recommended
references and bibliography.

## Problems

At the moment this script only works to solve the following problems:
* Symmetric TSP (Traveling Salesman Problem)

## ACO metaheuristic

At the moment this code only includes support for Ant System (AS), Elitist Ant
System (EAS), Rank-Based Ant System (RAS) and MAX-MIN Ant System (MMAS) algorithms.
There are more algorithms in the ACO family that are not yet available.
Also, local search is not supported at the moment.
They will be added to this code eventually, just be patient and check the project change log.

The default algorithm is AS, to use another algorithm the `flag` variable should be used.
User should assing values to the algorithm's parameters, otherwise default values acording to literature are used.

This code is closely based in the original [ACOTSP](http://www.aco-metaheuristic.org/aco-code/public-software.html) created by Thomas Stützle.
It could be not *Pythonic* because the original code is in *C* language. The main advantages to use this code over the original,
maybe the its simplicity, readability (for Non-C programmers) and graphical additions.

## Requirements

This framework requires:
* Python 2.7 (Not tested in Python 3)
* Numpy
* Matplotlib

### Setting up Python

For people new to Python, it is recommended to install a distribution
like [Anaconda](https://www.anaconda.com/) (or maybe
[Python(x,y)](https://python-xy.github.io/)) in order to get the
requirements easily. Be sure to install a Python 2.7 (not Python 3) instance.

## Installation

For `git` users, clone this project to your computer:

```
git clone https://github.com/ecoslacker/yaaco.git
```

or download this repository as **zip** file (green button above) and extract it.

This framework does not require installation, just copy the directory and run
the main script from a Python interpreter.

## Usage

To use this code you should create an ACO instance with the proper data and
parameters, then use the function called **run** to actually execute the
metaheuristic algorithms.

The problem instance data should be in a text file formatted as indicated in
the [TSPLIB](https://www.iwr.uni-heidelberg.de/groups/comopt/software/TSPLIB95/) documentation ([here](https://www.iwr.uni-heidelberg.de/groups/comopt/software/TSPLIB95/tsp95.pdf)) for the symmetric TSP.

NOTE: Only euclidean distance (`EUC_2D`) is acceptable for now. Other kind of distances will not work.

### Run directly

To use the code directly, open the **yaaco.py** file and go to the end of it.
There should be a block of code starting with:

```python
if __name__ == "__main__":
```

You should not edit the code above this line unless you know what you are doing.

First, you need to create an ACO object. You should pass the proper parameters,
only `filename` a string containing the path to a text file with the data (coordinates) of the problem instance is required.
Other parameters are optional. Important parameters are:

* `filename`: a text file with the data of the problem instance
* `ants`: number of ants in the colony
* `nn_ants`: number of ants in the nearest-neighbor
* `rho`: the pheromone evaporation parameter
* `alpha`: the pheromone trail influence
* `beta`: the heuristic information influence
* `max_iters`: maximum number of iterations of the algorithm
* `flag`: type of algorithm to be used in the ACO metaheuristic
* `function`: a function for distance calculation
* `instance_type`: Description of problem instance (Default 'TSP')

Possible flags include:

* AS: Ant System (Default)
* EAS: Elitist Ant System
* RAS: Rank-Based Ant System
* MMAS: Max-Min Ant System

Please refer to code documentation to more detailed description.

### Example

The following code is an example of the creation of an ACO object to solve the `eil51` problem,
using a colony of 25 ants, 500 iterations, evaporation parameter of 0.2 and the `MMAS` algorithm.

```python
    instance = 'test_data/eil51.tsp'

    # Create the ACO object & run
    tsp_aco = ACO(instance, ants=25, rho=0.2, max_iters=500, flag='MMAS')
    best = tsp_aco.run()
```

The best solution is saved in the variable called `best`.

### Import as module

If you are working in external project that requires to solve a TSP instance,
you can always import all the classes and functions to your main project. For
this you only have to copy the `yaaco.py` file and import the classes:

```python
from yaaco import ACO
```

Note that you need to configure the working directories properly.
The `ACO` class imports the other classes by default, so `Ant` and `Problem` classes
should not be imported unless necessary.

## Contribute

Contributors are welcome! If you want to improve this code please let me know by the proper channels.
It is required that you know how to work with `git` and `github` though.
Revisions, bugfixes and Python 3 testers are welcome.

## References

*  Dorigo, M., & Stützle, T. (2004). Ant colony optimization. Massachusetts,
  United States of America: Massachusetts Institute of Technology.

## License

Copyright 2017-2018 Eduardo Jiménez &lt;<ecoslacker@irriapps.com>&gt;

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
