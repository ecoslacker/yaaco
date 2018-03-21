#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 19 11:28:19 2018

@author: ecoslacker
"""

from yaaco import ACO

instance1 = 'test_data/eil51.tsp'
instance2 = 'test_data/poke33'

# Create the ACO objects & run
tsp_aco1 = ACO(30, instance1, rho=0.2, max_iters=50)
tsp_aco2 = ACO(10, instance2, rho=0.5, max_iters=50)

best1 = tsp_aco1.run()
best2 = tsp_aco2.run()

# Show the results
print("\nBest solution for eil51:")
print(best1)

print("\nBest solution for poke33:")
print(best2)
