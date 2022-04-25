# Multi-Objective Ergodic Search

This work is motivated by humanitarian assistant and disaster relief (HADR) where often it is critical to find signs of life in the presence of conflicting criteria, objectives, and information. 
We believe ergodic search can provide a framework for exploiting available information as well as exploring for new information for applications such as HADR, especially when time is of the essence. 
Ergodic search algorithms plan trajectories such that the time spent in a region is proportional to the amount of information in that region, and is able to naturally balance exploitation (myopically searching high-information areas) and exploration (visiting all locations in the search space for new information).
Existing ergodic search algorithms, as well as other information-based approaches, typically consider search using only a single information map.
However, in many scenarios, the use of multiple information maps that encode different types of relevant information is common. 
Ergodic search methods currently do not possess the ability for simultaneous nor do they have a way to balance which information gets priority.
This leads us to formulate a Multi-Objective Ergodic Search (MOES) problem, which aims at finding the so-called Pareto-optimal solutions, for the purpose of providing human decision makers various solutions that trade off between conflicting criteria.
To efficiently solve MOES, we develop a framework called Sequential Local Ergodic Search (SLES) that converts a MOES problem into a "weight space coverage" problem. It leverages the recent advances in ergodic search methods as well as the idea of local optimization to efficiently approximate the Pareto-optimal front.
More details can be found in our [paper]().

<p align="center">
<img src="https://github.com/wonderren/wonderren.github.io/blob/master/images/fig_moes_overview.png" alt="" hspace="15" style=" border: #FFFFFF 2px none;">
</p>

(Fig 1: A conceptual visualization of the MOES problem and our method. (a) shows a search and rescue task in a hazardous material warehouse with leakage, where colored areas indicate different types of information/targets such as survivors, leakage sources, etc. (b) shows the weight space in the presence of three objectives (i.e. three info maps to be covered). (c) shows the scalarized info map, which is the weighted-sum of the three info maps, and an ergodic trajectory with respect to the scalarized info map. (d) visualizes the objective space, where each element is an ergodic vector that describes the ergodic metric of the computed trajectory with respect to each of the three info maps. The computed ergodic vectors approximate the Pareto-optimal front.)


## Requirements

* Python 3.7.3, higher or lower version may also work.

* Python package scipy is required.

* The ergodic search implementation in this repository leverages [Ian's code](https://github.com/i-abr/ErgodicControl), which requires [JAX](https://jax.readthedocs.io/en/latest/notebooks/quickstart.html).

* This implementation leverages the NSGA-II from [pymoo 0.4.2.2](https://pymoo.org/). If a higher version of pymoo is chosen, some modification of code may be necessary to adapt to the new APIs of pymoo.


## Instructions

* `python3 run_example.py` provides an entry point to the code.


## Others

### References

TODO


