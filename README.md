# TIPE-Graphe

Tirte : Modélisation et étude de la propagation d'une épidémie sur un graphe et application à la recherche de stratégie vaccinale.

Subsection : étude du cas plus général de la propagation sur un graphe plus "quelconque"


Recherche de modélisation adéquate par les modèles de Watts-Strogatz et de Barabasi-Albert
Pour Watts-Strogatz : http://snap.stanford.edu/class/cs224w-readings/watts98smallworld.pdf
Pour Barabasi-Albert : https://journals.aps.org/rmp/abstract/10.1103/RevModPhys.74.47

Procédure : Introduire des coefficients de centralité pour évaluer la vaccination d'un noeud. Supposer que le procédé idéal est une combinaison linéaire 
de ces coefficients. Maximiser les résultats en faisant varier les coefficients à l'aide d'un algorithme génétique.

Résultats de la recherche : La stratégie Shortest_Path (qui est du coup simplifiable en faisant juste une boucle for sur les infectés et en renvoyant le plus proche)
est la plus efficace : les coefficients proposés par l'algorithme génétique favorisaient en général cette stratégie.

Interpréattion : finalement c'est logique, c'est la même stratégie d'isolement des cas contact qu'on met en place actuellement...








