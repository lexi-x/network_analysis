# Lexi Xu
# Lx31
# COMP 182 Spring 2021 - Homework 4, Problem 3

# You can import any standard library, as well as Numpy and Matplotlib.
# You can use helper functions from comp182.py, provided.py, and autograder.py,
# but they have to be copied over here.

import random
import numpy
import matplotlib.pyplot as plt
import pylab
import copy


def read_graph(filename):
    """
    Read a graph from a file.  The file is assumed to hold a graph
    that was written via the write_graph function.

    Arguments:
    filename -- name of file that contains the graph

    Returns:
    The graph that was stored in the input file.
    """
    with open(filename) as f:
        g = eval(f.read())
    return g


def upa(n, m):
    """
    Generate an undirected graph with n node and m edges per node
    using the preferential attachment algorithm.

    Arguments:
    n -- number of nodes
    m -- number of edges per node

    Returns:
    undirected random graph in UPAG(n, m)
    """
    g = {}
    if m <= n:
        g = make_complete_graph(m)
        for new_node in range(m, n):
            # Find <=m nodes to attach to new_node
            totdeg = float(total_degree(g))
            nodes = list(g.keys())
            probs = []
            for node in nodes:
                probs.append(len(g[node]) / totdeg)
            mult = distinct_multinomial(m, probs)

            # Add new_node and its random neighbors
            g[new_node] = set()
            for idx in mult:
                node = nodes[idx]
                g[new_node].add(node)
                g[node].add(new_node)
    return g            

def erdos_renyi(n, p):
    """
    Generate a random Erdos-Renyi graph with n nodes and edge probability p.

    Arguments:
    n -- number of nodes
    p -- probability of an edge between any pair of nodes

    Returns:
    undirected random graph in G(n, p)
    """
    g = {}

    ### Add n nodes to the graph
    for node in range(n):
        g[node] = set()

    ### Iterate through each possible edge and add it with 
    ### probability p.
    for u in range(n):
        for v in range(u+1, n):
            r = random.random()
            if r < p:
                g[u].add(v)
                g[v].add(u)

    return g


def total_degree(g):
    """
    Compute total degree of the undirected graph g.

    Arguments:
    g -- undirected graph

    Returns:
    Total degree of all nodes in g
    """
    return sum(map(len, g.values()))

def make_complete_graph(num_nodes):
    """
    Returns a complete graph containing num_nodes nodes.
 
    The nodes of the returned graph will be 0...(num_nodes-1) if num_nodes-1 is positive.
    An empty graph will be returned in all other cases.
 
    Arguments:
    num_nodes -- The number of nodes in the returned graph.
 
    Returns:
    A complete graph in dictionary form.
    """
    result = {}
         
    for node_key in range(num_nodes):
        result[node_key] = set()
        for node_value in range(num_nodes):
            if node_key != node_value: 
                result[node_key].add(node_value)
 
    return result

def distinct_multinomial(ntrials, probs):
    """
    Draw ntrials samples from a multinomial distribution given by
    probs.  Return a list of indices into probs for all distinct
    elements that were selected.  Always returns a list with between 1
    and ntrials elements.

    Arguments:
    ntrials -- number of trials
    probs   -- probability vector for the multinomial, must sum to 1

    Returns: 
    A list of indices into probs for each element that was chosen one
    or more times.  If an element was chosen more than once, it will
    only appear once in the result.  
    """
    ### select ntrials elements randomly
    mult = numpy.random.multinomial(ntrials, probs)

    ### turn the results into a list of indices without duplicates
    result = [i for i, v in enumerate(mult) if v > 0]
    return result


def copy_graph(g):
    """
    Return a copy of the input graph, g

    Arguments:
    g -- a graph

    Returns:
    A copy of the input graph that does not share any objects.
    """
    return copy.deepcopy(g)

def _plot_dict_line(d, label=None):
    """
    Plot data in the dictionary d on the current plot as a line.

    Arguments:
    d     -- dictionary
    label -- optional legend label

    Returns:
    None
    """
    xvals, yvals = _dict2lists(d)
    if label:
        pylab.plot(xvals, yvals, label=label)
    else:
        pylab.plot(xvals, yvals)

def _dict2lists(data):
    """
    Convert a dictionary into a list of keys and values, sorted by
    key.  

    Arguments:
    data -- dictionary

    Returns:
    A tuple of two lists: the first is the keys, the second is the values
    """
    xvals = list(data.keys())
    xvals.sort()
    yvals = []
    for x in xvals:
        yvals.append(data[x])
    return xvals, yvals

def _plot_dict_line(d, label=None):
    """
    Plot data in the dictionary d on the current plot as a line.

    Arguments:
    d     -- dictionary
    label -- optional legend label

    Returns:
    None
    """
    xvals, yvals = _dict2lists(d)
    if label:
        pylab.plot(xvals, yvals, label=label)
    else:
        pylab.plot(xvals, yvals)

def plot_lines(data, title, xlabel, ylabel, labels=None, filename=None):
    """
    Plot a line graph with the provided data.

    Arguments: 
    data     -- a list of dictionaries, each of which will be plotted 
                as a line with the keys on the x axis and the values on
                the y axis.
    title    -- title label for the plot
    xlabel   -- x axis label for the plot
    ylabel   -- y axis label for the plot
    labels   -- optional list of strings that will be used for a legend
                this list must correspond to the data list
    filename -- optional name of file to which plot will be
                saved (in png format)

    Returns:
    None
    """
    ### Check that the data is a list
    if not isinstance(data, list):
        msg = "data must be a list, not {0}".format(type(data).__name__)
        raise TypeError(msg)

    ### Create a new figure
    fig = pylab.figure()

    ### Plot the data
    if labels:
        mylabels = labels[:]
        for _ in range(len(data)-len(labels)):
            mylabels.append("")
        for d, l in zip(data, mylabels):
            _plot_dict_line(d, l)
        # Add legend
        pylab.legend(loc='best')
        gca = pylab.gca()
        legend = gca.get_legend()
        pylab.setp(legend.get_texts(), fontsize='medium')
    else:
        for d in data:
            _plot_dict_line(d)

    ### Set the lower y limit to 0 or the lowest number in the values
    mins = [min(l.values()) for l in data]
    ymin = min(0, min(mins))
    pylab.ylim(ymin=ymin)

    ### Label the plot
    pylab.title(title)
    pylab.xlabel(xlabel)
    pylab.ylabel(ylabel)

    ### Draw grid lines
    pylab.grid(True)

    ### Show the plot
    fig.show()

    ### Save to file
    if filename:
        pylab.savefig(filename)

def compute_largest_cc_size(g: dict) -> int:
    max_size = 0
    stack = []
    visited = set()
    for node in g:
        if node not in visited:
            stack.append(node)
            visited.add(node)
            cc_size = 1
            while len(stack) != 0:
                curr_node = stack.pop()
                for neighbor in g[curr_node]:
                    if (neighbor in g.keys()) and (neighbor not in visited):
                        stack.append(neighbor)
                        visited.add(neighbor)
                        cc_size += 1
            if cc_size > max_size:
                max_size = cc_size
    return max_size

def avg_degrees(graph):
    total = 0
    for neighbors in graph.values():
        total += len(neighbors)
    return total / len(graph.keys())

def random_attack(graph):
    """
    Perform random attack on given graph

    Arguments:
    g -- graph to be processed

    Returns:
    attack_dict -- dictionary mapping number of nodes removed to size of largest connected component in g
    """
    g = copy_graph(graph)
    attack_dict = {}
    attack_dict[0] = compute_largest_cc_size(g)
    for i in range (int(len(g)*0.2)):
        rm_key = random.choice(list(g.keys()))
        g.pop(rm_key)
        attack_dict[i] = compute_largest_cc_size(g)
    return attack_dict

def targeted_attack(graph):
    """
    Perform targeted attack on given graph based on node degree

    Arguments:
    g -- graph

    Returns:
    attack_dict -- dictionary mapping number of nodes removed to size of largest connected component in g
    """
    g = copy_graph(graph)
    attack_dict = {}
    attack_dict[0] = compute_largest_cc_size(g)
    for i in range (1,int(len(g)*0.2)):
        max_deg = 0
        max_deg_node = 0
        for node in g:
            if len(g[node]) > max_deg:
                max_deg = len(g[node])
                max_deg_node = node
        g.pop(max_deg_node)
        attack_dict[i] = compute_largest_cc_size(g)
    return attack_dict

def degree_distribution(graph):
    g = copy_graph(graph)
    degrees = [len(nbrs) for nbrs in g.values()]
    deg_dist = {}
    for num in degrees:
        if num in deg_dist:
            deg_dist[num] += 1 
        else:
            deg_dist[num] = 1
    return deg_dist


def main():
    rf7 = read_graph("rf7.repr")
    deg_avg = avg_degrees(rf7)
    # print(deg_avg)
    upa_graph = upa(1347,2)
    # upa_avg = avg_degrees(upa_graph)
    # print(upa_avg)
    erdos_graph = erdos_renyi(1347, deg_avg/1347)
    # er_avg = avg_degrees(erdos_graph)
    # print(er_avg)
    # rf7_dist = degree_distribution(rf7)
    # upa_dist = degree_distribution(upa_graph)
    # erdos_dist = degree_distribution(erdos_graph)
    # plot_lines([rf7_dist,upa_dist,erdos_dist], "Graph Connectivity", "Degree of Node", "Frequency", labels=["Rf7","UPA","Erdos-Renyi"], filename="Connectivity_Plot_LX")
    rf7_rand = random_attack(rf7)
    upa_rand = random_attack(upa_graph)
    erdos_rand = random_attack(erdos_graph)
    rf7_targ = targeted_attack(rf7)
    upa_targ = targeted_attack(upa_graph)
    erdos_targ = targeted_attack(erdos_graph)
    plot_lines([rf7_rand,upa_rand,erdos_rand,rf7_targ,upa_targ,erdos_targ], "Nodes Removed vs. Largest Connected Component", "Number of Nodes Removed", "Size of Largest Connected Component", labels=["Rf7 Random","UPA Random","E-R Random","Rf7 Targeted","UPA Targeted","E-R Targeted"], filename="Attack_Plot_LX")

main()