import sys

# SearchGraph.py
#
# Implementation of iterative deepening search for use in finding optimal routes
# between locations in a graph. In the graph to be searched, nodes have names
# (e.g. city names for a map).
#
# An undirected graph is passed in as a text file (first command line argument). 
#
# Usage: python SearchGraph.py graphFile startLocation endLocation
# 
# Author: Richard Zanibbi, RIT, Nov. 2011
# Author: Deepak Sharma, RIT, SEP, 2016

class problem:
    """
     represent an instance of the problem and provide
     necessary abstraction for the internals of the problem
     :slot start: initial state
     :slot goal:  final state
     :slot graph: an adjacency list for representing the graph
    """
    __slots__ = 'start', 'goal', 'graph'

    def __init__(self, start, goal, graph):
        """
        Initialize a node.
        :param start: Starting state for the search problem
        :param goal: goal state for the search problem
        :param graph: adjacency list for representing the graph
        :return: None
        """
        self.start = start
        self.goal = goal
        self.graph = graph

    def successor_function(self,current_state):
        """
        provide list of possible actions
        For the given problem action and result is same
        if action is any city then result will be the same city
        :param current_state:current state of the agent
        :return: list of possible actions/results
        """
        return self.graph[current_state]

    def goal_test(self,aState):
        """
        can be use by agent for performing goal test
        :param aState: input state
        :return: boolean value, true if input state is the goal state
        """
        return aState==self.goal

    def path_cost(self,source,destination):
        """
        Method provide the distance between two points if
        an edge exist between them
        :param source:
        :param destination:
        :return: return 1 if there is a direct edge between two points
        else return infinity.
        """
        if destination in self.graph[source]:
            return 1
        else:
            return float("inf");

def read_graph(filename):
    """Read in edges of a graph represented one per line,
    using the format: srcStateName destStateName"""
    print("Loading graph: " + filename)
    edges = {}
    inFile = open(filename)
    for line in inFile:
        roadInfo = line.split()
        # Skip blank lines, read in contents from non-empty lines.
        if (len(roadInfo) > 0):
            srcCity = roadInfo[0]
            destCity = roadInfo[1]

            if srcCity in edges:
                edges[srcCity] = edges[srcCity] + [destCity ]
            else:
                edges[srcCity] = [ destCity ]

            if destCity in edges:
                edges[destCity] = edges[destCity] + [ srcCity ]
            else:
                edges[destCity] = [ srcCity ]
    print("  done.\n")
    return edges


def RDLS(node,search_problem,depth,limit,path,explored):
    """
    Implementation of recursive depth limited DFS
    :param node: Starting node for recursive process
    :param search_problem: an instance of the search problem
    :param depth: depth of the search tree for current recursive call
    :param limit: depth limit for limited depth recursive DFS
    :param path:  Store the path from source to destination
    :param explored: store detail of the explored nodes in a hashMap
    :param space:   store the tab distance for printing the search tree
    :return:
    """
    explored[node]=depth
    print(depth*"    "+node)
    cutoff_occurred=False;
    if(search_problem.goal_test(node)):
        path.append(node)
        return path
    elif depth==limit:
        return "cutoff"
    else:
        successors=search_problem.successor_function(node)
        for successor in successors:
            if successor not in explored or explored[successor]>depth:

                result=RDLS(successor , search_problem,depth+1,limit,path,explored)
                if result=="cutoff":
                    cutoff_occurred=True
                elif result!="Failure":
                    result.insert(0,node)
                    return result
    if cutoff_occurred:
        return "cutoff"
    else:
        return "Failure"


def DLS(search_problem,depth):
    """
    Implementation of depth limited DFS algorithm
    :param search_problem:
    :param depth:
    :return:
    """
    #explored={search_problem.start:0}
    return RDLS(search_problem.start, search_problem, 0, depth, [], {})


def IDS(search_problem):
    """
    Implementation of iterative  DFS algorithm
    :param search_problem:
    :return:
    """
    for depth in range(1000):
        print("\n ----------iteration: "+str(depth+1)+"------------ \n")
        result=DLS(search_problem,depth)
        if result =="Failure":
            return ["FAIL"]
        if result !="cutoff":
            return result


######################################
# Add functions for search, output
# etc. here
######################################

# TBD

#########################
# Main program
#########################

def main():
    if len(sys.argv) != 4:
        print('Usage: python SearchGraph.py graphFilename startNode goalNode')
        return
    else:
        # Create a dictionary (i.e. associative array, implemented as a hash
        # table) for edges in the map file, and define start and end states for
        # the search. Each dictionary entry key is a string for a location,
        # associated with a list of strings for the adjacent states (cities) in
        # the state space.
        edges = {}
        edges = read_graph(sys.argv[1])
        start = sys.argv[2]
        goal = sys.argv[3]

        #Comment out the following lines to hide the graph description.
        #print("-- Adjacent Cities (Edge Dictionary Data) ------------------------")
        #for location in edges.keys():
        #	s = '  ' + location + ':\n     '
        #	s = s + str(edges[location])
        #	print(s)

    if not start in edges.keys():
        print("Start location is not in the graph.")
    else:
        search_problem=problem(start,goal,edges)
        print('')
        print('-- States Visited ----------------'+start+" to "+goal)
        path=IDS(search_problem)

        print('')
        print('--  Solution for: ' + start + ' to ' + goal + '-------------------')
        print(path) # program will need to provide solution path or indicate failure.
        print('')

        """
        start = "Arad"
        goal = "Neamt"
        search_problem=problem(start,goal,edges)
        print('')
        print('-- States Visited ----------------'+start+" to "+goal)
        path=IDS(search_problem)

        print('')
        print('--  Solution for: ' + start + ' to ' + goal + '-------------------')
        print(path) # program will need to provide solution path or indicate failure.
        print('')

        start = "Bucharest"
        goal = "Zerind"
        search_problem=problem(start,goal,edges)
        print('')
        print('-- States Visited ----------------'+start+" to "+goal)
        path=IDS(search_problem)

        print('')
        print('--  Solution for: ' + start + ' to ' + goal + '-------------------')
        print(path) # program will need to provide solution path or indicate failure.
        print('')
        """

main()


