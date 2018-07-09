import numpy as np
import random
import math
import time
import copy

class CustomerLocation:
    # number of the city on the file
    indis = 0
    # coordianates of the cities
    x = 0
    y = 0

    def __init__(self, indis=None, x=None, y=None):
        self.indis = indis
        self.x = x
        self.y = y

    def __str__(self):
        return str(self.indis)

    def __repr__(self):
        return self.__str__()

    def __eq__(self, other):
        return self.indis == other.indis

    def __hash__(self):
        return self.indis

def euclideanDistance(ca, cb):
    dist = np.sqrt((ca.x - cb.x) ** 2 + (ca.y - cb.y) ** 2)
    return round(dist, 2)

def closest_centroid(f, centroids):
    closest = centroids[0]
    c_distance = float("inf")
    for centroid in centroids:
        distance = euclideanDistance(centroid, f)
        if distance - c_distance <= 0:
            closest = centroid
            c_distance = distance
    return closest

def total_distance(centroids):
    total_dist = 0
    for cent, nodes in centroids.items():
        for node in nodes:
            total_dist += euclideanDistance(cent, node)
    return total_dist

a = []

optimum_res = {}

def init(fileName, result_file):

    with open(filename, "r") as f:
        points = [line.replace("\r\n", "") for line in f]

    i = 0
    for point in points:
        pointData = point.split(' ')
        pointData = [float(x) for x in pointData]
        i += 1
        cust = CustomerLocation(i, pointData[0], pointData[1])
        a.append(cust)

    with open(result_file, "r") as rf:
        opts_line = [line.replace("\n", "") for line in rf]
        for opt in opts_line:
            opt_res = opt.split(' ')
            optimum_res[int(opt_res[0])] = float(opt_res[1])


filename = "p654.txt"
result_file = 'result_file.txt'
init(filename, result_file)

customer_assign = {}

#Choose the customer that has not moved from its assigned cluster at that iteration
def choose_customer(customer_assign, changed_cust):
    customer = random.sample(customer_assign.keys(), 1)[0]
    while (customer in changed_cust):
        customer = random.sample(customer_assign.keys(), 1)[0]
    return customer

#Choose a random cluster that the chosen customer is not assigned to so far
def choose_cluster(customer, clusters):
    can_clusters = clusters.keys()
    random.shuffle(can_clusters)

    for cluster in can_clusters:
        if cluster not in customer_assign[customer]:
            return cluster


def generate_neighbor(solution, customer_assign, k):
    clusters = copy.deepcopy(solution)
    #keep track of reassigned customers in the iteration
    changed_cust = []
    for i in range(k):
        # If there is still a customer that has not been relocated recently,
        # new neighbor can be generated
        if len(customer_assign.keys()) > 0:
            customer = choose_customer(customer_assign, changed_cust)
            changed_cust.append(customer)
            #Remove the customer from its current cluster
            c = customer_assign[customer]
            from_cluster = c[-1]
            for can_to in customer_assign[customer]:
                if customer in clusters[can_to]:
                    from_cluster = can_to
            clusters[from_cluster].remove(customer)
            # Assign customer to new cluster
            to_cluster = choose_cluster(customer, clusters)
            clusters[to_cluster].append(customer)
            customer_assign[customer].append(to_cluster)
            # Manage history
            if len(customer_assign[customer]) == len(clusters.keys()):
                if k == 1:
                    # Customer has visited all facilities, no change any more
                    del customer_assign[customer]
                else:
                    # Customer is ready to visit some of the same facilities again
                    customer_assign.clear()
                    customer_assign.append(to_cluster)
        else:
            return copy.deepcopy(solution)
    return clusters


#Generate a random unvisited neighbor
def cust_shake(opt_solution, k, customer_assign):
    neighbor = generate_neighbor(opt_solution, customer_assign, k)
    recalculate_cluster(neighbor)
    return neighbor

#Cluster centroids are basically the mean of x and y values
def recalculate_cluster(clusters):
    for cluster, nodes in clusters.items():
        nxs = [n.x for n in nodes]
        cx = np.mean(nxs)
        nys = [n.y for n in nodes]
        cy = np.mean(nys)
        cluster.x = cx
        cluster.y = cy



def local_search(solution, solution_cost, customer_assign, max_neighborhood):
    current_best = solution
    current_cost = solution_cost
    new_cost = 0
    count = 0
    max_count = len(customer_assign.keys())
    customer_assign.clear()
    is_improved = True
    k = 1
    # Use the variable neighborhood structure in local search
    while k < max_neighborhood + 1:
        # For each neighborhood size, clean the history
        for cluster, nodes in current_best.items():
            for node in nodes:
                customer_assign[node] = [cluster]
        while (len(customer_assign.keys()) > 0) and (max_count > count):
            count +=1
            neighbor = generate_neighbor(current_best, customer_assign, k)
            recalculate_cluster(neighbor)
            new_cost = total_distance(neighbor)
            # Decide the local optimum
            if current_cost > new_cost:
                current_best = neighbor
                current_cost = new_cost
                is_improved = True
        # For diversity, try different neighborhood structures
        if not is_improved:
            k += 1
        else:
            break
    return current_cost, current_best


def apply_vns(initial_solution, max_neighborhood, customer_assign):
 
    k = 1
    opt_solution = copy.deepcopy(initial_solution)
    opt_sol_cost = total_distance(initial_solution)
    no_imp_count = 0
    while no_imp_count < 10:
        k = 1
        while k < max_neighborhood + 1:
            customer_assign.clear()
            for cluster, nodes in opt_solution.items():
                for node in nodes:
                    customer_assign[node] = [cluster]
            random_solution = cust_shake(opt_solution, k, customer_assign)
            random_sol_cost = total_distance(random_solution)
            local_sol_cost, local_solution = local_search(random_solution, random_sol_cost, customer_assign, max_neighborhood)
            if opt_sol_cost > local_sol_cost:
                opt_sol_cost = local_sol_cost
                opt_solution = local_solution
                k = 1
                no_imp_count = 0
            else:
                k += 1
                no_imp_count += 1

    return opt_sol_cost, opt_solution

def apply_k_means(run_num, number_of_facility, customer_assign):
    for run in range(1, run_num + 1):
        start_time = time.time()
        final_distance = 0;
        total_dist = 1
        centroids = random.sample(a, number_of_facility)
        clusters = {facility: [] for facility in centroids}
        isImproved = True
        no_imp_count = 0
        while (isImproved):
            for f in a:
                closest = closest_centroid(f, clusters.keys())
                clusters[closest].append(f)
            total_dist = total_distance(clusters)
            if abs(final_distance - total_dist) < 0.01:
                no_imp_count +=1
                if no_imp_count > 5:
                    isImproved = False
                    opt_dist = optimum_res[number_of_facility]
                    ratio = final_distance / opt_dist
                    ratio = round(ratio, 2)
                    print run, ". run : ", final_distance, " %", (100 * ratio) - 100
                    opt_sol_cost, opt_solution = apply_vns(clusters, 3, customer_assign)
                    ratio2 = opt_sol_cost / opt_dist
                    ratio2 = round(ratio2, 2)
                    print run, ". run : ", 'Result of VNS algorithm', opt_sol_cost, " %", (100 * ratio2) - 100

            if isImproved:
                final_distance = total_dist
                new_clusters = {}
                i = 0
                for cluster, nodes in clusters.items():
                    nxs = [n.x for n in nodes]
                    cx = np.mean(nxs)
                    nys = [n.y for n in nodes]
                    cy = np.mean(nys)
                    i -= 1
                    new_cent = CustomerLocation(i, cx, cy)
                    new_clusters[new_cent] = []
                clusters.clear()
                clusters = new_clusters

        end_time = time.time()
        print 'CPU time:', end_time - start_time
    return clusters

number_of_facility = 8
run_num = 5
final_clusters = apply_k_means(run_num, number_of_facility, customer_assign)
