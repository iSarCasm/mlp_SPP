import numpy as np
# import tkinter
# import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import networkx as nx
import csv, sys
import time

G=nx.Graph()
G.add_nodes_from(list(range(1,30)))
G.add_edge(1, 2)
G.add_edge(2, 3)
G.add_edge(2, 5)
G.add_edge(3, 5)
G.add_edge(3, 4)
G.add_edge(2, 20)
G.add_edge(6, 20)
G.add_edge(5, 6)
G.add_edge(6, 10)
G.add_edge(6, 7)
G.add_edge(7, 8)
G.add_edge(8, 9)
G.add_edge(9, 10)
G.add_edge(10, 11)
G.add_edge(11, 12)
G.add_edge(11, 13)
G.add_edge(13, 14)
G.add_edge(14, 15)
G.add_edge(15, 16)
G.add_edge(16, 17)
G.add_edge(17, 18)
G.add_edge(18, 19)
G.add_edge(19, 20)
G.add_edge(17, 22)
G.add_edge(16, 20)
G.add_edge(20, 21)
G.add_edge(21, 22)
G.add_edge(22, 1)
G.add_edge(21, 23)
G.add_edge(23, 24)
G.add_edge(24, 26)
G.add_edge(26, 1)
G.add_edge(1, 29)
G.add_edge(29, 30)
G.add_edge(24, 25)
G.add_edge(26, 27)
G.add_edge(27, 28)

nodes = len(G.nodes())
edges = len(G.edges())
print("Network has {} nodes".format(nodes))
print("Network has {} edges".format(edges))

source = 12
target = 1
print("Finding SP from {} to {}".format(source, target))


gigabit_max_packet_size = 1518 + 12 + 8 # Ethernet Frame + Interframe Gap + Preamble
gigabit_max_throughput = 125 * 1e6 # 1000 MB/s
gigabit_min_throughput = 1e2       # 100 bytes/s
f_min = gigabit_max_packet_size / gigabit_max_throughput
f_max = gigabit_max_packet_size / gigabit_min_throughput

print('Max througtput: ' + str(gigabit_max_throughput))
print('Min througtput: ' + str(gigabit_min_throughput))
print('f max: ' + str(f_max))
print('f min: ' + str(f_min))

def generate_example_inputs(G):
    throughputs = []
    for edge in G.edges():
        i, j = edge
        throughput = np.random.uniform(gigabit_min_throughput, gigabit_max_throughput)
        G[i][j]['weight'] = gigabit_max_packet_size / throughput
        throughputs.append(throughput)
    return throughputs

def generate_example_output(G):
    dijkstra = nx.bidirectional_dijkstra(G, source=source, target=target)
    node_path_list = dijkstra[1]
    edge_path = nodes_to_edges(node_path_list)
    path_vector = route_to_vec(edge_path)
    return path_vector

def route_to_vec(route):
    path_vector = [0] * edges
    for edge in route:
        sorted_edge = tuple(sorted(edge))
        edge_indx = edge_index[sorted_edge]
        path_vector[edge_indx] = 1
    return path_vector

def nodes_to_edges(node_route):
    edges = []
    y = node_route[0]
    for i in range(1, len(node_route)):
        x = y
        y = node_route[i]
        edge = (x, y)
        edges.append(edge)
    return edges

GG = list(G.edges())
edge_index = dict((GG[value], value) for value in range(len(GG)))

def generate_example():
    x = generate_example_inputs(G)
    y = generate_example_output(G)
    return (x,y)

m = 14
vm = 14
tm = 14

print("Generating {} examples".format(m))

data = []
labels = []

for i in range(m):
    x, y = generate_example()
    data.append(x)
    labels.append(y)

unique_routes = list(nx.all_simple_paths(G, source=source, target=target))
unique_routes_edges = [nodes_to_edges(i) for i in unique_routes]
unique_route_vecs = dict()
unique_routes_cnt = len(unique_routes)
for r in range(unique_routes_cnt):
    route = unique_routes_edges[r]
    route_vec = tuple(route_to_vec(route))
    unique_route_vecs[route_vec] = r

unique_routes.sort(key=len)
print("All possible routes from {} to {}".format(source, target))
for r in range(len(unique_route_vecs)):
    print("Route {} => {}".format(r, list(unique_routes)[r]))

def analyze_dataset(G, labels):
    route_ids = []
    route_counts = [0] * len(unique_routes)
    for i in range(m):
        l = tuple(labels[i])
        r_id = unique_route_vecs[l]
        route_ids.append(r_id)
        route_counts[r_id] += 1

    for i in range(len(unique_routes)):
        print("Route {} has {} examples in dataset".format(i, route_counts[i]))

    # fig, ax = plt.subplots()
    # bins = list(range(0, 14))
    # plt.bar(bins, route_counts)
    # ax.set_xticks([i for i in bins])
    # ax.set_xticklabels(('0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13'))
    # fig.set_size_inches(10,5)
    # plt.show()

    # plt.pie(route_counts, labels=bins)
    # plt.show()

analyze_dataset(G, labels)

def generate_dataset(m):
    cnt = [0] * unique_routes_cnt
    examples_per_route = m // unique_routes_cnt
    print('Generating {} examples per route'.format(examples_per_route))
    new_datas = []
    new_labels = []
    i = 0
    start = time.time()
    while True:
        new_data, new_label = generate_example()

        route_type = unique_route_vecs[tuple(new_label)]
        if cnt[route_type] < examples_per_route:
            new_datas.append(new_data)
            new_labels.append(new_label)
            cnt[route_type] += 1

        if len(new_datas) >= m:
            break

        i += 1
        if i % 100000 == 0:
            time_passed = time.time() - start
            start = time.time()
            print([time_passed, i, cnt, len(new_datas) / m])
    return (new_datas, new_labels)

train_data, train_labels = generate_dataset(11800)

with open('v2train_dataset_11800_8.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    for i in range(m):
        example = train_data[i] + train_labels[i]
        writer.writerow(example)

with open('v2valid_routes.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    for r in list(unique_route_vecs):
        writer.writerow(r)