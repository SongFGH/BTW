import random


def read_edges():
    edges = []
    with open('edges-1.csv') as f:
        for line in f:
            words = line.split(',')
            src = int(words[0]) - 1
            dest = int(words[1]) - 1
            while max(src, dest) >= len(edges):
                edges.append(set())
            edges[src].add(dest)
            edges[dest].add(src)
    return edges


def make_map(nodes):
    mapper = {}
    for node in sorted(nodes):
        mapper[node] = len(mapper)
    return mapper


def write_edges(file, edges, mapper):
    with open(file, 'w') as f:
        for src, dests in enumerate(edges):
            if src in mapper:
                for dest in sorted(dests):
                    if dest in mapper and dest > src:
                        f.write('{},{}\n'.format(mapper[src], mapper[dest]))


def filter_labels(file1, file2, mapper):
    with open(file1) as f1:
        with open(file2, 'w') as f2:
            for line in f1:
                words = line.split(',')
                node = int(words[0]) - 1
                if node in mapper:
                    label = int(words[1]) - 1
                    f2.write('{},{}\n'.format(mapper[node], label))


def find_gcc(edges):
    nodes = set(range(len(edges)))
    while nodes:
        set1 = set(random.sample(nodes, 1))
        set2 = set(set1)
        while set2:
            set3 = set()
            for n in set2:
                set3 = set3.union(edges[n])
            set2 = set3.difference(set1)
            set1 = set1.union(set3)
        if len(set1) == 35579:
            return set1
        nodes = nodes.difference(set1)


def main():
    edges = read_edges()
    nodes = find_gcc(edges)
    mapper = make_map(nodes)
    write_edges('edges.csv', edges, mapper)
    filter_labels('labels-1.csv', 'labels.csv', mapper)


if __name__ == '__main__':
    main()
