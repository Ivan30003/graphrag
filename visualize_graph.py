import networkx as nx
import matplotlib.pyplot as plt
import argparse
from files_utils import read_csv_file

def main(args):
    graph_list_triples = read_csv_file(args.input_path)
    output_file = args.output_path
    # Step 2: Create a graph instance
    graph = nx.MultiDiGraph() # NetworkxEntityGraph()

    # Step 3: Add entities and relationships to the graph
    for triple in graph_list_triples:
        subject, predicate, object_ = triple[:3]
        subject = '\n'.join(subject.split(' '))
        predicate = '\n'.join(predicate.split(' '))
        object_ = '\n'.join(object_.split(' '))
        graph.add_node(subject)
        graph.add_node(object_)
        graph.add_edge(subject, object_, label=predicate)

    # Step 4: Visualize the graph
    # Convert to a networkx graph for visualization
    # nx_graph = graph.to_networkx()
    pos = nx.spring_layout(graph)  # positions for all nodes

    # Nodes
    nx.draw_networkx_nodes(graph, pos, node_size=400)

    # Edges
    nx.draw_networkx_edges(graph, pos, width=5)
    nx.draw_networkx_labels(graph, pos, font_size=5, font_family="sans-serif")

    # Edge labels
    edge_labels = dict([((u, v,), d['label'])
                        for u, v, d in graph.edges(data=True)])
    nx.draw_networkx_edge_labels(graph, pos, edge_labels=edge_labels, font_size=5)
    plt.savefig(output_file)
    
# Example usage

# visualize_graph(graph_dict, 'output_graph.pdf')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_path', type=str, help='Path to saved csv triples')
    parser.add_argument('--output_path', type=str, help='Specific path result write to')
    args = parser.parse_args()
    main(args)
