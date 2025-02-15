import rdflib
from rdflib.namespace import RDF, RDFS, OWL
import networkx as nx
import matplotlib.pyplot as plt

# 1. Load/Parse your RDF graph
rdf_graph = rdflib.Graph()
rdf_graph.parse("enriched_final_graph.ttl", format="turtle")

# 2. Initialize a NetworkX DiGraph (directed graph) or Graph (undirected)
nx_graph = nx.DiGraph()

# 3. Extract and add classes as nodes
all_classes = set(rdf_graph.subjects(RDF.type, RDFS.Class))

for cls in all_classes:
    nx_graph.add_node(str(cls), rdf_type="Class")

    for superclass in rdf_graph.objects(cls, RDFS.subClassOf):
        nx_graph.add_node(str(superclass), rdf_type="Class")
        nx_graph.add_edge(str(superclass), str(cls), label="subClassOf")

# 4. Extract and add properties as nodes
all_properties = set(rdf_graph.subjects(RDF.type, RDF.Property))

all_properties |= set(rdf_graph.subjects(RDF.type, OWL.ObjectProperty))
all_properties |= set(rdf_graph.subjects(RDF.type, OWL.DatatypeProperty))

for prop in all_properties:
    prop_str = str(prop)
    nx_graph.add_node(prop_str, rdf_type="Property")

    # 4a. Domain links
    for domain in rdf_graph.objects(prop, RDFS.domain):
        nx_graph.add_node(str(domain), rdf_type="Class")
        # Connect domain to property
        nx_graph.add_edge(str(domain), prop_str, label="hasProperty")

    # 4b. Range links
    for rng in rdf_graph.objects(prop, RDFS.range):
        nx_graph.add_node(str(rng), rdf_type="Class")
        # Connect property to range
        nx_graph.add_edge(prop_str, str(rng), label="hasRange")

# 5. Draw the graph
plt.figure(figsize=(12,12))

# Increase the 'k' for more spacing
pos = nx.spring_layout(
    nx_graph, 
    k=2.0,         
    scale=2.0,     
    iterations=50, 
    seed=42
)

# Draw nodes with different colors or shapes based on their type
class_nodes = [n for n, attr in nx_graph.nodes(data=True) if attr.get("rdf_type") == "Class"]
prop_nodes  = [n for n, attr in nx_graph.nodes(data=True) if attr.get("rdf_type") == "Property"]

nx.draw_networkx_nodes(nx_graph, pos, nodelist=class_nodes, node_color="lightblue", node_shape="o", label="Classes")
nx.draw_networkx_nodes(nx_graph, pos, nodelist=prop_nodes,  node_color="pink", node_shape="s", label="Properties")
nx.draw_networkx_edges(nx_graph, pos, arrowstyle="->", arrowsize=10)

nx.draw_networkx_labels(nx_graph, pos, font_size=8)

edge_labels = nx.get_edge_attributes(nx_graph, "label")
nx.draw_networkx_edge_labels(nx_graph, pos, edge_labels=edge_labels, font_color="red")

plt.title("Classes and Their Properties")
plt.axis("off")
plt.legend()
plt.show()
