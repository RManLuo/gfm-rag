import logging
from abc import ABC, abstractmethod
from typing import NotRequired, TypedDict

logger = logging.getLogger(__name__)


class Node(TypedDict):
    """
    Represents a node in the graph with its attributes.

    Attributes:
        name (str): Unique identifier for the node.
        type (str): Type of the node (e.g., "entity", "document").
        attributes (dict): Additional attributes of the node.
        uid (str, optional): Optional unique identifier for the node.
    """

    name: str
    type: str
    attributes: dict
    uid: NotRequired[str]  # Optional unique identifier for the node


class Edge(TypedDict):
    """
    Represents an edge in the graph.

    Attributes:
        source (str): Source node name or uid.
        relation (str): Relation (name or uid) between the nodes.
        target (str): Target node name or uid.
        attributes (dict): Additional attributes of the edge.
    """

    source: str
    relation: str
    target: str
    attributes: dict


class Relation(TypedDict):
    """
    Represents a relation in the graph.

    Attributes:
        name (str): Unique identifier for the relation.
        attributes (dict): Additional attributes of the relation.
        uid (str, optional): Optional unique identifier for the relation.
    """

    name: str
    attributes: dict
    uid: NotRequired[str]


class Graph(TypedDict):
    """
    Represents a graph structure containing nodes, edges, and relations.

    Attributes:
        nodes (list[Node]): List of nodes in the graph.
        relations (list[Relation]): List of relations in the graph.
        edges (list[Edge]): List of edges in the graph.
    """

    nodes: list[Node]
    relations: list[Relation]
    edges: list[Edge]


class BaseGraphConstructor(ABC):
    """
    Abstract base class for graph construction.

    This class defines the interface for constructing graphs from datasets.
    Subclasses must implement create_graph()methods.

    Attributes:
        None

    Methods:
        create_graph: Creates a graph from the specified dataset.

    """

    @abstractmethod
    def build_graph(self, data_root: str, data_name: str) -> Graph:
        """
        Create a graph from the dataset

        Args:
            data_root (str): path to the dataset
            data_name (str): name of the dataset
        Returns:
            Graph: The constructed graph containing nodes, edges, and relations.
        """
        pass
