"""
Visualization utilities for the Agentor framework.

This module provides utilities for visualizing agent pipelines, tool compositions,
and other components of the Agentor framework. It supports both HTML and Graphviz
output formats.
"""

import json
from typing import Any, Dict, List, Optional, Union, Tuple
from enum import Enum
import html
import os
import tempfile
import webbrowser


class VisualizationFormat(Enum):
    """Supported visualization formats."""
    HTML = "html"
    GRAPHVIZ = "graphviz"
    JSON = "json"


class VisualizationError(Exception):
    """Exception raised for visualization errors."""
    pass


class Node:
    """A node in a visualization graph."""

    def __init__(
        self,
        id_: str,
        label: str,
        type_: str = "default",
        properties: Optional[Dict[str, Any]] = None,
        style: Optional[Dict[str, str]] = None
    ):
        self.id = id_
        self.label = label
        self.type = type_
        self.properties = properties or {}
        self.style = style or {}

    def to_dict(self) -> Dict[str, Any]:
        """Convert the node to a dictionary."""
        return {
            "id": self.id,
            "label": self.label,
            "type": self.type,
            "properties": self.properties,
            "style": self.style
        }


class Edge:
    """An edge in a visualization graph."""

    def __init__(
        self,
        source: str,
        target: str,
        label: Optional[str] = None,
        type_: str = "default",
        properties: Optional[Dict[str, Any]] = None,
        style: Optional[Dict[str, str]] = None
    ):
        self.source = source
        self.target = target
        self.label = label
        self.type = type_
        self.properties = properties or {}
        self.style = style or {}

    def to_dict(self) -> Dict[str, Any]:
        """Convert the edge to a dictionary."""
        result = {
            "source": self.source,
            "target": self.target,
            "type": self.type,
            "properties": self.properties,
            "style": self.style
        }
        if self.label:
            result["label"] = self.label
        return result


class Graph:
    """A graph for visualization."""

    def __init__(self, name: str, directed: bool = True):
        self.name = name
        self.directed = directed
        self.nodes: List[Node] = []
        self.edges: List[Edge] = []
        self.properties: Dict[str, Any] = {}

    def add_node(self, node: Node) -> None:
        """Add a node to the graph."""
        self.nodes.append(node)

    def add_edge(self, edge: Edge) -> None:
        """Add an edge to the graph."""
        self.edges.append(edge)

    def to_dict(self) -> Dict[str, Any]:
        """Convert the graph to a dictionary."""
        return {
            "name": self.name,
            "directed": self.directed,
            "nodes": [node.to_dict() for node in self.nodes],
            "edges": [edge.to_dict() for edge in self.edges],
            "properties": self.properties
        }

    def to_json(self, indent: Optional[int] = None) -> str:
        """Convert the graph to a JSON string."""
        return json.dumps(self.to_dict(), indent=indent)


def create_html_visualization(graph: Graph, title: Optional[str] = None) -> str:
    """Create an HTML visualization of a graph.

    Args:
        graph: The graph to visualize
        title: The title of the visualization

    Returns:
        An HTML string containing the visualization
    """
    if not title:
        title = f"Visualization of {graph.name}"

    # Escape the graph data for embedding in HTML
    graph_data = html.escape(json.dumps(graph.to_dict()))

    # Create the HTML template with embedded D3.js
    html_template = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="utf-8">
        <title>{html.escape(title)}</title>
        <script src="https://d3js.org/d3.v7.min.js"></script>
        <style>
            body {{ margin: 0; padding: 0; overflow: hidden; font-family: Arial, sans-serif; }}
            #graph {{ width: 100vw; height: 100vh; }}
            .node {{ stroke: #fff; stroke-width: 1.5px; }}
            .node.default {{ fill: #66b3ff; }}
            .node.agent {{ fill: #ff6666; }}
            .node.tool {{ fill: #66ff66; }}
            .node.environment {{ fill: #ffcc66; }}
            .node.component {{ fill: #cc66ff; }}
            .link {{ stroke: #999; stroke-opacity: 0.6; }}
            .link.default {{ stroke-width: 1px; }}
            .link.data {{ stroke-width: 2px; stroke: #66b3ff; }}
            .link.control {{ stroke-width: 2px; stroke: #ff6666; }}
            .node-label {{ font-size: 12px; pointer-events: none; }}
            .link-label {{ font-size: 10px; pointer-events: none; fill: #666; }}
        </style>
    </head>
    <body>
        <div id="graph"></div>
        <script>
            const graphData = JSON.parse("{graph_data}");

            // Create a force-directed graph
            const width = window.innerWidth;
            const height = window.innerHeight;

            const svg = d3.select("#graph")
                .append("svg")
                .attr("width", width)
                .attr("height", height);

            // Add zoom behavior
            const zoom = d3.zoom()
                .scaleExtent([0.1, 10])
                .on("zoom", (event) => {
                    container.attr("transform", event.transform);
                });

            svg.call(zoom);

            const container = svg.append("g");

            // Add arrow marker for directed edges
            if (graphData.directed) {{
                svg.append("defs").selectAll("marker")
                    .data(["default", "data", "control"])
                    .enter().append("marker")
                    .attr("id", d => `arrow-${{d}}`)
                    .attr("viewBox", "0 -5 10 10")
                    .attr("refX", 15)
                    .attr("refY", 0)
                    .attr("markerWidth", 6)
                    .attr("markerHeight", 6)
                    .attr("orient", "auto")
                    .append("path")
                    .attr("d", "M0,-5L10,0L0,5")
                    .attr("fill", d => d === "data" ? "#66b3ff" : d === "control" ? "#ff6666" : "#999");
            }}

            // Create the simulation
            const simulation = d3.forceSimulation()
                .force("link", d3.forceLink().id(d => d.id).distance(100))
                .force("charge", d3.forceManyBody().strength(-300))
                .force("center", d3.forceCenter(width / 2, height / 2));

            // Create the links
            const link = container.append("g")
                .selectAll("line")
                .data(graphData.edges)
                .enter().append("line")
                .attr("class", d => `link ${{d.type}}`)
                .attr("marker-end", d => graphData.directed ? `url(#arrow-${{d.type}})` : null)
                .style("stroke-width", d => d.style.strokeWidth || (d.type === "data" ? 2 : d.type === "control" ? 2 : 1))
                .style("stroke", d => d.style.stroke || (d.type === "data" ? "#66b3ff" : d.type === "control" ? "#ff6666" : "#999"));

            // Create the nodes
            const node = container.append("g")
                .selectAll("circle")
                .data(graphData.nodes)
                .enter().append("circle")
                .attr("class", d => `node ${{d.type}}`)
                .attr("r", d => d.style.radius || 10)
                .style("fill", d => d.style.fill || null)
                .call(d3.drag()
                    .on("start", dragstarted)
                    .on("drag", dragged)
                    .on("end", dragended));

            // Add node labels
            const nodeLabel = container.append("g")
                .selectAll("text")
                .data(graphData.nodes)
                .enter().append("text")
                .attr("class", "node-label")
                .attr("text-anchor", "middle")
                .attr("dy", 20)
                .text(d => d.label);

            // Add edge labels if they exist
            const edgeLabel = container.append("g")
                .selectAll("text")
                .data(graphData.edges.filter(d => d.label))
                .enter().append("text")
                .attr("class", "link-label")
                .attr("text-anchor", "middle")
                .attr("dy", -5)
                .text(d => d.label);

            // Update positions on simulation tick
            simulation.nodes(graphData.nodes)
                .on("tick", ticked);

            simulation.force("link")
                .links(graphData.edges);

            function ticked() {{
                link
                    .attr("x1", d => d.source.x)
                    .attr("y1", d => d.source.y)
                    .attr("x2", d => d.target.x)
                    .attr("y2", d => d.target.y);

                node
                    .attr("cx", d => d.x)
                    .attr("cy", d => d.y);

                nodeLabel
                    .attr("x", d => d.x)
                    .attr("y", d => d.y);

                edgeLabel
                    .attr("x", d => (d.source.x + d.target.x) / 2)
                    .attr("y", d => (d.source.y + d.target.y) / 2);
            }}

            function dragstarted(event, d) {{
                if (!event.active) simulation.alphaTarget(0.3).restart();
                d.fx = d.x;
                d.fy = d.y;
            }}

            function dragged(event, d) {{
                d.fx = event.x;
                d.fy = event.y;
            }}

            function dragended(event, d) {{
                if (!event.active) simulation.alphaTarget(0);
                d.fx = null;
                d.fy = null;
            }}
        </script>
    </body>
    </html>
    """

    return html_template


def create_graphviz_visualization(graph: Graph) -> str:
    """Create a Graphviz DOT visualization of a graph.

    Args:
        graph: The graph to visualize

    Returns:
        A Graphviz DOT string
    """
    # Start the DOT file
    if graph.directed:
        dot = f"digraph {graph.name.replace(' ', '_')} {{"
    else:
        dot = f"graph {graph.name.replace(' ', '_')} {{"

    # Add global graph properties
    dot += """
    graph [fontname="Arial", fontsize=12, overlap=false, splines=true];
    node [fontname="Arial", fontsize=10, shape=circle, style=filled];
    edge [fontname="Arial", fontsize=8];
    """

    # Add nodes
    for node in graph.nodes:
        # Set node attributes based on type and style
        attrs = []
        if node.type == "agent":
            attrs.append("fillcolor=\"#ff6666\"")
        elif node.type == "tool":
            attrs.append("fillcolor=\"#66ff66\"")
        elif node.type == "environment":
            attrs.append("fillcolor=\"#ffcc66\"")
        elif node.type == "component":
            attrs.append("fillcolor=\"#cc66ff\"")
        else:
            attrs.append("fillcolor=\"#66b3ff\"")

        # Add custom style attributes
        for key, value in node.style.items():
            if key == "shape" and value:
                attrs.append(f"shape=\"{value}\"")
            elif key == "fillcolor" and value:
                attrs.append(f"fillcolor=\"{value}\"")
            elif key == "color" and value:
                attrs.append(f"color=\"{value}\"")

        # Add the node definition
        dot += f'    "{node.id}" [label="{node.label}", {", ".join(attrs)}];
'

    # Add edges
    for edge in graph.edges:
        # Set edge attributes based on type and style
        attrs = []
        if edge.type == "data":
            attrs.append("color=\"#66b3ff\"")
            attrs.append("penwidth=2")
        elif edge.type == "control":
            attrs.append("color=\"#ff6666\"")
            attrs.append("penwidth=2")

        # Add custom style attributes
        for key, value in edge.style.items():
            if key == "color" and value:
                attrs.append(f"color=\"{value}\"")
            elif key == "penwidth" and value:
                attrs.append(f"penwidth={value}")
            elif key == "style" and value:
                attrs.append(f"style=\"{value}\"")

        # Add label if it exists
        if edge.label:
            attrs.append(f"label=\"{edge.label}\"")

        # Add the edge definition
        if graph.directed:
            dot += f'    "{edge.source}" -> "{edge.target}" [{", ".join(attrs)}];
'
        else:
            dot += f'    "{edge.source}" -- "{edge.target}" [{", ".join(attrs)}];
'

    # Close the graph
    dot += "}"

    return dot


def visualize(
    graph: Graph,
    format_: VisualizationFormat = VisualizationFormat.HTML,
    output_file: Optional[str] = None,
    open_browser: bool = False,
    title: Optional[str] = None
) -> str:
    """Visualize a graph.

    Args:
        graph: The graph to visualize
        format_: The visualization format
        output_file: The output file path (if None, returns the visualization as a string)
        open_browser: Whether to open the visualization in a browser
        title: The title of the visualization (for HTML format)

    Returns:
        The visualization as a string, or the path to the output file if output_file is specified
    """
    # Generate the visualization
    if format_ == VisualizationFormat.HTML:
        content = create_html_visualization(graph, title)
    elif format_ == VisualizationFormat.GRAPHVIZ:
        content = create_graphviz_visualization(graph)
    elif format_ == VisualizationFormat.JSON:
        content = graph.to_json(indent=2)
    else:
        raise VisualizationError(f"Unsupported visualization format: {format_}")

    # Write to file if specified
    if output_file:
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(content)

        # Open in browser if requested
        if open_browser:
            if format_ == VisualizationFormat.HTML:
                webbrowser.open(f"file://{os.path.abspath(output_file)}")
            elif format_ == VisualizationFormat.GRAPHVIZ and output_file.endswith(".dot"):
                # Try to convert to SVG and open that
                try:
                    import graphviz
                    svg_file = output_file.replace(".dot", ".svg")
                    graphviz.render("dot", "svg", output_file, outfile=svg_file)
                    webbrowser.open(f"file://{os.path.abspath(svg_file)}")
                except ImportError:
                    print("Graphviz Python package not installed. Cannot render DOT file.")
                except Exception as e:
                    print(f"Error rendering DOT file: {e}")

        return output_file
    else:
        return content


def visualize_pipeline(pipeline: Any, **kwargs) -> str:
    """Visualize a pipeline.

    This is a convenience function that creates a graph from a pipeline and visualizes it.

    Args:
        pipeline: The pipeline to visualize
        **kwargs: Additional arguments to pass to visualize()

    Returns:
        The visualization as a string, or the path to the output file if output_file is specified
    """
    # Import here to avoid circular imports
    from agentor.core.pipeline import Pipeline

    if not isinstance(pipeline, Pipeline):
        raise VisualizationError("Object is not a Pipeline")

    # Create a graph from the pipeline
    graph = Graph(name=pipeline.name or "Pipeline", directed=True)

    # Add nodes for each component
    for component in pipeline.components:
        node_type = "component"
        if hasattr(component, "type") and component.type:
            node_type = component.type

        node = Node(
            id_=component.id,
            label=component.name or component.id,
            type_=node_type
        )
        graph.add_node(node)

    # Add edges for connections
    for connection in pipeline.connections:
        edge = Edge(
            source=connection.source_id,
            target=connection.target_id,
            label=connection.name,
            type_=connection.type or "default"
        )
        graph.add_edge(edge)

    # Visualize the graph
    return visualize(graph, **kwargs)


def visualize_tool_composition(tool_composition: Any, **kwargs) -> str:
    """Visualize a tool composition.

    This is a convenience function that creates a graph from a tool composition and visualizes it.

    Args:
        tool_composition: The tool composition to visualize
        **kwargs: Additional arguments to pass to visualize()

    Returns:
        The visualization as a string, or the path to the output file if output_file is specified
    """
    # Import here to avoid circular imports
    from agentor.agents.tools.composition import ToolComposition

    if not isinstance(tool_composition, ToolComposition):
        raise VisualizationError("Object is not a ToolComposition")

    # Create a graph from the tool composition
    graph = Graph(name=tool_composition.name or "Tool Composition", directed=True)

    # Add nodes for each tool
    for tool in tool_composition.tools:
        node = Node(
            id_=tool.id,
            label=tool.name or tool.id,
            type_="tool"
        )
        graph.add_node(node)

    # Add edges for connections
    for i, tool in enumerate(tool_composition.tools[:-1]):
        next_tool = tool_composition.tools[i + 1]
        edge = Edge(
            source=tool.id,
            target=next_tool.id,
            type_="data"
        )
        graph.add_edge(edge)

    # Visualize the graph
    return visualize(graph, **kwargs)


def quick_visualize(obj: Any, open_browser: bool = True) -> str:
    """Quickly visualize an object.

    This is a convenience function that creates a temporary HTML file and opens it in a browser.

    Args:
        obj: The object to visualize (Pipeline or ToolComposition)
        open_browser: Whether to open the visualization in a browser

    Returns:
        The path to the temporary file
    """
    # Import here to avoid circular imports
    from agentor.core.pipeline import Pipeline
    from agentor.agents.tools.composition import ToolComposition

    # Create a temporary file
    fd, path = tempfile.mkstemp(suffix=".html")
    os.close(fd)

    # Visualize the object
    if isinstance(obj, Pipeline):
        visualize_pipeline(obj, output_file=path, open_browser=open_browser, format_=VisualizationFormat.HTML)
    elif isinstance(obj, ToolComposition):
        visualize_tool_composition(obj, output_file=path, open_browser=open_browser, format_=VisualizationFormat.HTML)
    else:
        raise VisualizationError("Object must be a Pipeline or ToolComposition")

    return path