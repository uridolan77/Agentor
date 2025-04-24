"""
Visualization utilities for the Agentor framework.

This module provides utilities for visualizing agent components, including:
- Tool pipeline visualization
- Agent state visualization
- Environment visualization
"""

import logging
from typing import Dict, Any, List, Optional, Union, Tuple, Set
import json
import os
import tempfile
import webbrowser
from pathlib import Path

from agentor.agents.composition import (
    ToolPipeline, ParallelToolPipeline, ConditionalBranch,
    ToolNode, ToolCondition
)

logger = logging.getLogger(__name__)

try:
    import graphviz
    GRAPHVIZ_AVAILABLE = True
except ImportError:
    logger.warning("Graphviz not available. Install with 'pip install graphviz'")
    GRAPHVIZ_AVAILABLE = False


def pipeline_to_graphviz(
    pipeline: Union[ToolPipeline, ParallelToolPipeline, ConditionalBranch],
    filename: Optional[str] = None,
    format: str = "png",
    view: bool = False
) -> Optional["graphviz.Digraph"]:
    """Convert a tool pipeline to a Graphviz diagram.
    
    Args:
        pipeline: The pipeline to visualize
        filename: Optional filename to save the diagram to
        format: The format to save the diagram in
        view: Whether to open the diagram in the default viewer
        
    Returns:
        The Graphviz diagram, or None if Graphviz is not available
    """
    if not GRAPHVIZ_AVAILABLE:
        logger.error("Graphviz not available. Install with 'pip install graphviz'")
        return None
    
    # Create a new directed graph
    dot = graphviz.Digraph(
        comment="Tool Pipeline",
        format=format,
        node_attr={"shape": "box", "style": "filled", "fillcolor": "lightblue"}
    )
    
    # Add nodes and edges based on the pipeline type
    if isinstance(pipeline, ToolPipeline):
        _add_tool_pipeline_to_graph(dot, pipeline)
    elif isinstance(pipeline, ParallelToolPipeline):
        _add_parallel_pipeline_to_graph(dot, pipeline)
    elif isinstance(pipeline, ConditionalBranch):
        _add_conditional_branch_to_graph(dot, pipeline)
    else:
        logger.error(f"Unknown pipeline type: {type(pipeline)}")
        return None
    
    # Save and optionally view the diagram
    if filename:
        dot.render(filename, view=view)
    
    return dot


def _add_tool_pipeline_to_graph(dot: "graphviz.Digraph", pipeline: ToolPipeline) -> None:
    """Add a tool pipeline to a Graphviz diagram.
    
    Args:
        dot: The Graphviz diagram
        pipeline: The pipeline to add
    """
    # Add the pipeline node
    pipeline_id = f"pipeline_{id(pipeline)}"
    dot.node(pipeline_id, f"Pipeline: {pipeline.name}")
    
    # Add the tools
    prev_node_id = pipeline_id
    for i, node in enumerate(pipeline.nodes):
        if isinstance(node, ToolNode):
            node_id = f"tool_{id(node)}"
            dot.node(node_id, f"Tool: {node.tool.name}")
            dot.edge(prev_node_id, node_id)
            prev_node_id = node_id
        elif isinstance(node, ToolPipeline):
            node_id = f"pipeline_{id(node)}"
            dot.node(node_id, f"Pipeline: {node.name}")
            dot.edge(prev_node_id, node_id)
            _add_tool_pipeline_to_graph(dot, node)
            prev_node_id = node_id
        elif isinstance(node, ParallelToolPipeline):
            node_id = f"parallel_{id(node)}"
            dot.node(node_id, f"Parallel: {node.name}")
            dot.edge(prev_node_id, node_id)
            _add_parallel_pipeline_to_graph(dot, node)
            prev_node_id = node_id
        elif isinstance(node, ConditionalBranch):
            node_id = f"conditional_{id(node)}"
            dot.node(node_id, f"Conditional: {node.name}")
            dot.edge(prev_node_id, node_id)
            _add_conditional_branch_to_graph(dot, node)
            prev_node_id = node_id


def _add_parallel_pipeline_to_graph(dot: "graphviz.Digraph", pipeline: ParallelToolPipeline) -> None:
    """Add a parallel tool pipeline to a Graphviz diagram.
    
    Args:
        dot: The Graphviz diagram
        pipeline: The pipeline to add
    """
    # Add the pipeline node
    pipeline_id = f"parallel_{id(pipeline)}"
    dot.node(pipeline_id, f"Parallel: {pipeline.name}")
    
    # Add the branches
    for i, branch in enumerate(pipeline.branches):
        branch_id = f"branch_{id(branch)}_{i}"
        dot.node(branch_id, f"Branch {i}")
        dot.edge(pipeline_id, branch_id)
        
        if isinstance(branch, ToolNode):
            node_id = f"tool_{id(branch)}"
            dot.node(node_id, f"Tool: {branch.tool.name}")
            dot.edge(branch_id, node_id)
        elif isinstance(branch, ToolPipeline):
            node_id = f"pipeline_{id(branch)}"
            dot.node(node_id, f"Pipeline: {branch.name}")
            dot.edge(branch_id, node_id)
            _add_tool_pipeline_to_graph(dot, branch)
        elif isinstance(branch, ParallelToolPipeline):
            node_id = f"parallel_{id(branch)}"
            dot.node(node_id, f"Parallel: {branch.name}")
            dot.edge(branch_id, node_id)
            _add_parallel_pipeline_to_graph(dot, branch)
        elif isinstance(branch, ConditionalBranch):
            node_id = f"conditional_{id(branch)}"
            dot.node(node_id, f"Conditional: {branch.name}")
            dot.edge(branch_id, node_id)
            _add_conditional_branch_to_graph(dot, branch)


def _add_conditional_branch_to_graph(dot: "graphviz.Digraph", branch: ConditionalBranch) -> None:
    """Add a conditional branch to a Graphviz diagram.
    
    Args:
        dot: The Graphviz diagram
        branch: The branch to add
    """
    # Add the branch node
    branch_id = f"conditional_{id(branch)}"
    dot.node(branch_id, f"Conditional: {branch.name}")
    
    # Add the condition
    condition_id = f"condition_{id(branch.condition)}"
    dot.node(condition_id, f"Condition: {branch.condition.name}")
    dot.edge(branch_id, condition_id)
    
    # Add the true branch
    true_id = f"true_{id(branch.true_branch)}"
    dot.node(true_id, "True Branch")
    dot.edge(condition_id, true_id, label="True")
    
    if isinstance(branch.true_branch, ToolNode):
        node_id = f"tool_{id(branch.true_branch)}"
        dot.node(node_id, f"Tool: {branch.true_branch.tool.name}")
        dot.edge(true_id, node_id)
    elif isinstance(branch.true_branch, ToolPipeline):
        node_id = f"pipeline_{id(branch.true_branch)}"
        dot.node(node_id, f"Pipeline: {branch.true_branch.name}")
        dot.edge(true_id, node_id)
        _add_tool_pipeline_to_graph(dot, branch.true_branch)
    elif isinstance(branch.true_branch, ParallelToolPipeline):
        node_id = f"parallel_{id(branch.true_branch)}"
        dot.node(node_id, f"Parallel: {branch.true_branch.name}")
        dot.edge(true_id, node_id)
        _add_parallel_pipeline_to_graph(dot, branch.true_branch)
    elif isinstance(branch.true_branch, ConditionalBranch):
        node_id = f"conditional_{id(branch.true_branch)}"
        dot.node(node_id, f"Conditional: {branch.true_branch.name}")
        dot.edge(true_id, node_id)
        _add_conditional_branch_to_graph(dot, branch.true_branch)
    
    # Add the false branch if it exists
    if branch.false_branch:
        false_id = f"false_{id(branch.false_branch)}"
        dot.node(false_id, "False Branch")
        dot.edge(condition_id, false_id, label="False")
        
        if isinstance(branch.false_branch, ToolNode):
            node_id = f"tool_{id(branch.false_branch)}"
            dot.node(node_id, f"Tool: {branch.false_branch.tool.name}")
            dot.edge(false_id, node_id)
        elif isinstance(branch.false_branch, ToolPipeline):
            node_id = f"pipeline_{id(branch.false_branch)}"
            dot.node(node_id, f"Pipeline: {branch.false_branch.name}")
            dot.edge(false_id, node_id)
            _add_tool_pipeline_to_graph(dot, branch.false_branch)
        elif isinstance(branch.false_branch, ParallelToolPipeline):
            node_id = f"parallel_{id(branch.false_branch)}"
            dot.node(node_id, f"Parallel: {branch.false_branch.name}")
            dot.edge(false_id, node_id)
            _add_parallel_pipeline_to_graph(dot, branch.false_branch)
        elif isinstance(branch.false_branch, ConditionalBranch):
            node_id = f"conditional_{id(branch.false_branch)}"
            dot.node(node_id, f"Conditional: {branch.false_branch.name}")
            dot.edge(false_id, node_id)
            _add_conditional_branch_to_graph(dot, branch.false_branch)


def pipeline_to_html(
    pipeline: Union[ToolPipeline, ParallelToolPipeline, ConditionalBranch],
    filename: Optional[str] = None,
    view: bool = False
) -> str:
    """Convert a tool pipeline to an HTML visualization.
    
    Args:
        pipeline: The pipeline to visualize
        filename: Optional filename to save the HTML to
        view: Whether to open the HTML in the default browser
        
    Returns:
        The HTML string
    """
    # Convert the pipeline to a JSON structure
    pipeline_json = _pipeline_to_json(pipeline)
    
    # Create the HTML
    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Tool Pipeline Visualization</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; }}
            .node {{ border: 1px solid #ccc; padding: 10px; margin: 10px; border-radius: 5px; }}
            .pipeline {{ background-color: #e6f7ff; }}
            .parallel {{ background-color: #f7ffe6; }}
            .conditional {{ background-color: #ffe6e6; }}
            .tool {{ background-color: #e6ffe6; }}
            .condition {{ background-color: #fff2e6; }}
            .branch {{ margin-left: 20px; }}
        </style>
        <script>
            const pipelineData = {json.dumps(pipeline_json)};
            
            function renderPipeline(container, pipeline) {{
                const div = document.createElement('div');
                div.className = `node ${{pipeline.type.toLowerCase()}}`;
                div.innerHTML = `<h3>${{pipeline.name}}</h3>`;
                
                if (pipeline.type === 'Pipeline') {{
                    const nodesList = document.createElement('div');
                    nodesList.className = 'nodes';
                    
                    for (const node of pipeline.nodes) {{
                        const nodeDiv = document.createElement('div');
                        nodeDiv.className = 'branch';
                        nodesList.appendChild(nodeDiv);
                        
                        if (node.type === 'Tool') {{
                            nodeDiv.innerHTML = `<div class="node tool"><h4>Tool: ${{node.name}}</h4></div>`;
                        }} else {{
                            renderPipeline(nodeDiv, node);
                        }}
                    }}
                    
                    div.appendChild(nodesList);
                }} else if (pipeline.type === 'Parallel') {{
                    const branchesList = document.createElement('div');
                    branchesList.className = 'branches';
                    
                    for (let i = 0; i < pipeline.branches.length; i++) {{
                        const branch = pipeline.branches[i];
                        const branchDiv = document.createElement('div');
                        branchDiv.className = 'branch';
                        branchDiv.innerHTML = `<h4>Branch ${{i + 1}}</h4>`;
                        
                        if (branch.type === 'Tool') {{
                            branchDiv.innerHTML += `<div class="node tool"><h4>Tool: ${{branch.name}}</h4></div>`;
                        }} else {{
                            renderPipeline(branchDiv, branch);
                        }}
                        
                        branchesList.appendChild(branchDiv);
                    }}
                    
                    div.appendChild(branchesList);
                }} else if (pipeline.type === 'Conditional') {{
                    const conditionDiv = document.createElement('div');
                    conditionDiv.className = 'node condition';
                    conditionDiv.innerHTML = `<h4>Condition: ${{pipeline.condition.name}}</h4>`;
                    div.appendChild(conditionDiv);
                    
                    const trueBranchDiv = document.createElement('div');
                    trueBranchDiv.className = 'branch';
                    trueBranchDiv.innerHTML = `<h4>True Branch</h4>`;
                    
                    if (pipeline.true_branch.type === 'Tool') {{
                        trueBranchDiv.innerHTML += `<div class="node tool"><h4>Tool: ${{pipeline.true_branch.name}}</h4></div>`;
                    }} else {{
                        renderPipeline(trueBranchDiv, pipeline.true_branch);
                    }}
                    
                    div.appendChild(trueBranchDiv);
                    
                    if (pipeline.false_branch) {{
                        const falseBranchDiv = document.createElement('div');
                        falseBranchDiv.className = 'branch';
                        falseBranchDiv.innerHTML = `<h4>False Branch</h4>`;
                        
                        if (pipeline.false_branch.type === 'Tool') {{
                            falseBranchDiv.innerHTML += `<div class="node tool"><h4>Tool: ${{pipeline.false_branch.name}}</h4></div>`;
                        }} else {{
                            renderPipeline(falseBranchDiv, pipeline.false_branch);
                        }}
                        
                        div.appendChild(falseBranchDiv);
                    }}
                }}
                
                container.appendChild(div);
            }}
            
            window.onload = function() {{
                const container = document.getElementById('pipeline-container');
                renderPipeline(container, pipelineData);
            }};
        </script>
    </head>
    <body>
        <h1>Tool Pipeline Visualization</h1>
        <div id="pipeline-container"></div>
    </body>
    </html>
    """
    
    # Save the HTML if requested
    if filename:
        with open(filename, "w") as f:
            f.write(html)
        
        # Open the HTML in the default browser if requested
        if view:
            webbrowser.open(f"file://{os.path.abspath(filename)}")
    
    return html


def _pipeline_to_json(
    pipeline: Union[ToolPipeline, ParallelToolPipeline, ConditionalBranch, ToolNode]
) -> Dict[str, Any]:
    """Convert a pipeline to a JSON structure.
    
    Args:
        pipeline: The pipeline to convert
        
    Returns:
        A JSON-serializable dictionary
    """
    if isinstance(pipeline, ToolNode):
        return {
            "type": "Tool",
            "name": pipeline.tool.name,
            "description": pipeline.tool.description
        }
    elif isinstance(pipeline, ToolPipeline):
        return {
            "type": "Pipeline",
            "name": pipeline.name,
            "nodes": [_pipeline_to_json(node) for node in pipeline.nodes]
        }
    elif isinstance(pipeline, ParallelToolPipeline):
        return {
            "type": "Parallel",
            "name": pipeline.name,
            "branches": [_pipeline_to_json(branch) for branch in pipeline.branches]
        }
    elif isinstance(pipeline, ConditionalBranch):
        result = {
            "type": "Conditional",
            "name": pipeline.name,
            "condition": {
                "name": pipeline.condition.name,
                "description": getattr(pipeline.condition, "description", "")
            },
            "true_branch": _pipeline_to_json(pipeline.true_branch)
        }
        
        if pipeline.false_branch:
            result["false_branch"] = _pipeline_to_json(pipeline.false_branch)
        
        return result
    else:
        return {
            "type": "Unknown",
            "name": str(pipeline)
        }


def visualize_pipeline(
    pipeline: Union[ToolPipeline, ParallelToolPipeline, ConditionalBranch],
    format: str = "html"
) -> None:
    """Visualize a tool pipeline.
    
    Args:
        pipeline: The pipeline to visualize
        format: The format to use (html or graphviz)
    """
    if format == "html":
        # Create a temporary file
        with tempfile.NamedTemporaryFile(suffix=".html", delete=False) as f:
            filename = f.name
        
        # Generate the HTML
        pipeline_to_html(pipeline, filename=filename, view=True)
        
        logger.info(f"Pipeline visualization saved to {filename}")
    elif format == "graphviz":
        if not GRAPHVIZ_AVAILABLE:
            logger.error("Graphviz not available. Install with 'pip install graphviz'")
            return
        
        # Create a temporary file
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            filename = f.name
        
        # Generate the Graphviz diagram
        pipeline_to_graphviz(pipeline, filename=filename, view=True)
        
        logger.info(f"Pipeline visualization saved to {filename}")
    else:
        logger.error(f"Unknown format: {format}")
