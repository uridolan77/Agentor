"""
Tests for the visualization module.
"""

import pytest
import os
import tempfile
from typing import Dict, Any

from agentor.agents.composition import (
    ToolPipeline, ParallelToolPipeline, ConditionalBranch,
    ToolNode, ToolCondition
)
from agentor.agents.visualization import (
    pipeline_to_graphviz, pipeline_to_html, _pipeline_to_json
)


class MockTool:
    """Mock tool for testing."""
    
    def __init__(self, name, description):
        """Initialize the mock tool."""
        self.name = name
        self.description = description
    
    async def run(self, **kwargs):
        """Run the mock tool."""
        return {"result": "Mock tool result"}


class MockCondition(ToolCondition):
    """Mock condition for testing."""
    
    def __init__(self, name):
        """Initialize the mock condition."""
        self.name = name
    
    async def evaluate(self, context: Dict[str, Any]) -> bool:
        """Evaluate the mock condition."""
        return True


def test_pipeline_to_json():
    """Test the _pipeline_to_json function."""
    # Create a simple pipeline
    tool1 = MockTool("tool1", "Tool 1")
    tool2 = MockTool("tool2", "Tool 2")
    tool3 = MockTool("tool3", "Tool 3")
    
    node1 = ToolNode(tool1)
    node2 = ToolNode(tool2)
    node3 = ToolNode(tool3)
    
    pipeline = ToolPipeline(name="Test Pipeline", nodes=[node1, node2])
    
    # Convert to JSON
    json_data = _pipeline_to_json(pipeline)
    
    # Check the result
    assert json_data["type"] == "Pipeline"
    assert json_data["name"] == "Test Pipeline"
    assert len(json_data["nodes"]) == 2
    assert json_data["nodes"][0]["type"] == "Tool"
    assert json_data["nodes"][0]["name"] == "tool1"
    assert json_data["nodes"][1]["type"] == "Tool"
    assert json_data["nodes"][1]["name"] == "tool2"
    
    # Create a parallel pipeline
    parallel = ParallelToolPipeline(name="Test Parallel", branches=[node1, node2])
    
    # Convert to JSON
    json_data = _pipeline_to_json(parallel)
    
    # Check the result
    assert json_data["type"] == "Parallel"
    assert json_data["name"] == "Test Parallel"
    assert len(json_data["branches"]) == 2
    assert json_data["branches"][0]["type"] == "Tool"
    assert json_data["branches"][0]["name"] == "tool1"
    assert json_data["branches"][1]["type"] == "Tool"
    assert json_data["branches"][1]["name"] == "tool2"
    
    # Create a conditional branch
    condition = MockCondition("Test Condition")
    conditional = ConditionalBranch(
        name="Test Conditional",
        condition=condition,
        true_branch=node1,
        false_branch=node2
    )
    
    # Convert to JSON
    json_data = _pipeline_to_json(conditional)
    
    # Check the result
    assert json_data["type"] == "Conditional"
    assert json_data["name"] == "Test Conditional"
    assert json_data["condition"]["name"] == "Test Condition"
    assert json_data["true_branch"]["type"] == "Tool"
    assert json_data["true_branch"]["name"] == "tool1"
    assert json_data["false_branch"]["type"] == "Tool"
    assert json_data["false_branch"]["name"] == "tool2"
    
    # Create a nested pipeline
    nested = ToolPipeline(name="Nested Pipeline", nodes=[node3, conditional])
    
    # Convert to JSON
    json_data = _pipeline_to_json(nested)
    
    # Check the result
    assert json_data["type"] == "Pipeline"
    assert json_data["name"] == "Nested Pipeline"
    assert len(json_data["nodes"]) == 2
    assert json_data["nodes"][0]["type"] == "Tool"
    assert json_data["nodes"][0]["name"] == "tool3"
    assert json_data["nodes"][1]["type"] == "Conditional"
    assert json_data["nodes"][1]["name"] == "Test Conditional"


def test_pipeline_to_html():
    """Test the pipeline_to_html function."""
    # Create a simple pipeline
    tool1 = MockTool("tool1", "Tool 1")
    tool2 = MockTool("tool2", "Tool 2")
    
    node1 = ToolNode(tool1)
    node2 = ToolNode(tool2)
    
    pipeline = ToolPipeline(name="Test Pipeline", nodes=[node1, node2])
    
    # Convert to HTML
    html = pipeline_to_html(pipeline)
    
    # Check that the HTML contains the pipeline name
    assert "Test Pipeline" in html
    
    # Check that the HTML contains the tool names
    assert "tool1" in html
    assert "tool2" in html
    
    # Save the HTML to a temporary file
    with tempfile.NamedTemporaryFile(suffix=".html", delete=False) as f:
        filename = f.name
    
    try:
        # Convert to HTML and save
        pipeline_to_html(pipeline, filename=filename)
        
        # Check that the file exists
        assert os.path.exists(filename)
        
        # Check that the file contains the HTML
        with open(filename, "r") as f:
            content = f.read()
            assert "Test Pipeline" in content
            assert "tool1" in content
            assert "tool2" in content
    finally:
        # Clean up
        if os.path.exists(filename):
            os.remove(filename)


@pytest.mark.skipif(not hasattr(pytest, "importorskip"), reason="importorskip not available")
def test_pipeline_to_graphviz():
    """Test the pipeline_to_graphviz function."""
    # Skip if graphviz is not available
    pytest.importorskip("graphviz")
    
    # Create a simple pipeline
    tool1 = MockTool("tool1", "Tool 1")
    tool2 = MockTool("tool2", "Tool 2")
    
    node1 = ToolNode(tool1)
    node2 = ToolNode(tool2)
    
    pipeline = ToolPipeline(name="Test Pipeline", nodes=[node1, node2])
    
    # Convert to Graphviz
    dot = pipeline_to_graphviz(pipeline)
    
    # Check that the dot object was created
    assert dot is not None
    
    # Save the diagram to a temporary file
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
        filename = f.name
    
    try:
        # Convert to Graphviz and save
        pipeline_to_graphviz(pipeline, filename=filename)
        
        # Check that the file exists (the .png extension is added by graphviz)
        assert os.path.exists(f"{filename}.png")
    finally:
        # Clean up
        if os.path.exists(f"{filename}.png"):
            os.remove(f"{filename}.png")
