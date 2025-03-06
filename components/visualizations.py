"""
Visualizations for the agent platform UI.
"""

import streamlit as st
import random
import json
import numpy as np
from typing import Dict, List, Any, Optional

def render_agent_metrics_chart(metrics: Dict[str, float]):
    """
    Render a radar chart showing agent metrics.
    
    Args:
        metrics: Dict mapping metric names to values (0-1)
    """
    # Convert metrics to format needed for chart
    labels = list(metrics.keys())
    values = [metrics[label] for label in labels]
    
    # Create radar chart using Plotly
    try:
        import plotly.graph_objects as go
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatterpolar(
            r=values,
            theta=labels,
            fill='toself',
            name='Agent Metrics',
            line_color='rgb(75, 111, 255)',
            fillcolor='rgba(75, 111, 255, 0.3)'
        ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 1]
                )
            ),
            showlegend=False,
            margin=dict(l=40, r=40, t=40, b=40)
        )
        
        st.plotly_chart(fig, use_container_width=True)
    except ImportError:
        # Fallback to a simpler visualization if Plotly is not available
        st.write("Agent Metrics:")
        for label, value in metrics.items():
            st.metric(label, f"{value:.2f}")
            st.progress(value)

def render_capability_network(capabilities: List[Dict[str, Any]]):
    """
    Render a network visualization of agent capabilities and their dependencies.
    
    Args:
        capabilities: List of capability dictionaries
    """
    if not capabilities:
        st.info("No capabilities to visualize.")
        return
    
    # Create nodes for capabilities
    nodes = []
    for i, cap in enumerate(capabilities):
        nodes.append({
            "id": i,
            "label": cap["name"],
            "color": f"rgb({random.randint(50, 200)}, {random.randint(50, 200)}, {random.randint(50, 200)})",
            "size": 25  # Size based on number of dependencies
        })
    
    # Create edges from capabilities to dependencies
    edges = []
    
    # Create nodes for dependencies
    dep_nodes = {}
    dep_idx = len(capabilities)
    
    for i, cap in enumerate(capabilities):
        for dep in cap.get("dependencies", []):
            if dep not in dep_nodes:
                dep_nodes[dep] = dep_idx
                nodes.append({
                    "id": dep_idx,
                    "label": dep,
                    "color": "rgb(200, 200, 200)",
                    "size": 15
                })
                dep_idx += 1
            
            # Add edge from capability to dependency
            edges.append({
                "from": i,
                "to": dep_nodes[dep]
            })
    
    # Convert to JSON for JavaScript - fix the string formatting issue
    nodes_json = json.dumps(nodes)
    edges_json = json.dumps(edges)
    
    network_js = f"""
    <script src="https://unpkg.com/vis-network/standalone/umd/vis-network.min.js"></script>
    <div id="capability-network" style="height: 400px; border: 1px solid #ddd; border-radius: 8px;"></div>
    <script>
        const nodes = {nodes_json};
        const edges = {edges_json};
        
        const container = document.getElementById("capability-network");
        const data = {{
            nodes: new vis.DataSet(nodes),
            edges: new vis.DataSet(edges)
        }};
        
        const options = {{
            nodes: {{
                shape: "dot",
                font: {{
                    size: 12,
                    face: "Tahoma"
                }}
            }},
            edges: {{
                width: 1.5,
                smooth: {{
                    type: "continuous"
                }}
            }},
            physics: {{
                stabilization: true,
                barnesHut: {{
                    gravitationalConstant: -8000,
                    springConstant: 0.04,
                    springLength: 95
                }}
            }},
            layout: {{
                randomSeed: 2
            }}
        }};
        
        const network = new vis.Network(container, data, options);
    </script>
    """
    
    # Render the visualization using custom HTML component
    st.components.v1.html(network_js, height=450)

def generate_sample_data(num_points: int = 100):
    """
    Generate sample data for testing visualizations.
    
    Args:
        num_points: Number of data points to generate
        
    Returns:
        Dict containing sample data
    """
    # Generate random time series data
    x = list(range(num_points))
    y1 = [random.random() for _ in range(num_points)]
    y2 = [random.random() * 0.8 + 0.1 for _ in range(num_points)]
    
    # Generate random metrics
    metrics = {
        "Accuracy": random.random() * 0.3 + 0.7,
        "Speed": random.random() * 0.4 + 0.5,
        "Reliability": random.random() * 0.2 + 0.7,
        "Coverage": random.random() * 0.5 + 0.4
    }
    
    # Generate sample capabilities
    capabilities = [
        {
            "name": "Text Analysis",
            "description": "Analyze and interpret text content",
            "dependencies": ["nlp", "sentiment_analysis", "entity_recognition"]
        },
        {
            "name": "Image Recognition",
            "description": "Identify objects in images",
            "dependencies": ["computer_vision", "object_detection"]
        },
        {
            "name": "Data Processing",
            "description": "Process and transform structured data",
            "dependencies": ["data_cleaning", "normalization"]
        }
    ]
    
    return {
        "time_series": {"x": x, "y1": y1, "y2": y2},
        "metrics": metrics,
        "capabilities": capabilities
    }
