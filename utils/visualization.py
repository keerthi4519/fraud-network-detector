
### File 5: `utils/visualization.py`

"""
Visualization utilities for fraud network detection.
"""

import plotly.graph_objects as go
import plotly.express as px
import networkx as nx
import numpy as np
import pandas as pd


def create_network_plotly(G, title="Entity Network Graph", max_nodes=500):
    """Create an interactive network visualization using Plotly."""
    if len(G.nodes) == 0:
        fig = go.Figure()
        fig.add_annotation(text="No data to display", xref="paper", yref="paper",
                          x=0.5, y=0.5, showarrow=False, font=dict(size=20))
        return fig
    
    # Sample if too large
    if len(G.nodes) > max_nodes:
        # Keep high-degree nodes
        degrees = dict(G.degree())
        top_nodes = sorted(degrees, key=degrees.get, reverse=True)[:max_nodes]
        G = G.subgraph(top_nodes).copy()
    
    # Layout
    pos = nx.spring_layout(G, k=2/np.sqrt(len(G.nodes)), iterations=50, seed=42)
    
    # Edge traces
    edge_x = []
    edge_y = []
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])
    
    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=0.5, color='#888'),
        hoverinfo='none',
        mode='lines',
        name='Relationships'
    )
    
    # Node traces - color by entity type
    color_map = {
        "claimant": "#FF6B6B",
        "provider": "#4ECDC4",
        "claim": "#45B7D1",
        "policy": "#96CEB4",
        "invoice": "#FFEAA7",
        "incident": "#DDA0DD",
    }
    
    node_traces = {}
    for node in G.nodes():
        etype = G.nodes[node].get("entity_type", "unknown")
        if etype not in node_traces:
            node_traces[etype] = {"x": [], "y": [], "text": [], "size": [], "color": color_map.get(etype, "#999")}
        
        x, y = pos[node]
        node_traces[etype]["x"].append(x)
        node_traces[etype]["y"].append(y)
        
        risk = G.nodes[node].get("risk_score", 0)
        name = G.nodes[node].get("name", G.nodes[node].get("entity_id", node))
        hover_text = f"Type: {etype}<br>ID: {G.nodes[node].get('entity_id', '')}<br>Name: {name}<br>Risk: {risk:.1f}<br>Connections: {G.degree(node)}"
        node_traces[etype]["text"].append(hover_text)
        node_traces[etype]["size"].append(max(8, min(30, G.degree(node) * 3)))
    
    traces = [edge_trace]
    for etype, data in node_traces.items():
        trace = go.Scatter(
            x=data["x"], y=data["y"],
            mode='markers',
            hoverinfo='text',
            text=data["text"],
            name=etype.capitalize(),
            marker=dict(
                size=data["size"],
                color=data["color"],
                line=dict(width=1, color='white'),
            )
        )
        traces.append(trace)
    
    fig = go.Figure(data=traces, layout=go.Layout(
        title=dict(text=title, font=dict(size=16)),
        showlegend=True,
        hovermode='closest',
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        height=600,
        margin=dict(l=20, r=20, t=40, b=20),
    ))
    
    return fig


def create_suspicion_heatmap(scores_df):
    """Create a heatmap of suspicion scores across communities."""
    if scores_df is None or len(scores_df) == 0:
        fig = go.Figure()
        fig.add_annotation(text="No data", xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
        return fig
    
    top_n = scores_df.head(30)
    
    fig = go.Figure(data=go.Bar(
        x=top_n["community_id"].astype(str),
        y=top_n["suspicion_score"],
        marker=dict(
            color=top_n["suspicion_score"],
            colorscale="RdYlGn_r",
            showscale=True,
            colorbar=dict(title="Score"),
        ),
        text=[f"{'⚠️ FRAUD' if row['is_known_fraud'] else '✓'}" for _, row in top_n.iterrows()],
        textposition='outside',
        hovertemplate=(
            "Community: %{x}<br>"
            "Score: %{y:.1f}<br>"
            "<extra></extra>"
        ),
    ))
    
    fig.update_layout(
        title="Top Suspicious Networks by Score",
        xaxis_title="Community ID",
        yaxis_title="Suspicion Score",
        height=400,
        plot_bgcolor='rgba(0,0,0,0)',
    )
    
    return fig


def create_entity_distribution_chart(datasets):
    """Create a chart showing entity distributions."""
    counts = {
        "Claimants": len(datasets.get("claimants", [])),
        "Policies": len(datasets.get("policies", [])),
        "Claims": len(datasets.get("claims", [])),
        "Incidents": len(datasets.get("incidents", [])),
        "Providers": len(datasets.get("providers", [])),
        "Invoices": len(datasets.get("invoices", [])),
        "Payments": len(datasets.get("payments", [])),
        "Edges": len(datasets.get("relationship_edges", [])),
    }
    
    fig = go.Figure(data=go.Bar(
        x=list(counts.keys()),
        y=list(counts.values()),
        marker_color=['#FF6B6B', '#96CEB4', '#45B7D1', '#DDA0DD', '#4ECDC4', '#FFEAA7', '#F0B27A', '#BB8FCE'],
        text=[f"{v:,}" for v in counts.values()],
        textposition='outside',
    ))
    
    fig.update_layout(
        title="Entity Count Distribution",
        yaxis_title="Count",
        height=400,
        plot_bgcolor='rgba(0,0,0,0)',
    )
    
    return fig


def create_risk_distribution(claimants_df, providers_df):
    """Create risk score distribution plots."""
    fig = go.Figure()
    
    if claimants_df is not None and "risk_score" in claimants_df.columns:
        fig.add_trace(go.Histogram(
            x=claimants_df["risk_score"],
            name="Claimant Risk",
            marker_color='#FF6B6B',
            opacity=0.7,
            nbinsx=50,
        ))
    
    if providers_df is not None and "risk_score" in providers_df.columns:
        fig.add_trace(go.Histogram(
            x=providers_df["risk_score"],
            name="Provider Risk",
            marker_color='#4ECDC4',
            opacity=0.7,
            nbinsx=50,
        ))
    
    fig.update_layout(
        title="Risk Score Distribution",
        xaxis_title="Risk Score",
        yaxis_title="Count",
        barmode='overlay',
        height=400,
        plot_bgcolor='rgba(0,0,0,0)',
    )
    
    return fig


def create_fraud_type_pie(ground_truth_df):
    """Create a pie chart of fraud types."""
    if ground_truth_df is None or len(ground_truth_df) == 0:
        fig = go.Figure()
        fig.add_annotation(text="No data", xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
        return fig
    
    fraud_only = ground_truth_df[ground_truth_df["is_fraud"] == True]
    if len(fraud_only) == 0:
        fig = go.Figure()
        fig.add_annotation(text="No fraud cases", xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
        return fig
    
    type_counts = fraud_only["fraud_type"].value_counts()
    
    fig = go.Figure(data=go.Pie(
        labels=type_counts.index,
        values=type_counts.values,
        hole=0.4,
        marker=dict(colors=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7', '#DDA0DD']),
    ))
    
    fig.update_layout(
        title="Fraud Type Distribution",
        height=400,
    )
    
    return fig


def create_investigation_timeline(investigations_df):
    """Create investigation timeline chart."""
    if investigations_df is None or len(investigations_df) == 0:
        fig = go.Figure()
        fig.add_annotation(text="No investigations", xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
        return fig
    
    status_colors = {
        "open": "#FF6B6B",
        "in_progress": "#FFEAA7",
        "pending_review": "#45B7D1",
        "closed_confirmed": "#96CEB4",
        "closed_cleared": "#BB8FCE",
    }
    
    fig = go.Figure()
    for _, row in investigations_df.iterrows():
        fig.add_trace(go.Bar(
            x=[row["estimated_loss"]],
            y=[row["investigation_id"]],
            orientation='h',
            marker_color=status_colors.get(row["status"], "#999"),
            name=row["status"],
            showlegend=False,
            hovertemplate=(
                f"ID: {row['investigation_id']}<br>"
                f"Network: {row['network_id']}<br>"
                f"Type: {row['fraud_type']}<br>"
                f"Loss: ${row['estimated_loss']:,.2f}<br>"
                f"Status: {row['status']}<br>"
                f"<extra></extra>"
            ),
        ))
    
    fig.update_layout(
        title="Investigations by Estimated Loss",
        xaxis_title="Estimated Loss ($)",
        yaxis_title="Investigation ID",
        height=max(400, len(investigations_df) * 25),
        plot_bgcolor='rgba(0,0,0,0)',
        showlegend=False,
    )
    
    return fig


def create_claims_timeline(claims_df):
    """Create claims filing timeline."""
    if claims_df is None or len(claims_df) == 0:
        fig = go.Figure()
        return fig
    
    df = claims_df.copy()
    df["filed_date"] = pd.to_datetime(df["filed_date"])
    daily = df.groupby(df["filed_date"].dt.to_period("M")).size().reset_index()
    daily.columns = ["month", "count"]
    daily["month"] = daily["month"].astype(str)
    
    fig = go.Figure(data=go.Scatter(
        x=daily["month"],
        y=daily["count"],
        mode='lines+markers',
        line=dict(color='#45B7D1', width=2),
        marker=dict(size=8),
        fill='tozeroy',
        fillcolor='rgba(69, 183, 209, 0.1)',
    ))
    
    fig.update_layout(
        title="Claims Filed Over Time",
        xaxis_title="Month",
        yaxis_title="Number of Claims",
        height=350,
        plot_bgcolor='rgba(0,0,0,0)',
    )
    
    return fig


def create_payment_analysis(payments_df, invoices_df):
    """Create payment vs invoice analysis."""
    if payments_df is None or invoices_df is None:
        return go.Figure()
    
    merged = payments_df.merge(invoices_df[["invoice_id", "total_amount"]], on="invoice_id", how="left")
    merged = merged.dropna(subset=["total_amount", "payment_amount"])
    
    # Sample for performance
    if len(merged) > 1000:
        merged = merged.sample(1000, random_state=42)
    
    overpayment = merged["payment_amount"] > merged["total_amount"] * 1.1
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=merged[~overpayment]["total_amount"],
        y=merged[~overpayment]["payment_amount"],
        mode='markers',
        name='Normal',
        marker=dict(color='#96CEB4', size=5, opacity=0.5),
    ))
    
    fig.add_trace(go.Scatter(
        x=merged[overpayment]["total_amount"],
        y=merged[overpayment]["payment_amount"],
        mode='markers',
        name='Overpayment ⚠️',
        marker=dict(color='#FF6B6B', size=8, symbol='x'),
    ))
    
    # Reference line
    max_val = max(merged["total_amount"].max(), merged["payment_amount"].max())
    fig.add_trace(go.Scatter(
        x=[0, max_val], y=[0, max_val],
        mode='lines', name='Expected',
        line=dict(dash='dash', color='gray'),
    ))
    
    fig.update_layout(
        title="Payment vs Invoice Amount Analysis",
        xaxis_title="Invoice Amount ($)",
        yaxis_title="Payment Amount ($)",
        height=400,
        plot_bgcolor='rgba(0,0,0,0)',
    )
    
    return fig