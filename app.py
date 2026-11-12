import streamlit as st
import pandas as pd
import numpy as np
import networkx as nx
import os
import json
from datetime import datetime

from utils.data_generator import generate_all_data
from utils.network_analysis import (
    build_entity_graph, compute_centrality_metrics,
    detect_communities, score_suspicious_networks,
    get_node_neighborhood, compute_evaluation_metrics,
    find_shortest_paths
)
from utils.visualization import (
    create_network_plotly, create_suspicion_heatmap,
    create_entity_distribution_chart, create_risk_distribution,
    create_fraud_type_pie, create_investigation_timeline,
    create_claims_timeline, create_payment_analysis
)

# Page config
st.set_page_config(
    page_title="🔍 Fraud Network Detector",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        background: linear-gradient(90deg, #FF6B6B, #4ECDC4);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        padding: 1rem 0;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.2rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 0.5rem 0;
    }
    .metric-value {
        font-size: 2rem;
        font-weight: bold;
    }
    .metric-label {
        font-size: 0.9rem;
        opacity: 0.9;
    }
    .status-badge {
        padding: 2px 8px;
        border-radius: 12px;
        font-size: 0.8rem;
        font-weight: bold;
    }
    .fraud-alert {
        background: #FFE5E5;
        border-left: 4px solid #FF6B6B;
        padding: 1rem;
        border-radius: 0 8px 8px 0;
        margin: 0.5rem 0;
    }
    .safe-badge {
        background: #E5FFE5;
        border-left: 4px solid #4ECDC4;
        padding: 1rem;
        border-radius: 0 8px 8px 0;
        margin: 0.5rem 0;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px;
        padding: 8px 16px;
    }
</style>
""", unsafe_allow_html=True)


def load_data():
    """Load data from CSV files or session state."""
    if "datasets" in st.session_state:
        return st.session_state["datasets"]
    
    data_dir = "data"
    if os.path.exists(os.path.join(data_dir, "claimants.csv")):
        datasets = {}
        file_list = [
            "claimants", "policies", "claims", "incidents", "providers",
            "invoices", "payments", "relationship_edges", "fraud_indicators",
            "investigations", "ground_truth"
        ]
        for f in file_list:
            fpath = os.path.join(data_dir, f"{f}.csv")
            if os.path.exists(fpath):
                datasets[f] = pd.read_csv(fpath)
        
        val_path = os.path.join(data_dir, "validation_summary.json")
        if os.path.exists(val_path):
            with open(val_path, "r") as vf:
                datasets["validation_summary"] = json.load(vf)
        
        st.session_state["datasets"] = datasets
        return datasets
    
    return None


def main():
    # Header
    st.markdown('<div class="main-header">🔍 Insurance Fraud Network Detection System</div>', unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: #666;'>Link claimants, policies, incidents, providers and payments to surface suspicious networks</p>", unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.image("https://img.icons8.com/color/96/000000/search-in-list.png", width=80)
        st.title("Navigation")
        
        page = st.radio(
            "Select Module",
            [
                "🏠 Dashboard",
                "📊 Data Explorer",
                "🕸️ Network Analysis",
                "🔴 Fraud Detection",
                "🔍 Investigation Center",
                "📈 Evaluation Metrics",
                "⚙️ System Admin",
            ],
            index=0,
        )
        
        st.markdown("---")
        
        # Sample Data Button
        st.subheader("🗄️ Data Management")
        
        col1, col2 = st.columns(2)
        with col1:
            scale = st.selectbox("Scale", [0.1, 0.25, 0.5, 1.0], index=0, help="Data scale factor")
        with col2:
            seed = st.number_input("Seed", value=20261022, help="Random seed")
        
        if st.button("🎲 Load Sample Data", type="primary", use_container_width=True):
            with st.spinner("Generating synthetic data..."):
                datasets = generate_all_data(seed=seed, scale=scale, data_dir="data")
                st.session_state["datasets"] = datasets
                st.session_state["graph"] = None
                st.session_state["analysis_done"] = False
                st.success(f"✅ Data generated successfully! Scale: {scale}")
                st.rerun()
        
        if st.button("🗑️ Clear All Data", use_container_width=True):
            for key in ["datasets", "graph", "analysis_done", "communities", "scores", "metrics_df"]:
                if key in st.session_state:
                    del st.session_state[key]
            import shutil
            if os.path.exists("data"):
                shutil.rmtree("data")
            st.success("Data cleared!")
            st.rerun()
        
        st.markdown("---")
        st.markdown("**Built for Hackathon** 🏆")
        st.markdown("v1.0.0 | Seed: 20261022")
    
    # Load data
    datasets = load_data()
    
    if datasets is None:
        st.warning("⚠️ No data loaded. Click **'Load Sample Data'** in the sidebar to generate synthetic data.")
        
        st.markdown("### 📋 System Overview")
        st.markdown("""
        This system provides end-to-end insurance fraud network detection:
        
        1. **Data Generation** - Synthetic claimants, policies, claims, incidents, providers, invoices, payments
        2. **Network Building** - Link entities through relationships and shared attributes
        3. **Community Detection** - Identify clusters of connected entities
        4. **Fraud Scoring** - Rule-based scoring of suspicious networks
        5. **Investigation** - Deep-dive tools for analysts
        6. **Evaluation** - Precision/recall metrics against ground truth
        
        **Key Features:**
        - 🔗 Entity linking across 10+ entity types
        - 🕸️ Interactive network visualization
        - 📊 Community detection algorithms
        - ⚠️ Planted fraud patterns with ground truth
        - 📈 Measurable evaluation metrics
        - 🔍 Full traceability and audit trail
        """)
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.markdown("""
            <div class="metric-card">
                <div class="metric-value">2,000</div>
                <div class="metric-label">Claimants (Full)</div>
            </div>
            """, unsafe_allow_html=True)
        with col2:
            st.markdown("""
            <div class="metric-card">
                <div class="metric-value">5,000</div>
                <div class="metric-label">Claims (Full)</div>
            </div>
            """, unsafe_allow_html=True)
        with col3:
            st.markdown("""
            <div class="metric-card">
                <div class="metric-value">500</div>
                <div class="metric-label">Providers (Full)</div>
            </div>
            """, unsafe_allow_html=True)
        with col4:
            st.markdown("""
            <div class="metric-card">
                <div class="metric-value">8,000</div>
                <div class="metric-label">Edges (Full)</div>
            </div>
            """, unsafe_allow_html=True)
        return
    
    # Route to pages
    if page == "🏠 Dashboard":
        render_dashboard(datasets)
    elif page == "📊 Data Explorer":
        render_data_explorer(datasets)
    elif page == "🕸️ Network Analysis":
        render_network_analysis(datasets)
    elif page == "🔴 Fraud Detection":
        render_fraud_detection(datasets)
    elif page == "🔍 Investigation Center":
        render_investigation_center(datasets)
    elif page == "📈 Evaluation Metrics":
        render_evaluation(datasets)
    elif page == "⚙️ System Admin":
        render_admin(datasets)


def render_dashboard(datasets):
    """Main dashboard with overview metrics."""
    st.header("🏠 Executive Dashboard")
    
    # Key Metrics Row
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        n_claimants = len(datasets.get("claimants", []))
        st.metric("👥 Claimants", f"{n_claimants:,}")
    with col2:
        n_claims = len(datasets.get("claims", []))
        st.metric("📋 Claims", f"{n_claims:,}")
    with col3:
        n_providers = len(datasets.get("providers", []))
        st.metric("🏥 Providers", f"{n_providers:,}")
    with col4:
        gt = datasets.get("ground_truth", pd.DataFrame())
        n_fraud = len(gt[gt["is_fraud"] == True]) if len(gt) > 0 else 0
        st.metric("⚠️ Fraud Indicators", f"{n_fraud}")
    with col5:
        n_investigations = len(datasets.get("investigations", []))
        st.metric("🔍 Investigations", f"{n_investigations}")
    
    st.markdown("---")
    
    # Charts Row 1
    col1, col2 = st.columns(2)
    
    with col1:
        fig = create_entity_distribution_chart(datasets)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        gt = datasets.get("ground_truth", pd.DataFrame())
        fig = create_fraud_type_pie(gt)
        st.plotly_chart(fig, use_container_width=True)
    
    # Charts Row 2
    col1, col2 = st.columns(2)
    
    with col1:
        fig = create_risk_distribution(
            datasets.get("claimants"),
            datasets.get("providers")
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        fig = create_claims_timeline(datasets.get("claims"))
        st.plotly_chart(fig, use_container_width=True)
    
    # Payment Analysis
    fig = create_payment_analysis(
        datasets.get("payments"),
        datasets.get("invoices")
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Recent fraud alerts
    st.subheader("🚨 Recent Fraud Indicators")
    fi = datasets.get("fraud_indicators", pd.DataFrame())
    if len(fi) > 0:
        critical = fi[fi["severity"] == "Critical"].head(10)
        if len(critical) > 0:
            for _, row in critical.iterrows():
                st.markdown(f"""
                <div class="fraud-alert">
                    <strong>🔴 {row['indicator_type'].upper()}</strong> | 
                    {row['entity_type']}: {row['entity_id']} | 
                    Network: {row['network_id']} | 
                    {row['description']}
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info("No critical fraud indicators found.")
    else:
        st.info("No fraud indicators available.")


def render_data_explorer(datasets):
    """Data exploration and browsing interface."""
    st.header("📊 Data Explorer")
    
    entity_choice = st.selectbox(
        "Select Entity to Explore",
        ["claimants", "policies", "claims", "incidents", "providers",
         "invoices", "payments", "relationship_edges", "fraud_indicators",
         "investigations", "ground_truth"]
    )
    
    df = datasets.get(entity_choice, pd.DataFrame())
    
    if len(df) == 0:
        st.warning(f"No data available for {entity_choice}")
        return
    
    # Summary stats
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Records", f"{len(df):,}")
    with col2:
        st.metric("Columns", f"{len(df.columns)}")
    with col3:
        st.metric("Memory", f"{df.memory_usage(deep=True).sum() / 1024:.1f} KB")
    
    # Schema info
    with st.expander("📐 Schema / Data Dictionary"):
        schema_df = pd.DataFrame({
            "Column": df.columns,
            "Type": df.dtypes.astype(str).values,
            "Non-Null": df.notna().sum().values,
            "Null": df.isna().sum().values,
            "Unique": df.nunique().values,
            "Sample": [str(df[col].iloc[0])[:50] if len(df) > 0 else "" for col in df.columns],
        })
        st.dataframe(schema_df, use_container_width=True)
    
    # Filters
    st.subheader("🔎 Filter & Search")
    
    col1, col2 = st.columns(2)
    with col1:
        filter_col = st.selectbox("Filter Column", ["None"] + list(df.columns))
    with col2:
        if filter_col != "None":
            unique_vals = df[filter_col].unique()
            if len(unique_vals) <= 50:
                filter_val = st.multiselect("Filter Values", unique_vals)
            else:
                filter_val = st.text_input("Search Value (contains)")
    
    filtered_df = df.copy()
    if filter_col != "None":
        if isinstance(filter_val, list) and len(filter_val) > 0:
            filtered_df = filtered_df[filtered_df[filter_col].isin(filter_val)]
        elif isinstance(filter_val, str) and filter_val:
            filtered_df = filtered_df[filtered_df[filter_col].astype(str).str.contains(filter_val, case=False, na=False)]
    
    st.dataframe(filtered_df, use_container_width=True, height=400)
    
    # Download
    csv = filtered_df.to_csv(index=False)
    st.download_button(
        f"📥 Download {entity_choice} CSV",
        csv, f"{entity_choice}.csv", "text/csv",
        use_container_width=True
    )
    
    # Quick stats for numeric columns
    numeric_cols = filtered_df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 0:
        st.subheader("📊 Numeric Statistics")
        st.dataframe(filtered_df[numeric_cols].describe(), use_container_width=True)


def render_network_analysis(datasets):
    """Network analysis and visualization."""
    st.header("🕸️ Network Analysis")
    
    edges = datasets.get("relationship_edges", pd.DataFrame())
    if len(edges) == 0:
        st.warning("No relationship edges available.")
        return
    
    # Build graph
    with st.spinner("Building entity graph..."):
        G = build_entity_graph(
            edges,
            datasets.get("claimants"),
            datasets.get("providers"),
            datasets.get("claims")
        )
        st.session_state["graph"] = G
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Nodes", f"{G.number_of_nodes():,}")
    with col2:
        st.metric("Edges", f"{G.number_of_edges():,}")
    with col3:
        components = nx.number_connected_components(G)
        st.metric("Components", f"{components:,}")
    with col4:
        density = nx.density(G)
        st.metric("Density", f"{density:.4f}")
    
    st.markdown("---")
    
    tab1, tab2, tab3, tab4 = st.tabs(["🌐 Network Viz", "📊 Centrality", "👥 Communities", "🔗 Path Finder"])
    
    with tab1:
        st.subheader("Interactive Network Visualization")
        max_nodes = st.slider("Max nodes to display", 50, 1000, 200, 50)
        
        # Filter options
        edge_type_filter = st.multiselect(
            "Filter by relationship type",
            edges["relationship_type"].unique(),
            default=[]
        )
        
        if edge_type_filter:
            filtered_edges = edges[edges["relationship_type"].isin(edge_type_filter)]
            G_filtered = build_entity_graph(
                filtered_edges,
                datasets.get("claimants"),
                datasets.get("providers")
            )
        else:
            G_filtered = G
        
        fig = create_network_plotly(G_filtered, "Entity Relationship Network", max_nodes)
        st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        st.subheader("Node Centrality Metrics")
        
        with st.spinner("Computing centrality metrics..."):
            # Use a sampled subgraph for performance
            if G.number_of_nodes() > 500:
                degrees = dict(G.degree())
                top_nodes = sorted(degrees, key=degrees.get, reverse=True)[:500]
                G_sample = G.subgraph(top_nodes).copy()
            else:
                G_sample = G
            
            metrics_df = compute_centrality_metrics(G_sample)
            st.session_state["metrics_df"] = metrics_df
        
        if len(metrics_df) > 0:
            # Sort options
            sort_by = st.selectbox(
                "Sort by metric",
                ["degree", "degree_centrality", "betweenness_centrality", "pagerank", "risk_score"]
            )
            
            sorted_df = metrics_df.sort_values(sort_by, ascending=False).head(50)
            st.dataframe(sorted_df, use_container_width=True)
            
            # Top nodes chart
            top_20 = sorted_df.head(20)
            import plotly.express as px
            fig = px.bar(
                top_20, x="node_id", y=sort_by,
                color="entity_type",
                title=f"Top 20 Nodes by {sort_by.replace('_', ' ').title()}"
            )
            fig.update_layout(xaxis_tickangle=-45, height=400)
            st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        st.subheader("Community Detection")
        
        with st.spinner("Detecting communities..."):
            if G.number_of_nodes() > 1000:
                degrees = dict(G.degree())
                top_nodes = sorted(degrees, key=degrees.get, reverse=True)[:1000]
                G_comm = G.subgraph(top_nodes).copy()
            else:
                G_comm = G
            
            node_community, community_nodes = detect_communities(G_comm)
            st.session_state["communities"] = (node_community, community_nodes)
        
        st.info(f"Detected **{len(community_nodes)}** communities")
        
        # Community size distribution
        sizes = [len(nodes) for nodes in community_nodes.values()]
        import plotly.express as px
        fig = px.histogram(x=sizes, nbins=30, title="Community Size Distribution",
                          labels={"x": "Community Size", "y": "Count"})
        st.plotly_chart(fig, use_container_width=True)
        
        # Community details table
        comm_details = []
        for cid, nodes in community_nodes.items():
            etypes = pd.Series([G_comm.nodes[n].get("entity_type", "unknown") for n in nodes if n in G_comm.nodes]).value_counts()
            comm_details.append({
                "Community": cid,
                "Size": len(nodes),
                "Claimants": etypes.get("claimant", 0),
                "Providers": etypes.get("provider", 0),
                "Claims": etypes.get("claim", 0),
            })
        
        comm_df = pd.DataFrame(comm_details).sort_values("Size", ascending=False)
        st.dataframe(comm_df.head(30), use_container_width=True)
    
    with tab4:
        st.subheader("Path Finder")
        st.info("Find connections between two entities")
        
        nodes_list = list(G.nodes())[:500]  # Limit for dropdown
        
        col1, col2 = st.columns(2)
        with col1:
            source = st.selectbox("Source Entity", nodes_list, index=0)
        with col2:
            target = st.selectbox("Target Entity", nodes_list, index=min(1, len(nodes_list)-1))
        
        if st.button("Find Path"):
            paths = find_shortest_paths(G, source, target)
            if paths:
                st.success(f"Found {len(paths)} shortest path(s) of length {len(paths[0])-1}")
                for i, path in enumerate(paths[:5]):
                    st.markdown(f"**Path {i+1}:** {' → '.join(path)}")
                
                # Visualize the path subgraph
                path_nodes = set()
                for p in paths[:3]:
                    path_nodes.update(p)
                path_G = G.subgraph(path_nodes).copy()
                fig = create_network_plotly(path_G, "Path Subgraph")
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("No path found between these entities.")


def render_fraud_detection(datasets):
    """Fraud detection and suspicious network scoring."""
    st.header("🔴 Fraud Detection Engine")
    
    edges = datasets.get("relationship_edges", pd.DataFrame())
    gt = datasets.get("ground_truth", pd.DataFrame())
    
    if len(edges) == 0:
        st.warning("No data available. Load sample data first.")
        return
    
    # Build and analyze
    if st.button("🚀 Run Fraud Detection Analysis", type="primary", use_container_width=True):
        with st.spinner("Building network and analyzing patterns..."):
            G = build_entity_graph(
                edges,
                datasets.get("claimants"),
                datasets.get("providers"),
                datasets.get("claims")
            )
            
            # Sample for performance if graph is very large
            if G.number_of_nodes() > 1000:
                degrees = dict(G.degree())
                top_nodes = sorted(degrees, key=degrees.get, reverse=True)[:1000]
                G = G.subgraph(top_nodes).copy()
            
            communities = detect_communities(G)
            scores = score_suspicious_networks(G, communities, gt)
            
            st.session_state["graph"] = G
            st.session_state["communities"] = communities
            st.session_state["scores"] = scores
            st.session_state["analysis_done"] = True
    
    if not st.session_state.get("analysis_done"):
        st.info("Click the button above to run the fraud detection analysis.")
        
        fi = datasets.get("fraud_indicators", pd.DataFrame())
        if len(fi) > 0:
            st.subheader("📋 Known Fraud Indicators")
            st.dataframe(fi, use_container_width=True)
        return
    
    scores = st.session_state.get("scores", pd.DataFrame())
    G = st.session_state.get("graph")
    
    if scores is None or len(scores) == 0:
        st.warning("No communities detected.")
        return
    
    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        high_risk = len(scores[scores["suspicion_score"] >= 70])
        st.metric("🔴 High Risk Networks", high_risk)
    with col2:
        medium_risk = len(scores[(scores["suspicion_score"] >= 50) & (scores["suspicion_score"] < 70)])
        st.metric("🟡 Medium Risk", medium_risk)
    with col3:
        low_risk = len(scores[scores["suspicion_score"] < 50])
        st.metric("🟢 Low Risk", low_risk)
    with col4:
        known_found = len(scores[scores["is_known_fraud"] == True])
        st.metric("✅ Known Fraud Found", known_found)
    
    st.markdown("---")
    
    # Suspicion score chart
    fig = create_suspicion_heatmap(scores)
    st.plotly_chart(fig, use_container_width=True)
    
    # Detailed results table
    st.subheader("📊 Detailed Network Scores")
    
    available_actions = scores["recommended_action"].dropna().unique().tolist()
    
    # Default to showing ALL available actions so table is never empty
    risk_filter = st.multiselect(
        "Filter by action (leave empty or select all to view all networks)",
        options=available_actions,
        default=available_actions
    )
    
    if risk_filter:
        filtered_scores = scores[scores["recommended_action"].isin(risk_filter)]
    else:
        filtered_scores = scores.copy()
    
    st.caption(f"Showing **{len(filtered_scores)}** of **{len(scores)}** total networks")
    
    if len(filtered_scores) > 0:
        st.dataframe(
            filtered_scores.style.background_gradient(subset=["suspicion_score"], cmap="RdYlGn_r"),
            use_container_width=True,
            height=300,
        )
    else:
        st.info("No networks match the selected filter. Showing all networks below:")
        st.dataframe(scores, use_container_width=True, height=300)
    
    st.markdown("---")
    
    # Network deep dive
    st.subheader("🔍 Network Deep Dive")
    
    community_ids = scores["community_id"].tolist()
    if not community_ids:
        st.warning("No communities available to inspect.")
        return
    
    selected_community = st.selectbox(
        "Select Community ID to Inspect Network & Evidence",
        community_ids
    )
    
    if selected_community is not None:
        comm_row = scores[scores["community_id"] == selected_community].iloc[0]
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"### 🕸️ Community {selected_community} Summary")
            st.markdown(f"""
            - **Nodes:** `{comm_row['num_nodes']}` | **Edges:** `{comm_row['num_edges']}`
            - **Graph Density:** `{comm_row['density']}`
            - **Suspicion Score:** `{comm_row['suspicion_score']} / 100`
            - **Action Recommendation:** `{comm_row['recommended_action']}`
            """)
        
        with col2:
            st.markdown(f"### 👥 Entity & Risk Breakdown")
            st.markdown(f"""
            - **Claimants:** `{comm_row['num_claimants']}` | **Providers:** `{comm_row['num_providers']}`
            - **Average Risk Score:** `{comm_row['avg_risk_score']}`
            - **Shared Contact Links:** `{comm_row['shared_contacts']}`
            - **Matches Known Fraud Ground Truth:** `{'✅ Yes' if comm_row['is_known_fraud'] else '❌ No'}`
            """)
        
        st.markdown("#### 📝 Flagged Evidence & Triggered Rules")
        evidence_text = str(comm_row.get("evidence", "No evidence available."))
        if float(comm_row.get("suspicion_score", 0)) >= 50:
            st.error(evidence_text)
        else:
            st.info(evidence_text)
        
        # Visualize community subgraph
        communities = st.session_state.get("communities", ({}, {}))
        _, community_nodes = communities
        if selected_community in community_nodes and G is not None:
            comm_nodes = community_nodes[selected_community]
            valid_nodes = [n for n in comm_nodes if n in G]
            if valid_nodes:
                sub_G = G.subgraph(valid_nodes).copy()
                fig = create_network_plotly(sub_G, f"Interactive Graph - Community {selected_community}")
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("No graph nodes available to display for this community.")

def render_investigation_center(datasets):
    """Investigation management interface."""
    st.header("🔍 Investigation Center")
    
    investigations = datasets.get("investigations", pd.DataFrame())
    
    if len(investigations) == 0:
        st.warning("No investigations available.")
        return
    
    # Overview metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Investigations", len(investigations))
    with col2:
        active = len(investigations[investigations["status"].isin(["open", "in_progress"])])
        st.metric("Active", active)
    with col3:
        total_loss = investigations["estimated_loss"].sum()
        st.metric("Total Est. Loss", f"${total_loss:,.0f}")
    with col4:
        confirmed = len(investigations[investigations["status"] == "closed_confirmed"])
        st.metric("Confirmed Fraud", confirmed)
    
    st.markdown("---")
    
    tab1, tab2, tab3 = st.tabs(["📋 All Investigations", "📊 Timeline", "🔎 Detail View"])
    
    with tab1:
        # Filter by status
        status_filter = st.multiselect(
            "Filter by Status",
            investigations["status"].unique(),
            default=investigations["status"].unique()
        )
        
        filtered = investigations[investigations["status"].isin(status_filter)]
        
        st.dataframe(
            filtered.style.background_gradient(subset=["estimated_loss"], cmap="Reds"),
            use_container_width=True,
        )
    
    with tab2:
        fig = create_investigation_timeline(investigations)
        st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        selected_inv = st.selectbox(
            "Select Investigation",
            investigations["investigation_id"].tolist()
        )
        
        if selected_inv:
            inv_row = investigations[investigations["investigation_id"] == selected_inv].iloc[0]
            
            col1, col2 = st.columns(2)
            with col1:
                st.markdown(f"""
                ### Investigation: {inv_row['investigation_id']}
                
                | Field | Value |
                |-------|-------|
                | Network | {inv_row['network_id']} |
                | Fraud Type | {inv_row['fraud_type']} |
                | Status | {inv_row['status']} |
                | Priority | {inv_row['priority']} |
                | Opened | {inv_row['opened_date']} |
                | Investigator | {inv_row['assigned_investigator']} |
                """)
            
            with col2:
                st.markdown(f"""
                ### Financial Impact
                
                | Metric | Value |
                |--------|-------|
                | Entities Involved | {inv_row['entity_count']} |
                | Estimated Loss | ${inv_row['estimated_loss']:,.2f} |
                
                **Notes:** {inv_row['notes']}
                """)
            
            # Show related ground truth
            gt = datasets.get("ground_truth", pd.DataFrame())
            if len(gt) > 0:
                related_gt = gt[gt["network_id"] == inv_row["network_id"]]
                if len(related_gt) > 0:
                    st.subheader("Related Ground Truth Records")
                    st.dataframe(related_gt, use_container_width=True)
            
            # Event history (simulated)
            st.subheader("📜 Event History / Audit Trail")
            events = [
                {"timestamp": inv_row["opened_date"], "event": "Investigation opened", "actor": inv_row["assigned_investigator"]},
                {"timestamp": inv_row["opened_date"], "event": f"Flagged as {inv_row['fraud_type']}", "actor": "System"},
                {"timestamp": inv_row["opened_date"], "event": f"Priority set to {inv_row['priority']}", "actor": "System"},
            ]
            if inv_row["status"] in ["closed_confirmed", "closed_cleared"]:
                events.append({"timestamp": "2025-10-01", "event": f"Investigation {inv_row['status']}", "actor": inv_row["assigned_investigator"]})
            
            st.dataframe(pd.DataFrame(events), use_container_width=True)


def render_evaluation(datasets):
    """Evaluation metrics and performance analysis."""
    st.header("📈 Evaluation Metrics")
    
    scores = st.session_state.get("scores")
    gt = datasets.get("ground_truth", pd.DataFrame())
    
    if scores is None:
        st.warning("Run Fraud Detection analysis first to see evaluation metrics.")
        st.info("Go to **🔴 Fraud Detection** and click **Run Fraud Detection Analysis**")
        return
    
    # Compute evaluation metrics
    eval_metrics = compute_evaluation_metrics(scores, gt)
    
    st.subheader("🎯 Detection Performance")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Precision", f"{eval_metrics['precision']:.2%}")
    with col2:
        st.metric("Recall", f"{eval_metrics['recall']:.2%}")
    with col3:
        st.metric("F1 Score", f"{eval_metrics['f1_score']:.2%}")
    with col4:
        st.metric("Accuracy", f"{eval_metrics['accuracy']:.2%}")
    
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Confusion matrix
        st.subheader("Confusion Matrix")
        import plotly.figure_factory as ff
        
        cm = [
            [eval_metrics["true_negatives"], eval_metrics["false_positives"]],
            [eval_metrics["false_negatives"], eval_metrics["true_positives"]],
        ]
        
        fig = go.Figure(data=go.Heatmap(
            z=cm,
            x=["Predicted Negative", "Predicted Positive"],
            y=["Actual Negative", "Actual Positive"],
            text=[[str(v) for v in row] for row in cm],
            texttemplate="%{text}",
            colorscale="Blues",
            showscale=True,
        ))
        fig.update_layout(title="Confusion Matrix", height=350)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Score distribution
        st.subheader("Score Distribution")
        import plotly.express as px
        
        fig = px.histogram(
            scores, x="suspicion_score",
            color="is_known_fraud",
            nbins=30,
            title="Suspicion Score Distribution",
            labels={"is_known_fraud": "Known Fraud"},
            color_discrete_map={True: "#FF6B6B", False: "#4ECDC4"},
        )
        fig.update_layout(height=350)
        st.plotly_chart(fig, use_container_width=True)
    
    # Threshold analysis
    st.subheader("🎚️ Threshold Analysis")
    threshold = st.slider("Suspicion Score Threshold", 0, 100, 50)
    
    pred_positive = scores[scores["suspicion_score"] >= threshold]
    pred_negative = scores[scores["suspicion_score"] < threshold]
    
    tp = len(pred_positive[pred_positive["is_known_fraud"] == True])
    fp = len(pred_positive[pred_positive["is_known_fraud"] == False])
    fn = len(pred_negative[pred_negative["is_known_fraud"] == True])
    tn = len(pred_negative[pred_negative["is_known_fraud"] == False])
    
    p = tp / (tp + fp) if (tp + fp) > 0 else 0
    r = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric(f"Precision @ {threshold}", f"{p:.2%}")
    with col2:
        st.metric(f"Recall @ {threshold}", f"{r:.2%}")
    with col3:
        st.metric(f"F1 @ {threshold}", f"{f1:.2%}")
    
    # Validation summary
    st.subheader("✅ Data Validation Summary")
    val = datasets.get("validation_summary")
    if val:
        col1, col2 = st.columns(2)
        with col1:
            st.json(val.get("row_counts", {}))
        with col2:
            checks = val.get("checks", [])
            for check in checks:
                icon = "✅" if check["passed"] else "❌"
                st.markdown(f"{icon} **{check['check']}**: {check['detail']}")
    
    # Ground truth stats
    st.subheader("📊 Ground Truth Analysis")
    if len(gt) > 0:
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**By Fraud Type:**")
            fraud_counts = gt[gt["is_fraud"] == True]["fraud_type"].value_counts()
            st.dataframe(fraud_counts.reset_index().rename(columns={"index": "Type", "fraud_type": "Type", "count": "Count"}))
        
        with col2:
            st.markdown("**By Entity Type:**")
            entity_counts = gt["entity_type"].value_counts()
            st.dataframe(entity_counts.reset_index().rename(columns={"index": "Entity", "entity_type": "Entity", "count": "Count"}))


def render_admin(datasets):
    """System administration panel."""
    st.header("⚙️ System Administration")
    
    tab1, tab2, tab3, tab4 = st.tabs(["💾 Data Status", "📐 Schema", "🔄 Regenerate", "📋 Audit Log"])
    
    with tab1:
        st.subheader("Data Files Status")
        data_dir = "data"
        if os.path.exists(data_dir):
            files = os.listdir(data_dir)
            file_info = []
            for f in sorted(files):
                fpath = os.path.join(data_dir, f)
                size = os.path.getsize(fpath)
                modified = datetime.fromtimestamp(os.path.getmtime(fpath))
                file_info.append({
                    "File": f,
                    "Size": f"{size/1024:.1f} KB",
                    "Modified": modified.strftime("%Y-%m-%d %H:%M:%S"),
                })
            st.dataframe(pd.DataFrame(file_info), use_container_width=True)
        else:
            st.warning("No data directory found.")
    
    with tab2:
        st.subheader("Entity Schema Documentation")
        
        schemas = {
            "Claimant": {
                "claimant_id": "STRING, PK, Required, Format: CLM-2026-NNNNNN, Unique",
                "first_name": "STRING, Required",
                "last_name": "STRING, Required",
                "full_name": "STRING, Required, Derived",
                "date_of_birth": "DATE, Required, YYYY-MM-DD",
                "address": "STRING, Required",
                "phone": "STRING, Required, Unique (baseline)",
                "email": "STRING, Required, Unique (baseline)",
                "bank_account": "STRING, Required, Format: BANK-NNNNNNNNN",
                "risk_score": "FLOAT, 0-100, Derived",
                "status": "ENUM[active, inactive, suspended]",
            },
            "Policy": {
                "policy_id": "STRING, PK, Required, Format: POL-2026-NNNNNN",
                "claimant_id": "STRING, FK->Claimant, Required",
                "policy_type": "ENUM[Auto, Home, Health, Life, Travel, Commercial]",
                "coverage_level": "ENUM[Basic, Standard, Premium, Comprehensive]",
                "premium_amount": "FLOAT, USD, Required, > 0",
                "coverage_limit": "FLOAT, USD, Required",
                "status": "ENUM[active, expired, cancelled, lapsed]",
            },
            "Claim": {
                "claim_id": "STRING, PK, Required, Format: CLN-2026-NNNNNN",
                "policy_id": "STRING, FK->Policy, Required",
                "claimant_id": "STRING, FK->Claimant, Required",
                "incident_id": "STRING, FK->Incident, Required",
                "claim_type": "ENUM[Auto, Property, Medical, Liability, Workers Comp]",
                "claimed_amount": "FLOAT, USD, Required, > 0",
                "approved_amount": "FLOAT, USD, >= 0",
                "status": "ENUM[submitted, under_review, approved, denied, paid, closed, appealed]",
            },
            "Provider": {
                "provider_id": "STRING, PK, Required, Format: PRV-NNNNNN",
                "provider_name": "STRING, Required",
                "provider_type": "ENUM[Medical, Auto Repair, Legal, Rehabilitation, Diagnostic, Pharmacy]",
                "license_number": "STRING, Required, Format: LIC-NNNNNN",
                "license_status": "ENUM[active, expired, suspended, revoked]",
                "risk_score": "FLOAT, 0-100, Derived",
            },
            "Invoice": {
                "invoice_id": "STRING, PK, Format: INV-PRV-<provider>-NNNNNN",
                "claim_id": "STRING, FK->Claim",
                "provider_id": "STRING, FK->Provider",
                "total_amount": "FLOAT, USD, Derived (subtotal + tax)",
            },
            "Payment": {
                "payment_id": "STRING, PK, Format: PAY-2026-NNNNNN",
                "invoice_id": "STRING, FK->Invoice",
                "claim_id": "STRING, FK->Claim",
                "payment_amount": "FLOAT, USD",
                "status": "ENUM[pending, processed, completed, failed, reversed]",
            },
            "RelationshipEdge": {
                "edge_id": "STRING, PK, Format: EDGE-NNNNNNNN",
                "source_type": "STRING, Entity type",
                "source_id": "STRING, FK->Source entity",
                "target_type": "STRING, Entity type",
                "target_id": "STRING, FK->Target entity",
                "relationship_type": "ENUM[holds_policy, filed_claim, serviced_by, shared_contact, ...]",
                "weight": "FLOAT, 0-1",
            },
        }
        
        for entity, fields in schemas.items():
            with st.expander(f"📐 {entity}"):
                for field, desc in fields.items():
                    st.markdown(f"- **{field}**: {desc}")
    
    with tab3:
        st.subheader("Regenerate Data")
        st.markdown("Regenerate data with the same seed to verify reproducibility.")
        
        col1, col2 = st.columns(2)
        with col1:
            regen_seed = st.number_input("Seed for regeneration", value=20261022, key="regen_seed")
        with col2:
            regen_scale = st.selectbox("Scale", [0.1, 0.25, 0.5, 1.0], index=0, key="regen_scale")
        
        if st.button("🔄 Regenerate & Compare"):
            with st.spinner("Regenerating..."):
                # Generate first time
                data1 = generate_all_data(seed=regen_seed, scale=regen_scale, data_dir="data")
                rc1 = {k: len(v) for k, v in data1.items() if isinstance(v, pd.DataFrame)}
                
                # Generate second time
                data2 = generate_all_data(seed=regen_seed, scale=regen_scale, data_dir="data")
                rc2 = {k: len(v) for k, v in data2.items() if isinstance(v, pd.DataFrame)}
                
                st.success("✅ Regeneration complete!")
                
                # Compare
                comparison = []
                for key in rc1:
                    match = rc1[key] == rc2.get(key, -1)
                    comparison.append({
                        "Entity": key,
                        "Run 1 Count": rc1[key],
                        "Run 2 Count": rc2.get(key, "N/A"),
                        "Match": "✅" if match else "❌",
                    })
                
                st.dataframe(pd.DataFrame(comparison), use_container_width=True)
                
                st.session_state["datasets"] = data2
    
    with tab4:
        st.subheader("📋 System Audit Log")
        
        audit_entries = [
            {"timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "actor": "system", "action": "Application started", "details": "All modules initialized"},
        ]
        
        if "datasets" in st.session_state:
            audit_entries.append({
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "actor": "user",
                "action": "Data loaded",
                "details": f"Datasets in memory: {len(st.session_state['datasets'])} tables"
            })
        
        if st.session_state.get("analysis_done"):
            audit_entries.append({
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "actor": "system",
                "action": "Fraud analysis completed",
                "details": "Network analysis and community detection executed"
            })
        
        st.dataframe(pd.DataFrame(audit_entries), use_container_width=True)
        
        # Roles
        st.subheader("👥 Actor Roles")
        roles = pd.DataFrame([
            {"Role": "Customer/Account Holder", "Access": "View own claims, policies", "Permissions": "Read-only on own data"},
            {"Role": "Operations Analyst", "Access": "Full data explorer, network analysis", "Permissions": "Read all, run analyses"},
            {"Role": "Risk/Compliance Reviewer", "Access": "Fraud detection, investigations, evaluation", "Permissions": "Read all, manage investigations"},
            {"Role": "System Administrator", "Access": "All modules including admin", "Permissions": "Full access, data management"},
        ])
        st.dataframe(roles, use_container_width=True)


import plotly.graph_objects as go

if __name__ == "__main__":
    main()

