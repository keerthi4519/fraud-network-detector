import pandas as pd
import numpy as np
import networkx as nx
from collections import defaultdict, Counter
from datetime import datetime


def build_entity_graph(edges_df, claimants_df=None, providers_df=None, claims_df=None):
    """Build a NetworkX graph from relationship edges."""
    G = nx.Graph()
    
    # Add edges
    for _, row in edges_df.iterrows():
        source = f"{row['source_type']}:{row['source_id']}"
        target = f"{row['target_type']}:{row['target_id']}"
        
        G.add_edge(
            source, target,
            edge_id=row.get("edge_id", ""),
            relationship_type=row.get("relationship_type", "unknown"),
            weight=float(row.get("weight", 1.0)),
            created_date=row.get("created_date", ""),
        )
        
        G.nodes[source]["entity_type"] = row["source_type"]
        G.nodes[source]["entity_id"] = row["source_id"]
        G.nodes[target]["entity_type"] = row["target_type"]
        G.nodes[target]["entity_id"] = row["target_id"]
    
    # Enrich nodes with entity data
    if claimants_df is not None:
        for _, row in claimants_df.iterrows():
            node_id = f"claimant:{row['claimant_id']}"
            if node_id in G.nodes:
                G.nodes[node_id]["name"] = row.get("full_name", "")
                G.nodes[node_id]["risk_score"] = float(row.get("risk_score", 0))
                G.nodes[node_id]["city"] = row.get("city", "")
                G.nodes[node_id]["status"] = row.get("status", "")
    
    if providers_df is not None:
        for _, row in providers_df.iterrows():
            node_id = f"provider:{row['provider_id']}"
            if node_id in G.nodes:
                G.nodes[node_id]["name"] = row.get("provider_name", "")
                G.nodes[node_id]["risk_score"] = float(row.get("risk_score", 0))
                G.nodes[node_id]["provider_type"] = row.get("provider_type", "")
    
    return G


def compute_centrality_metrics(G):
    """Compute various centrality metrics for all nodes."""
    metrics = {}
    
    if len(G.nodes) == 0:
        return pd.DataFrame()
    
    # Degree centrality
    degree = nx.degree_centrality(G)
    
    # Betweenness centrality (sampled for large graphs)
    if len(G.nodes) > 1000:
        betweenness = nx.betweenness_centrality(G, k=min(100, len(G.nodes)))
    else:
        betweenness = nx.betweenness_centrality(G)
    
    # Closeness centrality
    if nx.is_connected(G):
        closeness = nx.closeness_centrality(G)
    else:
        closeness = {}
        for component in nx.connected_components(G):
            subgraph = G.subgraph(component)
            if len(subgraph) > 1:
                cc = nx.closeness_centrality(subgraph)
                closeness.update(cc)
            else:
                for node in component:
                    closeness[node] = 0.0
    
    # PageRank
    try:
        pagerank = nx.pagerank(G, max_iter=100)
    except:
        pagerank = {n: 1.0/len(G.nodes) for n in G.nodes}
    
    # Compile metrics
    records = []
    for node in G.nodes:
        records.append({
            "node_id": node,
            "entity_type": G.nodes[node].get("entity_type", "unknown"),
            "entity_id": G.nodes[node].get("entity_id", ""),
            "degree": G.degree(node),
            "degree_centrality": round(degree.get(node, 0), 6),
            "betweenness_centrality": round(betweenness.get(node, 0), 6),
            "closeness_centrality": round(closeness.get(node, 0), 6),
            "pagerank": round(pagerank.get(node, 0), 6),
            "risk_score": G.nodes[node].get("risk_score", 0),
        })
    
    return pd.DataFrame(records)


def detect_communities(G):
    """Detect communities using Louvain method."""
    if len(G.nodes) == 0:
        return {}, {}
    
    try:
        communities = nx.community.louvain_communities(G, seed=42)
    except:
        communities = list(nx.connected_components(G))
    
    # Map nodes to community IDs
    node_community = {}
    community_nodes = {}
    for i, comm in enumerate(communities):
        community_nodes[i] = list(comm)
        for node in comm:
            node_community[node] = i
    
    return node_community, community_nodes


def score_suspicious_networks(G, communities, ground_truth_df=None):
    """Score each community/network for suspiciousness."""
    _, community_nodes = communities
    
    scores = []
    for comm_id, nodes in community_nodes.items():
        if len(nodes) < 2:
            continue
        
        subgraph = G.subgraph(nodes)
        
        # Feature extraction
        num_nodes = len(nodes)
        num_edges = subgraph.number_of_edges()
        density = nx.density(subgraph)
        
        # Entity type distribution
        entity_types = Counter([G.nodes[n].get("entity_type", "unknown") for n in nodes])
        num_claimants = entity_types.get("claimant", 0)
        num_providers = entity_types.get("provider", 0)
        
        # Risk score aggregation
        risk_scores = [G.nodes[n].get("risk_score", 0) for n in nodes if G.nodes[n].get("risk_score", 0) > 0]
        avg_risk = np.mean(risk_scores) if risk_scores else 0
        max_risk = max(risk_scores) if risk_scores else 0
        
        # Relationship type distribution
        rel_types = Counter([G.edges[e].get("relationship_type", "unknown") for e in subgraph.edges])
        shared_contacts = rel_types.get("shared_contact", 0)
        
        # Compute suspicion score (rule-based)
        suspicion_score = 0
        evidence = []
        
        # Rule 1: High density with many nodes
        if density > 0.5 and num_nodes > 3:
            suspicion_score += 25
            evidence.append(f"High network density ({density:.2f}) with {num_nodes} nodes")
        
        # Rule 2: Shared contacts
        if shared_contacts > 0:
            suspicion_score += 20 * min(shared_contacts, 5) / 5
            evidence.append(f"{shared_contacts} shared contact relationships")
        
        # Rule 3: High average risk
        if avg_risk > 50:
            suspicion_score += 20
            evidence.append(f"High average risk score ({avg_risk:.1f})")
        
        # Rule 4: Multiple claimants linked to same provider
        if num_claimants > 2 and num_providers > 0:
            suspicion_score += 15
            evidence.append(f"{num_claimants} claimants linked to {num_providers} providers")
        
        # Rule 5: Unusual entity ratio
        if num_claimants > 0 and num_providers > 0:
            ratio = num_claimants / num_providers
            if ratio > 5:
                suspicion_score += 10
                evidence.append(f"High claimant-to-provider ratio ({ratio:.1f})")
        
        # Rule 6: Community size
        if num_nodes > 10:
            suspicion_score += 10
            evidence.append(f"Large network ({num_nodes} nodes)")
        
        suspicion_score = min(suspicion_score, 100)
        
        # Check against ground truth
        is_known_fraud = False
        fraud_network_id = None
        if ground_truth_df is not None and len(ground_truth_df) > 0:
            for node in nodes:
                entity_id = G.nodes[node].get("entity_id", "")
                matches = ground_truth_df[
                    (ground_truth_df["entity_id"] == entity_id) & 
                    (ground_truth_df["is_fraud"] == True)
                ]
                if len(matches) > 0:
                    is_known_fraud = True
                    fraud_network_id = matches.iloc[0].get("network_id", "")
                    break
        
        scores.append({
            "community_id": comm_id,
            "num_nodes": num_nodes,
            "num_edges": num_edges,
            "density": round(density, 4),
            "num_claimants": num_claimants,
            "num_providers": num_providers,
            "avg_risk_score": round(avg_risk, 2),
            "max_risk_score": round(max_risk, 2),
            "shared_contacts": shared_contacts,
            "suspicion_score": round(suspicion_score, 2),
            "evidence": "; ".join(evidence) if evidence else "No significant indicators",
            "is_known_fraud": is_known_fraud,
            "fraud_network_id": fraud_network_id if fraud_network_id else "",
            "recommended_action": _recommend_action(suspicion_score),
        })
    
    return pd.DataFrame(scores).sort_values("suspicion_score", ascending=False).reset_index(drop=True)


def _recommend_action(score):
    if score >= 70:
        return "Immediate Investigation"
    elif score >= 50:
        return "Priority Review"
    elif score >= 30:
        return "Monitor"
    else:
        return "Low Risk - No Action"


def find_shortest_paths(G, source_id, target_id):
    """Find shortest paths between two entities."""
    try:
        paths = list(nx.all_shortest_paths(G, source_id, target_id))
        return paths
    except (nx.NetworkXNoPath, nx.NodeNotFound):
        return []


def get_node_neighborhood(G, node_id, depth=2):
    """Get the neighborhood subgraph around a node."""
    if node_id not in G:
        return nx.Graph()
    
    nodes = set()
    current_level = {node_id}
    for _ in range(depth):
        next_level = set()
        for n in current_level:
            next_level.update(G.neighbors(n))
        nodes.update(current_level)
        current_level = next_level - nodes
    nodes.update(current_level)
    
    return G.subgraph(nodes).copy()


def compute_evaluation_metrics(suspicious_df, ground_truth_df):
    """Compute precision, recall, F1 for fraud detection."""
    if suspicious_df is None or len(suspicious_df) == 0:
        return {"precision": 0, "recall": 0, "f1": 0, "tp": 0, "fp": 0, "fn": 0, "tn": 0}
    
    # Predicted positives: communities with suspicion_score >= 50
    predicted_fraud = set(suspicious_df[suspicious_df["suspicion_score"] >= 50]["community_id"].tolist())
    
    # Actual fraud communities
    actual_fraud = set(suspicious_df[suspicious_df["is_known_fraud"] == True]["community_id"].tolist())
    all_communities = set(suspicious_df["community_id"].tolist())
    
    tp = len(predicted_fraud & actual_fraud)
    fp = len(predicted_fraud - actual_fraud)
    fn = len(actual_fraud - predicted_fraud)
    tn = len(all_communities - predicted_fraud - actual_fraud)
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1_score": round(f1, 4),
        "true_positives": tp,
        "false_positives": fp,
        "false_negatives": fn,
        "true_negatives": tn,
        "accuracy": round((tp + tn) / (tp + fp + fn + tn), 4) if (tp + fp + fn + tn) > 0 else 0,
    }