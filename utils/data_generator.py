import pandas as pd
import numpy as np
import json
import os
import hashlib
from datetime import datetime, timedelta
from collections import defaultdict

SEED = 20261022

# Volume parameters
NUM_CLAIMANTS = 2000
NUM_POLICIES = 3000
NUM_CLAIMS = 5000
NUM_PROVIDERS = 500
NUM_RELATIONSHIP_EDGES = 8000
NUM_INCIDENTS = 4500
NUM_INVOICES = 6000
NUM_PAYMENTS = 5500

# Fraud network parameters
NUM_FRAUD_NETWORKS = 25
FRAUD_CLAIM_RATIO = 0.12  # 12% of claims are fraudulent


def generate_all_data(seed=SEED, scale=1.0, data_dir="data"):
    """Master generator function. Returns dict of DataFrames and saves to CSV."""
    np.random.seed(seed)
    
    num_claimants = int(NUM_CLAIMANTS * scale)
    num_policies = int(NUM_POLICIES * scale)
    num_claims = int(NUM_CLAIMS * scale)
    num_providers = int(NUM_PROVIDERS * scale)
    num_incidents = int(NUM_INCIDENTS * scale)
    num_invoices = int(NUM_INVOICES * scale)
    num_payments = int(NUM_PAYMENTS * scale)
    num_edges = int(NUM_RELATIONSHIP_EDGES * scale)
    num_fraud_networks = int(NUM_FRAUD_NETWORKS * scale)

    # ---- Step 1: Generate Master/Parent Tables ----
    claimants = _generate_claimants(num_claimants)
    providers = _generate_providers(num_providers)
    policies = _generate_policies(num_policies, claimants)
    
    # ---- Step 2: Generate Dependent Tables ----
    incidents = _generate_incidents(num_incidents)
    claims = _generate_claims(num_claims, policies, incidents, claimants)
    invoices = _generate_invoices(num_invoices, claims, providers)
    payments = _generate_payments(num_payments, invoices, claims)
    
    # ---- Step 3: Inject Fraud Patterns ----
    fraud_result = _inject_fraud_patterns(
        claimants, policies, claims, incidents, providers, invoices, payments,
        num_fraud_networks
    )
    claimants, policies, claims, incidents, providers, invoices, payments, ground_truth, fraud_indicators = fraud_result
    
    # ---- Step 4: Generate Relationship Edges ----
    edges = _generate_relationship_edges(
        num_edges, claimants, policies, claims, providers, invoices, ground_truth
    )
    
    # ---- Step 5: Generate Investigations ----
    investigations = _generate_investigations(ground_truth, fraud_indicators)
    
    # ---- Step 6: Inject Edge Cases ----
    claims, payments, invoices, edges, edge_case_records = _inject_edge_cases(
        claims, payments, invoices, edges, claimants, policies, providers
    )
    
    # Add edge cases to ground truth
    for ec in edge_case_records:
        ground_truth.append(ec)
    
    # ---- Step 7: Validate ----
    validation_summary = _validate_data(
        claimants, policies, claims, incidents, providers, invoices, payments, edges
    )
    
    # ---- Package results ----
    datasets = {
        "claimants": claimants,
        "policies": policies,
        "claims": claims,
        "incidents": incidents,
        "providers": providers,
        "invoices": invoices,
        "payments": payments,
        "relationship_edges": edges,
        "fraud_indicators": pd.DataFrame(fraud_indicators),
        "investigations": investigations,
    }
    
    # Save to CSV
    os.makedirs(data_dir, exist_ok=True)
    for name, df in datasets.items():
        df.to_csv(os.path.join(data_dir, f"{name}.csv"), index=False)
    
    # Save ground truth separately
    gt_df = pd.DataFrame(ground_truth)
    gt_df.to_csv(os.path.join(data_dir, "ground_truth.csv"), index=False)
    gt_df.to_json(os.path.join(data_dir, "ground_truth.json"), orient="records", indent=2)
    
    # Save validation summary
    with open(os.path.join(data_dir, "validation_summary.json"), "w") as f:
        json.dump(validation_summary, f, indent=2, default=str)
    
    datasets["ground_truth"] = gt_df
    datasets["validation_summary"] = validation_summary
    
    return datasets


def _generate_claimants(n):
    """Generate claimant master records."""
    first_names = [
        "James", "Mary", "Robert", "Patricia", "John", "Jennifer", "Michael",
        "Linda", "David", "Elizabeth", "William", "Barbara", "Richard", "Susan",
        "Joseph", "Jessica", "Thomas", "Sarah", "Christopher", "Karen", "Charles",
        "Lisa", "Daniel", "Nancy", "Matthew", "Betty", "Anthony", "Margaret",
        "Mark", "Sandra", "Donald", "Ashley", "Steven", "Dorothy", "Paul", "Kimberly",
        "Andrew", "Emily", "Joshua", "Donna", "Kenneth", "Michelle", "Kevin", "Carol",
        "Brian", "Amanda", "George", "Melissa", "Timothy", "Deborah"
    ]
    last_names = [
        "Smith", "Johnson", "Williams", "Brown", "Jones", "Garcia", "Miller",
        "Davis", "Rodriguez", "Martinez", "Hernandez", "Lopez", "Gonzalez",
        "Wilson", "Anderson", "Thomas", "Taylor", "Moore", "Jackson", "Martin",
        "Lee", "Perez", "Thompson", "White", "Harris", "Sanchez", "Clark",
        "Ramirez", "Lewis", "Robinson", "Walker", "Young", "Allen", "King",
        "Wright", "Scott", "Torres", "Nguyen", "Hill", "Flores", "Green",
        "Adams", "Nelson", "Baker", "Hall", "Rivera", "Campbell", "Mitchell",
        "Carter", "Roberts"
    ]
    streets = [
        "Main St", "Oak Ave", "Pine Rd", "Maple Dr", "Cedar Ln", "Elm St",
        "Washington Blvd", "Park Ave", "Lake Dr", "Hill Rd", "River Rd",
        "Forest Ave", "Sunset Blvd", "Broadway", "Church St", "Mill Rd",
        "Spring St", "High St", "Union St", "Center St"
    ]
    cities = [
        "Springfield", "Riverside", "Fairview", "Georgetown", "Madison",
        "Clinton", "Arlington", "Burlington", "Manchester", "Milton",
        "Salem", "Franklin", "Greenville", "Bristol", "Oxford",
        "Newport", "Chester", "Hudson", "Clayton", "Ashland"
    ]
    states = ["CA", "TX", "FL", "NY", "PA", "IL", "OH", "GA", "NC", "MI",
              "NJ", "VA", "WA", "AZ", "MA", "TN", "IN", "MO", "MD", "WI"]
    
    records = []
    phone_set = set()
    email_set = set()
    
    for i in range(n):
        claimant_id = f"CLM-2026-{i+1:06d}"
        fname = np.random.choice(first_names)
        lname = np.random.choice(last_names)
        
        dob = datetime(1950, 1, 1) + timedelta(days=int(np.random.randint(0, 365 * 55)))
        
        street_num = np.random.randint(100, 9999)
        street = np.random.choice(streets)
        city = np.random.choice(cities)
        state = np.random.choice(states)
        zipcode = f"{np.random.randint(10000, 99999)}"
        address = f"{street_num} {street}, {city}, {state} {zipcode}"
        
        phone = f"({np.random.randint(200,999)}) {np.random.randint(200,999)}-{np.random.randint(1000,9999)}"
        while phone in phone_set:
            phone = f"({np.random.randint(200,999)}) {np.random.randint(200,999)}-{np.random.randint(1000,9999)}"
        phone_set.add(phone)
        
        email = f"{fname.lower()}.{lname.lower()}{np.random.randint(1,999)}@{'gmail.com' if np.random.random() < 0.5 else 'yahoo.com'}"
        while email in email_set:
            email = f"{fname.lower()}.{lname.lower()}{np.random.randint(1,9999)}@{'gmail.com' if np.random.random() < 0.5 else 'yahoo.com'}"
        email_set.add(email)
        
        bank_account = f"BANK-{np.random.randint(100000000, 999999999)}"
        created_date = datetime(2024, 1, 1) + timedelta(days=int(np.random.randint(0, 700)))
        status = np.random.choice(["active", "inactive", "suspended"], p=[0.85, 0.10, 0.05])
        risk_score = round(np.random.beta(2, 8) * 100, 2)
        
        records.append({
            "claimant_id": claimant_id,
            "first_name": fname,
            "last_name": lname,
            "full_name": f"{fname} {lname}",
            "date_of_birth": dob.strftime("%Y-%m-%d"),
            "address": address,
            "city": city,
            "state": state,
            "zip_code": zipcode,
            "phone": phone,
            "email": email,
            "bank_account": bank_account,
            "created_date": created_date.strftime("%Y-%m-%d"),
            "status": status,
            "risk_score": risk_score,
        })
    
    return pd.DataFrame(records)


def _generate_providers(n):
    """Generate provider master records."""
    provider_types = ["Medical", "Auto Repair", "Legal", "Rehabilitation", "Diagnostic", "Pharmacy"]
    specialties = {
        "Medical": ["General Practice", "Orthopedics", "Neurology", "Chiropractic", "Emergency"],
        "Auto Repair": ["Body Shop", "Mechanical", "Glass Repair", "Towing", "Paint"],
        "Legal": ["Personal Injury", "Insurance Law", "General Practice", "Litigation"],
        "Rehabilitation": ["Physical Therapy", "Occupational Therapy", "Speech Therapy"],
        "Diagnostic": ["MRI/CT", "X-Ray", "Laboratory", "Ultrasound"],
        "Pharmacy": ["Retail", "Compounding", "Specialty", "Mail Order"],
    }
    
    records = []
    for i in range(n):
        prov_id = f"PRV-{i+1:06d}"
        ptype = np.random.choice(provider_types)
        specialty = np.random.choice(specialties[ptype])
        
        name_prefixes = ["Premier", "Advanced", "Quality", "Elite", "Trusted", "Metro", "City", "National"]
        name_suffixes = {
            "Medical": ["Medical Center", "Clinic", "Health Services", "Medical Group"],
            "Auto Repair": ["Auto Body", "Collision Center", "Auto Repair", "Car Care"],
            "Legal": ["Law Firm", "Legal Services", "& Associates", "Law Group"],
            "Rehabilitation": ["Rehab Center", "Therapy Center", "Recovery Services"],
            "Diagnostic": ["Imaging Center", "Diagnostics Lab", "Testing Center"],
            "Pharmacy": ["Pharmacy", "Drug Store", "Rx Services"],
        }
        
        prov_name = f"{np.random.choice(name_prefixes)} {np.random.choice(name_suffixes[ptype])}"
        license_num = f"LIC-{np.random.randint(100000, 999999)}"
        license_status = np.random.choice(["active", "expired", "suspended", "revoked"], p=[0.85, 0.08, 0.05, 0.02])
        license_expiry = (datetime(2025, 1, 1) + timedelta(days=int(np.random.randint(0, 1095)))).strftime("%Y-%m-%d")
        
        state = np.random.choice(["CA", "TX", "FL", "NY", "PA", "IL", "OH", "GA", "NC", "MI"])
        avg_billing = round(np.random.lognormal(7, 1), 2)
        claim_volume = int(np.random.poisson(50))
        registered_date = (datetime(2020, 1, 1) + timedelta(days=int(np.random.randint(0, 1800)))).strftime("%Y-%m-%d")
        risk_score = round(np.random.beta(2, 8) * 100, 2)
        
        records.append({
            "provider_id": prov_id,
            "provider_name": prov_name,
            "provider_type": ptype,
            "specialty": specialty,
            "license_number": license_num,
            "license_status": license_status,
            "license_expiry": license_expiry,
            "state": state,
            "average_billing": avg_billing,
            "total_claim_volume": claim_volume,
            "registered_date": registered_date,
            "risk_score": risk_score,
            "status": "active" if license_status == "active" else "flagged",
        })
    
    return pd.DataFrame(records)


def _generate_policies(n, claimants):
    """Generate policy records linked to claimants."""
    policy_types = ["Auto", "Home", "Health", "Life", "Travel", "Commercial"]
    coverage_levels = ["Basic", "Standard", "Premium", "Comprehensive"]
    
    claimant_ids = claimants["claimant_id"].tolist()
    records = []
    
    for i in range(n):
        policy_id = f"POL-2026-{i+1:06d}"
        claimant_id = np.random.choice(claimant_ids)
        ptype = np.random.choice(policy_types, p=[0.35, 0.20, 0.25, 0.05, 0.05, 0.10])
        coverage = np.random.choice(coverage_levels)
        
        start_date = datetime(2023, 1, 1) + timedelta(days=int(np.random.randint(0, 900)))
        end_date = start_date + timedelta(days=365)
        
        premium_map = {"Basic": (200, 800), "Standard": (500, 1500), "Premium": (1000, 3000), "Comprehensive": (2000, 5000)}
        pmin, pmax = premium_map[coverage]
        premium = round(np.random.uniform(pmin, pmax), 2)
        
        coverage_limit_map = {"Basic": 25000, "Standard": 50000, "Premium": 100000, "Comprehensive": 250000}
        coverage_limit = coverage_limit_map[coverage]
        deductible = np.random.choice([250, 500, 1000, 2000, 5000])
        
        status_choices = ["active", "expired", "cancelled", "lapsed"]
        if end_date > datetime(2025, 10, 1):
            status = np.random.choice(status_choices, p=[0.80, 0.05, 0.10, 0.05])
        else:
            status = np.random.choice(status_choices, p=[0.10, 0.70, 0.10, 0.10])
        
        records.append({
            "policy_id": policy_id,
            "claimant_id": claimant_id,
            "policy_type": ptype,
            "coverage_level": coverage,
            "start_date": start_date.strftime("%Y-%m-%d"),
            "end_date": end_date.strftime("%Y-%m-%d"),
            "premium_amount": premium,
            "coverage_limit": coverage_limit,
            "deductible": deductible,
            "status": status,
        })
    
    return pd.DataFrame(records)


def _generate_incidents(n):
    """Generate incident records."""
    incident_types = ["Vehicle Collision", "Property Damage", "Medical Injury",
                      "Theft", "Natural Disaster", "Slip and Fall", "Fire",
                      "Water Damage", "Vandalism", "Work Injury"]
    
    severity_levels = ["Minor", "Moderate", "Severe", "Critical"]
    
    narratives = [
        "Rear-ended at traffic light intersection during rush hour.",
        "Side collision at four-way stop, other driver ran sign.",
        "Slipped on wet floor in grocery store, injured back.",
        "Vehicle broken into, electronics and personal items stolen.",
        "Tree fell on property during severe thunderstorm.",
        "Kitchen fire caused by electrical fault, significant damage.",
        "Water pipe burst causing flooding in basement.",
        "Hit by uninsured driver while parked at shopping center.",
        "Fell down stairs at commercial building, knee injury.",
        "Hail damage to vehicle and property during severe weather.",
        "Multi-vehicle pile-up on highway during fog.",
        "Burglar alarm triggered, window broken, items missing.",
        "Workplace machinery malfunction caused hand injury.",
        "Pedestrian struck by vehicle at crosswalk.",
        "Roof damaged during hurricane, interior water damage.",
    ]
    
    records = []
    for i in range(n):
        inc_id = f"INC-2026-{i+1:06d}"
        inc_type = np.random.choice(incident_types)
        severity = np.random.choice(severity_levels, p=[0.35, 0.35, 0.20, 0.10])
        
        inc_date = datetime(2024, 1, 1) + timedelta(days=int(np.random.randint(0, 650)))
        report_delay = int(np.random.exponential(5))
        report_date = inc_date + timedelta(days=report_delay)
        
        narrative = np.random.choice(narratives)
        location_types = ["Road/Highway", "Residence", "Commercial Property", "Parking Lot", "Workplace", "Public Space"]
        location = np.random.choice(location_types)
        
        police_report = np.random.choice([True, False], p=[0.6, 0.4])
        witnesses = int(np.random.poisson(1.5))
        
        records.append({
            "incident_id": inc_id,
            "incident_type": inc_type,
            "incident_date": inc_date.strftime("%Y-%m-%d"),
            "report_date": report_date.strftime("%Y-%m-%d"),
            "report_delay_days": report_delay,
            "severity": severity,
            "narrative": narrative,
            "location_type": location,
            "police_report_filed": police_report,
            "witness_count": witnesses,
            "status": np.random.choice(["reported", "under_investigation", "verified", "closed"],
                                       p=[0.20, 0.30, 0.35, 0.15]),
        })
    
    return pd.DataFrame(records)


def _generate_claims(n, policies, incidents, claimants):
    """Generate claim records linking policies and incidents."""
    policy_ids = policies["policy_id"].tolist()
    incident_ids = incidents["incident_id"].tolist()
    claimant_ids = claimants["claimant_id"].tolist()
    
    claim_types = ["Auto", "Property", "Medical", "Liability", "Workers Comp"]
    statuses = ["submitted", "under_review", "approved", "denied", "paid", "closed", "appealed"]
    
    records = []
    for i in range(n):
        claim_id = f"CLN-2026-{i+1:06d}"
        policy_id = np.random.choice(policy_ids)
        incident_id = np.random.choice(incident_ids[:len(incident_ids)])
        
        # Get claimant from policy
        pol_row = policies[policies["policy_id"] == policy_id]
        if len(pol_row) > 0:
            claimant_id = pol_row.iloc[0]["claimant_id"]
        else:
            claimant_id = np.random.choice(claimant_ids)
        
        claim_type = np.random.choice(claim_types)
        
        filed_date = datetime(2024, 3, 1) + timedelta(days=int(np.random.randint(0, 600)))
        
        claimed_amount = round(np.random.lognormal(8, 1.2), 2)
        claimed_amount = min(claimed_amount, 500000)
        
        approved_ratio = np.random.beta(5, 2)
        approved_amount = round(claimed_amount * approved_ratio, 2) if np.random.random() > 0.15 else 0
        
        status = np.random.choice(statuses, p=[0.10, 0.15, 0.25, 0.10, 0.25, 0.10, 0.05])
        if approved_amount == 0:
            status = "denied"
        
        records.append({
            "claim_id": claim_id,
            "policy_id": policy_id,
            "claimant_id": claimant_id,
            "incident_id": incident_id,
            "claim_type": claim_type,
            "filed_date": filed_date.strftime("%Y-%m-%d"),
            "claimed_amount": claimed_amount,
            "approved_amount": approved_amount,
            "status": status,
            "priority": np.random.choice(["Low", "Medium", "High", "Critical"], p=[0.30, 0.40, 0.20, 0.10]),
            "assigned_adjuster": f"ADJ-{np.random.randint(1, 50):03d}",
        })
    
    return pd.DataFrame(records)


def _generate_invoices(n, claims, providers):
    """Generate invoice records."""
    claim_ids = claims["claim_id"].tolist()
    provider_ids = providers["provider_id"].tolist()
    
    service_descriptions = [
        "Emergency room visit and treatment",
        "Vehicle damage assessment and repair",
        "Physical therapy - 10 sessions",
        "MRI scan and diagnostic report",
        "Legal consultation and representation",
        "Vehicle towing and storage",
        "Prescription medication",
        "Property damage assessment",
        "Surgery - orthopedic procedure",
        "Chiropractic treatment - 8 sessions",
        "Dental repair following trauma",
        "Home repair and restoration",
        "Ambulance transport",
        "Specialist consultation",
        "Rehabilitation equipment",
    ]
    
    records = []
    for i in range(n):
        provider_id = np.random.choice(provider_ids)
        prov_num = provider_id.split("-")[1]
        inv_id = f"INV-PRV-{prov_num}-{i+1:06d}"
        claim_id = np.random.choice(claim_ids)
        
        inv_date = datetime(2024, 4, 1) + timedelta(days=int(np.random.randint(0, 580)))
        due_date = inv_date + timedelta(days=int(np.random.choice([30, 45, 60, 90])))
        
        line_items = int(np.random.randint(1, 8))
        amount = round(np.random.lognormal(7, 1), 2)
        amount = min(amount, 200000)
        tax = round(amount * 0.08, 2)
        total = round(amount + tax, 2)
        
        status = np.random.choice(["submitted", "verified", "approved", "paid", "disputed", "rejected"],
                                  p=[0.15, 0.15, 0.20, 0.30, 0.10, 0.10])
        
        records.append({
            "invoice_id": inv_id,
            "claim_id": claim_id,
            "provider_id": provider_id,
            "invoice_date": inv_date.strftime("%Y-%m-%d"),
            "due_date": due_date.strftime("%Y-%m-%d"),
            "service_description": np.random.choice(service_descriptions),
            "line_item_count": line_items,
            "subtotal": amount,
            "tax": tax,
            "total_amount": total,
            "status": status,
        })
    
    return pd.DataFrame(records)


def _generate_payments(n, invoices, claims):
    """Generate payment event records."""
    invoice_ids = invoices["invoice_id"].tolist()
    claim_ids = claims["claim_id"].tolist()
    
    records = []
    for i in range(n):
        pay_id = f"PAY-2026-{i+1:06d}"
        invoice_id = np.random.choice(invoice_ids)
        
        inv_row = invoices[invoices["invoice_id"] == invoice_id]
        if len(inv_row) > 0:
            claim_id = inv_row.iloc[0]["claim_id"]
            expected_amount = float(inv_row.iloc[0]["total_amount"])
        else:
            claim_id = np.random.choice(claim_ids)
            expected_amount = round(np.random.lognormal(7, 1), 2)
        
        payment_amount = round(expected_amount * np.random.beta(8, 2), 2)
        
        pay_date = datetime(2024, 5, 1) + timedelta(days=int(np.random.randint(0, 570)))
        
        method = np.random.choice(["Bank Transfer", "Check", "Direct Deposit", "Wire Transfer"],
                                  p=[0.40, 0.20, 0.30, 0.10])
        
        status = np.random.choice(["pending", "processed", "completed", "failed", "reversed"],
                                  p=[0.15, 0.20, 0.45, 0.10, 0.10])
        
        records.append({
            "payment_id": pay_id,
            "invoice_id": invoice_id,
            "claim_id": claim_id,
            "payment_date": pay_date.strftime("%Y-%m-%d"),
            "payment_amount": payment_amount,
            "payment_method": method,
            "status": status,
            "reference_number": f"REF-{np.random.randint(100000000, 999999999)}",
        })
    
    return pd.DataFrame(records)


def _inject_fraud_patterns(claimants, policies, claims, incidents, providers, invoices, payments, num_networks):
    """Inject controlled fraud patterns and create ground truth."""
    ground_truth = []
    fraud_indicators = []
    
    claimant_ids = claimants["claimant_id"].tolist()
    provider_ids = providers["provider_id"].tolist()
    claim_ids = claims["claim_id"].tolist()
    
    # Pattern 1: Shared contact networks (same phone/email/bank across multiple claimants)
    for net_idx in range(min(8, num_networks)):
        network_id = f"FNET-{net_idx+1:04d}"
        network_size = np.random.randint(3, 7)
        selected_claimants = np.random.choice(claimant_ids, size=min(network_size, len(claimant_ids)), replace=False)
        
        shared_phone = f"(555) {np.random.randint(200,999)}-{np.random.randint(1000,9999)}"
        shared_bank = f"BANK-{np.random.randint(100000000, 999999999)}"
        
        for j, cid in enumerate(selected_claimants):
            idx = claimants[claimants["claimant_id"] == cid].index
            if len(idx) > 0:
                if j > 0:  # Keep first claimant's original, share with rest
                    claimants.loc[idx[0], "phone"] = shared_phone
                    claimants.loc[idx[0], "bank_account"] = shared_bank
                
                ground_truth.append({
                    "entity_type": "claimant",
                    "entity_id": cid,
                    "network_id": network_id,
                    "fraud_type": "shared_contact_network",
                    "is_fraud": True,
                    "confidence": 0.85,
                    "description": f"Claimant shares phone/bank with {network_size-1} others in network {network_id}",
                })
                
                fraud_indicators.append({
                    "indicator_id": f"FI-{len(fraud_indicators)+1:06d}",
                    "entity_type": "claimant",
                    "entity_id": cid,
                    "network_id": network_id,
                    "indicator_type": "shared_contact",
                    "severity": "High",
                    "description": f"Shared phone: {shared_phone}, Shared bank: {shared_bank}",
                    "detected_date": datetime(2025, 6, 1).strftime("%Y-%m-%d"),
                    "status": "active",
                })
    
    # Pattern 2: Provider billing anomalies (unusually high billing from certain providers)
    for net_idx in range(8, min(14, num_networks)):
        network_id = f"FNET-{net_idx+1:04d}"
        prov_id = np.random.choice(provider_ids)
        
        prov_idx = providers[providers["provider_id"] == prov_id].index
        if len(prov_idx) > 0:
            providers.loc[prov_idx[0], "average_billing"] = round(np.random.uniform(50000, 200000), 2)
            providers.loc[prov_idx[0], "risk_score"] = round(np.random.uniform(70, 95), 2)
        
        # Find invoices from this provider and inflate them
        prov_invoices = invoices[invoices["provider_id"] == prov_id].index
        for inv_idx in prov_invoices[:10]:
            invoices.loc[inv_idx, "total_amount"] = round(float(invoices.loc[inv_idx, "total_amount"]) * np.random.uniform(3, 8), 2)
            invoices.loc[inv_idx, "subtotal"] = round(float(invoices.loc[inv_idx, "subtotal"]) * np.random.uniform(3, 8), 2)
        
        ground_truth.append({
            "entity_type": "provider",
            "entity_id": prov_id,
            "network_id": network_id,
            "fraud_type": "billing_anomaly",
            "is_fraud": True,
            "confidence": 0.78,
            "description": f"Provider {prov_id} shows inflated billing patterns",
        })
        
        fraud_indicators.append({
            "indicator_id": f"FI-{len(fraud_indicators)+1:06d}",
            "entity_type": "provider",
            "entity_id": prov_id,
            "network_id": network_id,
            "indicator_type": "billing_anomaly",
            "severity": "Critical",
            "description": f"Provider billing 3-8x above normal rates",
            "detected_date": datetime(2025, 7, 1).strftime("%Y-%m-%d"),
            "status": "active",
        })
    
    # Pattern 3: Repeated incident narratives with unusual timing
    for net_idx in range(14, min(20, num_networks)):
        network_id = f"FNET-{net_idx+1:04d}"
        repeated_narrative = "Rear-ended at traffic light, severe whiplash, unable to work for 6 months."
        
        selected_claims_idx = np.random.choice(claims.index, size=min(5, len(claims)), replace=False)
        base_date = datetime(2025, 3, 1) + timedelta(days=int(np.random.randint(0, 60)))
        
        for j, cl_idx in enumerate(selected_claims_idx):
            # Set claims to same date range (suspicious timing)
            claims.loc[cl_idx, "filed_date"] = (base_date + timedelta(days=j)).strftime("%Y-%m-%d")
            claim_id = claims.loc[cl_idx, "claim_id"]
            
            # Update linked incident narrative
            inc_id = claims.loc[cl_idx, "incident_id"]
            inc_idx = incidents[incidents["incident_id"] == inc_id].index
            if len(inc_idx) > 0:
                incidents.loc[inc_idx[0], "narrative"] = repeated_narrative
                incidents.loc[inc_idx[0], "report_delay_days"] = 0
            
            ground_truth.append({
                "entity_type": "claim",
                "entity_id": claim_id,
                "network_id": network_id,
                "fraud_type": "repeated_narrative_timing",
                "is_fraud": True,
                "confidence": 0.72,
                "description": f"Claim uses identical narrative with suspicious timing cluster",
            })
            
            fraud_indicators.append({
                "indicator_id": f"FI-{len(fraud_indicators)+1:06d}",
                "entity_type": "claim",
                "entity_id": claim_id,
                "network_id": network_id,
                "indicator_type": "repeated_narrative",
                "severity": "High",
                "description": f"Identical incident narrative with {len(selected_claims_idx)} other claims filed within days",
                "detected_date": datetime(2025, 8, 1).strftime("%Y-%m-%d"),
                "status": "active",
            })
    
    # Pattern 4: Ring networks (claimant-provider collusion)
    for net_idx in range(20, num_networks):
        network_id = f"FNET-{net_idx+1:04d}"
        ring_claimants = np.random.choice(claimant_ids, size=min(4, len(claimant_ids)), replace=False)
        ring_provider = np.random.choice(provider_ids)
        
        # Share address among ring claimants
        shared_address = f"999 Suspicious Ave, Fraudtown, XX 99999"
        for cid in ring_claimants:
            idx = claimants[claimants["claimant_id"] == cid].index
            if len(idx) > 0:
                claimants.loc[idx[0], "address"] = shared_address
                claimants.loc[idx[0], "city"] = "Fraudtown"
                claimants.loc[idx[0], "risk_score"] = round(np.random.uniform(60, 90), 2)
            
            ground_truth.append({
                "entity_type": "claimant",
                "entity_id": cid,
                "network_id": network_id,
                "fraud_type": "ring_network",
                "is_fraud": True,
                "confidence": 0.90,
                "description": f"Part of claimant-provider collusion ring with {ring_provider}",
            })
        
        ground_truth.append({
            "entity_type": "provider",
            "entity_id": ring_provider,
            "network_id": network_id,
            "fraud_type": "ring_network",
            "is_fraud": True,
            "confidence": 0.88,
            "description": f"Provider involved in collusion ring with {len(ring_claimants)} claimants",
        })
        
        fraud_indicators.append({
            "indicator_id": f"FI-{len(fraud_indicators)+1:06d}",
            "entity_type": "network",
            "entity_id": network_id,
            "network_id": network_id,
            "indicator_type": "ring_network",
            "severity": "Critical",
            "description": f"Collusion ring: {len(ring_claimants)} claimants + provider {ring_provider}",
            "detected_date": datetime(2025, 9, 1).strftime("%Y-%m-%d"),
            "status": "under_investigation",
        })
    
    # Pattern 5: Benign shared households (NOT fraud - to test false positives)
    for benign_idx in range(15):
        household_members = np.random.choice(claimant_ids, size=min(3, len(claimant_ids)), replace=False)
        shared_addr = f"{np.random.randint(100,999)} Family Lane, Suburbia, CA 90210"
        
        for cid in household_members:
            idx = claimants[claimants["claimant_id"] == cid].index
            if len(idx) > 0:
                claimants.loc[idx[0], "address"] = shared_addr
            
            ground_truth.append({
                "entity_type": "claimant",
                "entity_id": cid,
                "network_id": f"BENIGN-{benign_idx+1:04d}",
                "fraud_type": "benign_shared_household",
                "is_fraud": False,
                "confidence": 0.95,
                "description": f"Legitimate family/household sharing address - NOT fraud",
            })
    
    return claimants, policies, claims, incidents, providers, invoices, payments, ground_truth, fraud_indicators


def _generate_relationship_edges(n, claimants, policies, claims, providers, invoices, ground_truth):
    """Generate relationship edges between entities."""
    edge_types = [
        ("claimant", "policy", "holds_policy"),
        ("claimant", "claim", "filed_claim"),
        ("claim", "incident", "relates_to"),
        ("claim", "provider", "serviced_by"),
        ("provider", "invoice", "issued_invoice"),
        ("claim", "invoice", "has_invoice"),
        ("claimant", "claimant", "shared_contact"),
        ("claimant", "provider", "referred_to"),
        ("provider", "provider", "affiliated_with"),
    ]
    
    records = []
    edge_id = 0
    
    # Generate structural edges from existing relationships
    # Claimant -> Policy
    for _, row in policies.iterrows():
        edge_id += 1
        records.append({
            "edge_id": f"EDGE-{edge_id:08d}",
            "source_type": "claimant",
            "source_id": row["claimant_id"],
            "target_type": "policy",
            "target_id": row["policy_id"],
            "relationship_type": "holds_policy",
            "weight": 1.0,
            "created_date": row["start_date"],
            "status": "active",
        })
        if edge_id >= n // 4:
            break
    
    # Claim -> Provider (via invoices)
    for _, row in invoices.iterrows():
        edge_id += 1
        records.append({
            "edge_id": f"EDGE-{edge_id:08d}",
            "source_type": "claim",
            "source_id": row["claim_id"],
            "target_type": "provider",
            "target_id": row["provider_id"],
            "relationship_type": "serviced_by",
            "weight": 1.0,
            "created_date": row["invoice_date"],
            "status": "active",
        })
        if edge_id >= n // 2:
            break
    
    # Claimant -> Claimant (shared attributes from fraud patterns)
    gt_claimant_networks = defaultdict(list)
    for gt in ground_truth:
        if gt["entity_type"] == "claimant" and gt.get("is_fraud"):
            gt_claimant_networks[gt["network_id"]].append(gt["entity_id"])
    
    for net_id, members in gt_claimant_networks.items():
        for i in range(len(members)):
            for j in range(i+1, len(members)):
                edge_id += 1
                records.append({
                    "edge_id": f"EDGE-{edge_id:08d}",
                    "source_type": "claimant",
                    "source_id": members[i],
                    "target_type": "claimant",
                    "target_id": members[j],
                    "relationship_type": "shared_contact",
                    "weight": 0.9,
                    "created_date": "2025-06-01",
                    "status": "active",
                })
    
    # Fill remaining with random edges
    claimant_ids = claimants["claimant_id"].tolist()
    provider_ids = providers["provider_id"].tolist()
    claim_ids = claims["claim_id"].tolist()
    
    while edge_id < n:
        edge_id += 1
        etype = np.random.choice(range(len(edge_types)))
        st, tt, rel = edge_types[etype]
        
        source_map = {"claimant": claimant_ids, "claim": claim_ids, "provider": provider_ids}
        target_map = {"claimant": claimant_ids, "claim": claim_ids, "provider": provider_ids,
                      "policy": policies["policy_id"].tolist(), "incident": [],
                      "invoice": invoices["invoice_id"].tolist()[:1000]}
        
        if st in source_map and tt in target_map and len(target_map[tt]) > 0:
            source = np.random.choice(source_map[st])
            target = np.random.choice(target_map[tt])
            
            records.append({
                "edge_id": f"EDGE-{edge_id:08d}",
                "source_type": st,
                "source_id": source,
                "target_type": tt,
                "target_id": target,
                "relationship_type": rel,
                "weight": round(np.random.uniform(0.1, 1.0), 3),
                "created_date": (datetime(2024, 1, 1) + timedelta(days=int(np.random.randint(0, 600)))).strftime("%Y-%m-%d"),
                "status": np.random.choice(["active", "inactive"], p=[0.9, 0.1]),
            })
    
    return pd.DataFrame(records[:n])


def _generate_investigations(ground_truth, fraud_indicators):
    """Generate investigation records from fraud patterns."""
    network_ids = set()
    for gt in ground_truth:
        if gt.get("is_fraud"):
            network_ids.add(gt["network_id"])
    
    records = []
    for i, net_id in enumerate(sorted(network_ids)):
        inv_id = f"INVG-2026-{i+1:06d}"
        
        related_entities = [gt for gt in ground_truth if gt["network_id"] == net_id and gt.get("is_fraud")]
        fraud_type = related_entities[0]["fraud_type"] if related_entities else "unknown"
        
        opened = datetime(2025, 6, 1) + timedelta(days=int(np.random.randint(0, 120)))
        status = np.random.choice(["open", "in_progress", "pending_review", "closed_confirmed", "closed_cleared"],
                                  p=[0.20, 0.30, 0.20, 0.20, 0.10])
        
        records.append({
            "investigation_id": inv_id,
            "network_id": net_id,
            "fraud_type": fraud_type,
            "entity_count": len(related_entities),
            "opened_date": opened.strftime("%Y-%m-%d"),
            "status": status,
            "assigned_investigator": f"INV-AGENT-{np.random.randint(1, 20):03d}",
            "priority": np.random.choice(["Low", "Medium", "High", "Critical"], p=[0.10, 0.30, 0.40, 0.20]),
            "estimated_loss": round(sum(np.random.uniform(5000, 100000) for _ in related_entities), 2),
            "notes": f"Investigation into {fraud_type} involving {len(related_entities)} entities",
        })
    
    return pd.DataFrame(records)


def _inject_edge_cases(claims, payments, invoices, edges, claimants, policies, providers):
    """Inject specific edge cases for testing."""
    edge_cases = []
    
    # Edge Case 1: Claim links to nonexistent policy
    fake_claim_idx = len(claims)
    new_claim = {
        "claim_id": f"CLN-2026-{len(claims)+1:06d}",
        "policy_id": "POL-2026-NONEXIST",
        "claimant_id": claimants.iloc[0]["claimant_id"],
        "incident_id": "INC-2026-000001",
        "claim_type": "Auto",
        "filed_date": "2025-08-15",
        "claimed_amount": 15000.00,
        "approved_amount": 0,
        "status": "denied",
        "priority": "High",
        "assigned_adjuster": "ADJ-001",
    }
    claims = pd.concat([claims, pd.DataFrame([new_claim])], ignore_index=True)
    edge_cases.append({
        "entity_type": "claim",
        "entity_id": new_claim["claim_id"],
        "network_id": "EDGE-CASE-001",
        "fraud_type": "orphan_claim",
        "is_fraud": False,
        "confidence": 1.0,
        "description": "Claim references nonexistent policy POL-2026-NONEXIST - data integrity issue",
    })
    
    # Edge Case 2: Payment exceeds approved amount
    if len(payments) > 0:
        overpay_idx = payments.index[0]
        original_amount = float(payments.loc[overpay_idx, "payment_amount"])
        payments.loc[overpay_idx, "payment_amount"] = round(original_amount * 5, 2)
        edge_cases.append({
            "entity_type": "payment",
            "entity_id": payments.loc[overpay_idx, "payment_id"],
            "network_id": "EDGE-CASE-002",
            "fraud_type": "overpayment",
            "is_fraud": True,
            "confidence": 0.95,
            "description": f"Payment amount {payments.loc[overpay_idx, 'payment_amount']} exceeds 5x original {original_amount}",
        })
    
    # Edge Case 3: Duplicate invoice across claimants
    if len(invoices) > 0:
        dup_inv = invoices.iloc[0].copy()
        dup_inv["invoice_id"] = f"INV-PRV-DUP-{1:06d}"
        dup_inv["claim_id"] = claims.iloc[min(100, len(claims)-1)]["claim_id"]
        invoices = pd.concat([invoices, pd.DataFrame([dup_inv])], ignore_index=True)
        edge_cases.append({
            "entity_type": "invoice",
            "entity_id": dup_inv["invoice_id"],
            "network_id": "EDGE-CASE-003",
            "fraud_type": "duplicate_invoice",
            "is_fraud": True,
            "confidence": 0.88,
            "description": f"Duplicate invoice submitted across different claims",
        })
    
    # Edge Case 4: Planted relationship not in edges
    edge_cases.append({
        "entity_type": "relationship",
        "entity_id": "HIDDEN-REL-001",
        "network_id": "EDGE-CASE-004",
        "fraud_type": "hidden_relationship",
        "is_fraud": True,
        "confidence": 0.70,
        "description": "Known relationship between CLM-2026-000001 and PRV-000001 not captured in edge table",
    })
    
    # Edge Case 5: Benign common address
    edge_cases.append({
        "entity_type": "claimant",
        "entity_id": "CLM-2026-000050",
        "network_id": "EDGE-CASE-005",
        "fraud_type": "benign_shared_address",
        "is_fraud": False,
        "confidence": 0.95,
        "description": "Common apartment complex address shared by multiple unrelated claimants - NOT fraud",
    })
    
    return claims, payments, invoices, edges, edge_cases


def _validate_data(claimants, policies, claims, incidents, providers, invoices, payments, edges):
    """Run validation checks on generated data."""
    results = {
        "timestamp": datetime.now().isoformat(),
        "seed": SEED,
        "checks": []
    }
    
    # PK uniqueness
    for name, df, pk in [
        ("claimants", claimants, "claimant_id"),
        ("policies", policies, "policy_id"),
        ("claims", claims, "claim_id"),
        ("incidents", incidents, "incident_id"),
        ("providers", providers, "provider_id"),
        ("invoices", invoices, "invoice_id"),
        ("payments", payments, "payment_id"),
    ]:
        is_unique = df[pk].is_unique
        results["checks"].append({
            "check": f"{name}_pk_unique",
            "passed": bool(is_unique),
            "detail": f"{pk} unique: {is_unique}, count: {len(df)}"
        })
    
    # FK integrity (with known exceptions)
    valid_policies = set(policies["policy_id"].tolist())
    orphan_claims = claims[~claims["policy_id"].isin(valid_policies)]
    results["checks"].append({
        "check": "claims_fk_policy",
        "passed": len(orphan_claims) <= 1,  # Allow 1 for edge case
        "detail": f"Orphan claims: {len(orphan_claims)} (1 expected edge case)"
    })
    
    # Row counts
    results["row_counts"] = {
        "claimants": len(claimants),
        "policies": len(policies),
        "claims": len(claims),
        "incidents": len(incidents),
        "providers": len(providers),
        "invoices": len(invoices),
        "payments": len(payments),
        "edges": len(edges),
    }
    
    # Checksums
    results["checksums"] = {}
    for name, df in [("claimants", claimants), ("policies", policies), ("claims", claims)]:
        checksum = hashlib.md5(pd.util.hash_pandas_object(df).values.tobytes()).hexdigest()
        results["checksums"][name] = checksum
    
    return results