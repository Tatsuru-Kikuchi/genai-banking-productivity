"""
Align Both Weight Matrices with Panel
=====================================

Ensures both Network W and Geographic W are properly aligned with the panel.

Network W:  Cosine similarity of interbank activity profiles
Geographic W: Inverse distance (Haversine formula)

Usage:
    python code/align_weight_matrices.py
"""

import pandas as pd
import numpy as np
import os


def align_w_to_panel(W_full, w_bank_list, panel_banks):
    """
    Align weight matrix to panel bank ordering.
    
    Parameters:
    -----------
    W_full : np.ndarray
        Full weight matrix
    w_bank_list : list
        Bank IDs in W matrix order
    panel_banks : list
        Bank IDs in panel order (target)
    
    Returns:
    --------
    W_aligned : np.ndarray
        Aligned weight matrix
    common_banks : list
        Banks in aligned order
    """
    
    # Convert to strings for consistent matching
    w_bank_str = [str(b) for b in w_bank_list]
    panel_bank_str = [str(b) for b in panel_banks]
    
    # Find common banks
    common_set = set(w_bank_str) & set(panel_bank_str)
    
    # Order by panel order
    common_banks = [b for b in panel_bank_str if b in common_set]
    
    print(f"  W banks: {len(w_bank_str)}")
    print(f"  Panel banks: {len(panel_bank_str)}")
    print(f"  Common banks: {len(common_banks)}")
    
    if len(common_banks) == 0:
        print("  ERROR: No common banks!")
        return None, []
    
    # Get indices in original W
    w_bank_to_idx = {b: i for i, b in enumerate(w_bank_str)}
    indices = [w_bank_to_idx[b] for b in common_banks]
    
    # Extract submatrix
    W_aligned = W_full[np.ix_(indices, indices)]
    
    # Re-row-standardize
    row_sums = W_aligned.sum(axis=1)
    W_aligned = np.divide(
        W_aligned,
        row_sums[:, np.newaxis],
        where=row_sums[:, np.newaxis] != 0
    )
    
    print(f"  Aligned W shape: {W_aligned.shape}")
    
    return W_aligned, common_banks


def main():
    """Align both weight matrices."""
    
    print("=" * 70)
    print("ALIGNING WEIGHT MATRICES WITH PANEL")
    print("=" * 70)
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    
    # Paths
    panel_path = os.path.join(project_root, "data", "processed", "dsdm_panel_aligned.csv")
    
    w_network_path = os.path.join(project_root, "data", "processed", "W_network_fed.npy")
    w_geo_path = os.path.join(project_root, "data", "processed", "W_geographic_fed.npy")
    
    w_network_csv = os.path.join(project_root, "data", "processed", "W_network_fed.csv")
    w_geo_csv = os.path.join(project_root, "data", "processed", "W_geographic_fed.csv")
    
    bank_order_path = os.path.join(project_root, "data", "processed", "w_matrix_bank_ordering.csv")
    
    # Output paths
    w_network_aligned_path = os.path.join(project_root, "data", "processed", "W_network_aligned.npy")
    w_geo_aligned_path = os.path.join(project_root, "data", "processed", "W_geographic_aligned.npy")
    aligned_order_path = os.path.join(project_root, "data", "processed", "w_aligned_bank_ordering.csv")
    
    # Load panel
    panel = pd.read_csv(panel_path, dtype={'rssd_id': str})
    panel_banks = panel['rssd_id'].dropna().unique().tolist()
    
    print(f"\nPanel: {len(panel)} observations, {len(panel_banks)} unique banks")
    
    # Get bank ordering from W matrix
    if os.path.exists(bank_order_path):
        bank_order_df = pd.read_csv(bank_order_path, dtype={'rssd_id': str})
        w_bank_list = bank_order_df['rssd_id'].tolist()
    elif os.path.exists(w_network_csv):
        w_df = pd.read_csv(w_network_csv, index_col=0)
        w_bank_list = [str(b) for b in w_df.index.tolist()]
    else:
        print("ERROR: Cannot determine W bank ordering")
        return
    
    print(f"W bank ordering: {len(w_bank_list)} banks")
    
    # =========================================================================
    # NETWORK W
    # =========================================================================
    
    print("\n" + "-" * 70)
    print("NETWORK WEIGHT MATRIX")
    print("-" * 70)
    
    if os.path.exists(w_network_path):
        W_network = np.load(w_network_path)
        print(f"Loaded: {w_network_path}")
        print(f"Shape: {W_network.shape}")
        
        W_net_aligned, common_banks = align_w_to_panel(W_network, w_bank_list, panel_banks)
        
        if W_net_aligned is not None:
            np.save(w_network_aligned_path, W_net_aligned)
            print(f"Saved: {w_network_aligned_path}")
    else:
        print(f"Not found: {w_network_path}")
        W_net_aligned = None
        common_banks = []
    
    # =========================================================================
    # GEOGRAPHIC W
    # =========================================================================
    
    print("\n" + "-" * 70)
    print("GEOGRAPHIC WEIGHT MATRIX")
    print("-" * 70)
    
    if os.path.exists(w_geo_path):
        W_geo = np.load(w_geo_path)
        print(f"Loaded: {w_geo_path}")
        print(f"Shape: {W_geo.shape}")
        
        # Geographic W may have different bank ordering (only banks with coordinates)
        if os.path.exists(w_geo_csv):
            w_geo_df = pd.read_csv(w_geo_csv, index_col=0)
            w_geo_bank_list = [str(b) for b in w_geo_df.index.tolist()]
        else:
            w_geo_bank_list = w_bank_list  # Assume same ordering
        
        W_geo_aligned, geo_common_banks = align_w_to_panel(W_geo, w_geo_bank_list, panel_banks)
        
        if W_geo_aligned is not None:
            # Check if same banks as Network W
            if len(geo_common_banks) != len(common_banks):
                print(f"  WARNING: Different bank coverage than Network W")
                print(f"    Network: {len(common_banks)} banks")
                print(f"    Geographic: {len(geo_common_banks)} banks")
                
                # Find intersection
                both_banks = [b for b in common_banks if b in geo_common_banks]
                print(f"    Common to both: {len(both_banks)} banks")
                
                # Re-align both to common set if needed
                if len(both_banks) < len(common_banks):
                    print("  Re-aligning both matrices to common bank set...")
                    
                    W_net_aligned, common_banks = align_w_to_panel(W_network, w_bank_list, both_banks)
                    W_geo_aligned, _ = align_w_to_panel(W_geo, w_geo_bank_list, both_banks)
                    
                    np.save(w_network_aligned_path, W_net_aligned)
                    print(f"  Re-saved: {w_network_aligned_path}")
            
            np.save(w_geo_aligned_path, W_geo_aligned)
            print(f"Saved: {w_geo_aligned_path}")
    else:
        print(f"Not found: {w_geo_path}")
        print("Geographic W will not be available for comparison")
    
    # =========================================================================
    # SAVE ALIGNED BANK ORDERING
    # =========================================================================
    
    if common_banks:
        aligned_df = pd.DataFrame({
            'w_index': range(len(common_banks)),
            'rssd_id': common_banks
        })
        aligned_df.to_csv(aligned_order_path, index=False)
        print(f"\nSaved bank ordering: {aligned_order_path}")
    
    # =========================================================================
    # SUMMARY
    # =========================================================================
    
    print("\n" + "=" * 70)
    print("ALIGNMENT SUMMARY")
    print("=" * 70)
    
    print(f"\nAligned banks: {len(common_banks)}")
    print(f"Network W:     {w_network_aligned_path}")
    print(f"Geographic W:  {w_geo_aligned_path}")
    print(f"Bank ordering: {aligned_order_path}")
    
    # Verification
    print("\nVerification:")
    
    if os.path.exists(w_network_aligned_path):
        W_net = np.load(w_network_aligned_path)
        print(f"  Network W: {W_net.shape}, row sums ≈ {W_net.sum(axis=1).mean():.4f}")
    
    if os.path.exists(w_geo_aligned_path):
        W_geo = np.load(w_geo_aligned_path)
        print(f"  Geographic W: {W_geo.shape}, row sums ≈ {W_geo.sum(axis=1).mean():.4f}")
    
    print("\n" + "=" * 70)
    print("READY FOR DSDM ESTIMATION")
    print("=" * 70)


if __name__ == "__main__":
    main()
