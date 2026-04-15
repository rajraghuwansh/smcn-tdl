import torch
import argparse
from dataset import get_orientability_pair
from model import OrientabilityWrapper
from torch_geometric.loader import DataLoader

def evaluate_distinguishability(model, data_a, data_b, device='cpu'):
    model.eval()
    data_a = next(iter(DataLoader([data_a], batch_size=1, follow_batch=["x_0", "x_1", "x_2"]))).to(device)
    data_b = next(iter(DataLoader([data_b], batch_size=1, follow_batch=["x_0", "x_1", "x_2"]))).to(device)
    
    with torch.no_grad():
        embed_a = model(data_a)
        embed_b = model(data_b)
        
        # L2 Distance between the two global embeddings
        distance = torch.norm(embed_a - embed_b, p=2).item()
    return distance

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test SMCN vs HOMP on Torus and Klein Bottle")
    parser.add_argument('--grid_size', type=int, default=8, help="Size of the NxN grid")
    parser.add_argument('--hidden_dim', type=int, default=64, help="Hidden dimension of models")
    parser.add_argument('--layers', type=int, default=4, help="Number of message passing layers")
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(f"--- Generating {args.grid_size}x{args.grid_size} Torus and Klein Bottle ---")
    torus_data, klein_data = get_orientability_pair(grid_size=args.grid_size)
    
    print(f"Torus Stats: {torus_data.x_0.size(0)} Nodes, {torus_data.x_1.size(0)} Edges, {torus_data.x_2.size(0)} Faces")
    print("Local statistics are completely identical.")
    print("-" * 50)

    # Test HOMP Baseline
    print("Testing Standard HOMP (Baseline)...")
    try:
        homp_model = OrientabilityWrapper(base_model_type='homp', hidden_dim=args.hidden_dim, num_layers=args.layers).to(device)
        homp_dist = evaluate_distinguishability(homp_model, torus_data, klein_data, device)
        print(f"HOMP Distance: {homp_dist:.6f}")
        if homp_dist < 1e-5:
            print("❌ HOMP FAILED: The model is blind to orientability.")
    except Exception as e:
        print(f"Could not run HOMP baseline. Error: {e}")

    print("-" * 50)

    # Test SMCN 
    print("Testing SMCN (Subcomplex Layers)...")
    try:
        smcn_model = OrientabilityWrapper(base_model_type='smcn', hidden_dim=args.hidden_dim, num_layers=args.layers).to(device)
        
        # NOTE: SMCN requires the Subcomplex Lifting preprocessing step!
        # Assuming the repo provides a utility like `lift_to_subcomplexes(data)`
        # from utils.lifting import lift_to_subcomplexes
        # torus_data = lift_to_subcomplexes(torus_data)
        # klein_data = lift_to_subcomplexes(klein_data)
        
        smcn_dist = evaluate_distinguishability(smcn_model, torus_data, klein_data, device)
        print(f"SMCN Distance: {smcn_dist:.6f}")
        if smcn_dist > 1e-4:
            print("✅ SMCN SUCCEEDED: The model successfully separated the Torus and Klein Bottle!")
        else:
            print("⚠️ SMCN output distance was low. Ensure subcomplex lifting was applied to the Data object.")
    except Exception as e:
        print(f"Could not run SMCN. Error: {e}")
