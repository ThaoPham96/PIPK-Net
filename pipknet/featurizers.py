import torch
from rdkit import Chem
from torch_geometric.data import Data

# Mapping IonType strings to integers for the Embedding layer
ION_TYPES = ['anionic', 'cationic', 'neutral', 'zwitterionic']
ION_MAP = {k: i for i, k in enumerate(ION_TYPES)}

def one_hot_encoding(x, allowable_set):
    """Encodes a value into a one-hot list."""
    return [int(x == s) for s in allowable_set]

def atom_to_feature_vector(atom):
    """Encodes atom properties into a comprehensive one-hot feature vector."""
    # Your specific drug-relevant atom set
    atom_types = ['C', 'O', 'N', 'F', 'Cl', 'S', 'Br', 'P', 'Na', 'I', 'Ca', 'H', 'K', 'B']
    degrees = [0, 1, 2, 3, 4]
    formal_charges = [-1, 0, 1]
    
    feats = []
    feats += one_hot_encoding(atom.GetSymbol(), atom_types)
    feats += one_hot_encoding(atom.GetDegree(), degrees)
    feats += one_hot_encoding(atom.GetFormalCharge(), formal_charges)
    feats.append(int(atom.GetIsAromatic()))
    
    return torch.tensor(feats, dtype=torch.float)

def build_edge_index_and_attr(mol):
    """Creates edge indices and features from molecular bonds."""
    edge_index = [[], []]
    edge_attr = []
    bond_types = [
        Chem.rdchem.BondType.SINGLE,
        Chem.rdchem.BondType.DOUBLE,
        Chem.rdchem.BondType.TRIPLE,
        Chem.rdchem.BondType.AROMATIC
    ]
    
    for b in mol.GetBonds():
        i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        
        # Bond features: One-hot Type + Conjugation + Ring status
        enc = one_hot_encoding(b.GetBondType(), bond_types) + \
              [int(b.GetIsConjugated()), int(b.IsInRing())]
        
        bf = torch.tensor(enc, dtype=torch.float)
        
        # Undirected graph (add both directions)
        edge_index[0] += [i, j]
        edge_index[1] += [j, i]
        edge_attr += [bf, bf]
        
    return torch.tensor(edge_index, dtype=torch.long), torch.stack(edge_attr, dim=0)

def smiles_to_pyg(smiles, ion_type):
    """
    Translates a SMILES string and an ion type into a PyTorch Geometric Data object.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None # Or raise ValueError for stricter debugging
        
    # Build Node Features
    x = torch.stack([atom_to_feature_vector(a) for a in mol.GetAtoms()])
    
    # Build Edge Features
    ei, ea = build_edge_index_and_attr(mol)
    
    # Map Ionization Type
    ion_idx = ION_MAP.get(str(ion_type).strip().lower(), ION_MAP['neutral'])
    ion_feat = torch.tensor([ion_idx], dtype=torch.long)
    
    return Data(x=x, edge_index=ei, edge_attr=ea, ion_feat=ion_feat)