#!/usr/bin/env python3
"""Expand seed pool from 4 base seeds to 16 diverse seeds by applying mutations."""
import os
import shutil
import sys
import random
from pathlib import Path

# Add src to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from lxml import etree
from src.mutations.registry import MUTATORS

def expand_pool(base_dir: str, out_dir: str, n_variants: int = 16):
    """Generate n_variants seeds from each base seed via mutation."""
    base_dir = Path(base_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Collect base seeds (top-level *.xml)
    base_seeds = sorted([f for f in base_dir.glob("*.xml") if f.is_file()])
    print(f"Found {len(base_seeds)} base seeds: {[f.stem for f in base_seeds]}")
    
    if not base_seeds:
        print("ERROR: No base seeds found!")
        return
    
    # For each base seed, generate variants
    variant_id = 0
    rng = random.Random(42)
    
    for base_seed in base_seeds:
        base_stem = base_seed.stem
        print(f"\nProcessing {base_stem}...")
        
        # Copy original
        out_path = out_dir / f"v{variant_id:02d}_{base_stem}.xml"
        shutil.copy(base_seed, out_path)
        variant_id += 1
        print(f"  v{variant_id-1:02d}: copy of original")
        
        # Generate variants by mutation
        for var_num in range(1, n_variants // len(base_seeds)):
            try:
                tree = etree.parse(str(base_seed))
                
                # Pick 1-3 random mutators and apply in sequence
                n_muts = rng.randint(1, 2)
                mutators = rng.sample(list(MUTATORS.values()), min(n_muts, len(MUTATORS)))
                
                for mutator in mutators:
                    try:
                        m_rng = random.Random(rng.randint(0, 2**31))
                        params = mutator.sample_params(tree, m_rng)
                        dummy_out = "/tmp/dummy.xml"
                        mres = mutator.apply(tree, params, dummy_out)
                        if mres.ok:
                            tree = etree.parse(str(mres.new_xml_path))
                            os.remove(dummy_out) if os.path.exists(dummy_out) else None
                    except Exception as e:
                        print(f"    Mutator {mutator.id} failed: {e}")
                        break
                
                # Write variant
                out_path = out_dir / f"v{variant_id:02d}_{base_stem}.xml"
                tree.write(str(out_path), xml_declaration=True, encoding="utf-8")
                variant_id += 1
                print(f"  v{variant_id-1:02d}: via {n_muts} mutations")
            except Exception as e:
                print(f"  Variant {var_num} failed: {e}")
    
    print(f"\nGenerated {variant_id} seeds in {out_dir}")
    return variant_id

if __name__ == "__main__":
    base_dir = project_root / "seeds"
    out_dir = project_root / "seeds"
    
    n = expand_pool(str(base_dir), str(out_dir), n_variants=16)
    print(f"\nExpanded pool: {n} seeds ready")
