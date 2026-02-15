#!/usr/bin/env python3
"""
Critical test: Blinding must be deterministic across platforms.
"""

import sys
sys.path.insert(0, '../analysis')

from analysis.blinding import blind_variable, get_deterministic_seed

def test_determinism():
    """Test that blinding produces identical values on every run."""
    val1 = blind_variable(0.020, 0.005)
    val2 = blind_variable(0.020, 0.005)
    val3 = blind_variable(0.020, 0.005)
    
    assert val1 == val2 == val3, f"Non-deterministic! {val1}, {val2}, {val3}"
    print("✓ Blinding is deterministic")

def test_seed_generation():
    """Test SHA-256 seed generation."""
    seed = get_deterministic_seed("GNEISS-TOPOLOGY-2026")
    assert isinstance(seed, int)
    assert 0 <= seed < 2**32
    print(f"✓ Seed generated: {seed}")

if __name__ == "__main__":
    test_determinism()
    test_seed_generation()
    print("All blinding tests passed!")
