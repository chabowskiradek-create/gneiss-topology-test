#!/usr/bin/env python3
"""
Cryptographic blinding protocol for OSF pre-registration.
Deterministic SHA-256 hashing ensures reproducibility across platforms.
"""

import hashlib
import numpy as np

def get_deterministic_seed(seed_string: str = "GNEISS-TOPOLOGY-2026") -> int:
    """
    Generate deterministic integer seed from string using SHA-256.
    Fixes Python's non-deterministic hash() function.
    """
    hash_obj = hashlib.sha256(seed_string.encode('utf-8'))
    hex_dig = hash_obj.hexdigest()
    # Convert hex to int and limit to 32-bit for numpy compatibility
    seed_int = int(hex_dig, 16) % (2**32)
    return seed_int

def blind_variable(x: float, sigma_expected: float, seed: str = "GNEISS-TOPOLOGY-2026") -> float:
    """
    Apply cryptographic blinding to measurement.
    
    Args:
        x: True measured value
        sigma_expected: Expected uncertainty (for offset magnitude)
        seed: Cryptographic seed (deterministic)
    
    Returns:
        Blinded value (x + offset)
    """
    seed_int = get_deterministic_seed(seed)
    rng = np.random.RandomState(seed_int)  # Deterministic RNG
    offset = rng.normal(0, 2 * sigma_expected)
    return x + offset

def unblind_variable(blinded_x: float, sigma_expected: float, seed: str = "GNEISS-TOPOLOGY-2026") -> float:
    """
    Remove blinding (only after unblinding event).
    """
    seed_int = get_deterministic_seed(seed)
    rng = np.random.RandomState(seed_int)
    offset = rng.normal(0, 2 * sigma_expected)
    return blinded_x - offset

if __name__ == "__main__":
    # Test determinism
    val = blind_variable(0.020, 0.005)
    val2 = blind_variable(0.020, 0.005)
    assert val == val2, "Blinding must be deterministic!"
    print(f"✓ Blinding deterministic: {val:.6f}")
