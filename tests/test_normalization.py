#!/usr/bin/env python3
"""Debug script to test normalization behavior."""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.processing.deduplicator import Deduplicator

def test_normalization():
    dedup = Deduplicator()
    
    test_strings = [
        "Hello World! #test @user1",
        "hello world test user1",
        "Hello, World! Test @user1!",
        "Different content here"
    ]
    
    print("Testing normalization:")
    print("-" * 50)
    
    for i, text in enumerate(test_strings, 1):
        normalized = dedup._normalize_content(text)
        content_hash = dedup._hash_content(text)
        print(f"Post {i}: {text!r}")
        print(f"  → Normalized: {normalized!r}")
        print(f"  → Hash: {content_hash[:16]}...")
        print()

if __name__ == "__main__":
    test_normalization()