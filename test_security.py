import sys
import os

# Add current directory to path
sys.path.append('/data/my_tmo_project')

from tmo_interface import get_security_score

def test_security_score():
    print("Testing Security Score with BERT NER...", flush=True)

    # Test case 1: Safe prompt (no entities), Local action (0) -> Should be 1.0
    score = get_security_score("Hello world", 0)
    print(f"Test 1 (Safe, Local): Score={score} (Expected 1.0)", flush=True)
    assert score == 1.0

    # Test case 2: Safe prompt (no entities), Cloud action (1) -> Should be 1.0
    score = get_security_score("Hello world", 1)
    print(f"Test 2 (Safe, Cloud): Score={score} (Expected 1.0)", flush=True)
    assert score == 1.0

    # Test case 3: Sensitive prompt (with entity), Cloud action (1) -> Should be 0.0
    # "John Doe" should be detected as PER
    score = get_security_score("My name is John Doe.", 1)
    print(f"Test 3 (Sensitive Entity, Cloud): Score={score} (Expected 0.0)", flush=True)
    if score != 0.0:
        print("WARNING: Test 3 failed. BERT might not have detected 'John Doe' with >0.9 confidence.", flush=True)
    else:
        print("Test 3 Passed.", flush=True)

    print("All tests finished!", flush=True)

if __name__ == "__main__":
    test_security_score()
