"""Runs 6 test questions (5 answerable + 1 out-of-scope).  python run_tests.py"""
import sys
from pathlib import Path

import rag

TESTS = [
    ("How many days of annual leave do employees get?", ["24"]),
    ("When do I need a medical certificate for sick leave?", ["3", "consecutive"]),
    ("How many weeks of parental leave does a primary caregiver get?", ["16"]),
    ("How many days a week can I work remotely, and when does that start?", ["3", "probation"]),
    ("What is the daily meal limit when traveling, and when must I submit expenses?", ["1,500", "30"]),
    ("What is the stock option vesting schedule?", [rag.NO_ANSWER]),  # not in docs -> must refuse
]

index, chunks = rag.build_index(sorted(Path("docs").glob("*")))
passed = 0
for i, (q, expected) in enumerate(TESTS, 1):
    ans, hits = rag.answer(q, index, chunks)
    ok = all(e.lower() in ans.lower() for e in expected)
    passed += ok
    print(f"\nQ{i}: {q}\nA : {ans}\nTop source: {hits[0]['source']} ({hits[0]['score']:.2f})\n{'PASS' if ok else 'FAIL'}")
print(f"\n{passed}/{len(TESTS)} passed")
sys.exit(0 if passed == len(TESTS) else 1)
