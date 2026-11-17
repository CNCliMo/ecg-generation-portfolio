"""Headless verification of the Streamlit app using streamlit.testing.AppTest.

Checks that all three functional areas render without exceptions and that the
main buttons of "controlled generation" and "multi-route comparison" work with
whatever routes are currently trained (the app is expected to degrade gracefully
when checkpoints are missing).

Usage: powershell -File tools\run.ps1 tools\test_app.py
"""
from __future__ import annotations

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
sys.path.insert(0, ROOT)

# The Windows sandbox cannot clean up temp dirs created with mode 0700
# (their DACL does not inherit the sandbox grant). Streamlit's AppTest creates
# such dirs, so the project's mkdtemp patch must be applied first.

from streamlit.testing.v1 import AppTest


def check(at: AppTest, label: str) -> bool:
    if at.exception:
        print(f"FAIL [{label}] {len(at.exception)} exception(s):")
        for e in at.exception:
            print("   ", e.value)
        return False
    print(f"OK   [{label}]")
    return True


def main() -> int:
    ok = True
    at = AppTest.from_file(os.path.join(ROOT, "app", "streamlit_app.py"), default_timeout=600)
    at.run()
    ok &= check(at, "initial render (three tabs)")

    # tab 1: sample from test set + diagnose
    try:
        at.button(key="t1_sample_btn").click().run()
        ok &= check(at, "tab1 sample-and-diagnose")
    except Exception as exc:  # noqa: BLE001
        print(f"FAIL tab1: {exc}")
        ok = False

    # tab 2: controlled generation with a currently available route
    try:
        routes = at.selectbox(key="t2_route").options
        print("available routes in app:", routes)
        if routes:
            at.button(key="t2_gen_btn").click().run()
            ok &= check(at, "tab2 controlled generation")
    except Exception as exc:  # noqa: BLE001
        print(f"FAIL tab2: {exc}")
        ok = False

    # tab 3: multi-route comparison (available routes only) + real reference
    try:
        at.button(key="t3_btn").click().run()
        ok &= check(at, "tab3 multi-route comparison")
    except Exception as exc:  # noqa: BLE001
        print(f"FAIL tab3: {exc}")
        ok = False

    print("RESULT:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
