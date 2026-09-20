"""
eval/test_agent.py
==================
Live verification CLI script for the Intelli Docs Agentic RAG backend.

Tests all 6 live scenarios against a running FastAPI backend instance:
1. Greeting Fast Path
2. Multi-Turn Memory (Context Rewriting)
3. Calculator Tool
4. Notes Metadata Tool (List Docs)
5. Document Search & Grounding
6. Cache Hit

Usage:
    python eval/test_agent.py [--base-url http://localhost:8000]
"""

import argparse
import sys
import uuid
import requests


def run_scenario(name, url, payload, validator):
    """Executes a POST request to /chat and runs validation logic."""
    print(f"-> Running {name}...")
    try:
        response = requests.post(url, json=payload, timeout=30)
        if response.status_code != 200:
            print(f"   [FAIL] HTTP status code {response.status_code}")
            print(f"   Response body: {response.text}")
            return False

        data = response.json()
        passed, reason = validator(data)
        if passed:
            print(f"   [PASS] {reason}")
            return True
        else:
            print(f"   [FAIL] {reason}")
            print(f"   Received Payload: {data}")
            return False

    except requests.exceptions.RequestException as e:
        print(f"   [FAIL] Network/Request error: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Live Agent Verification CLI Script")
    parser.add_argument(
        "--base-url",
        default="http://localhost:8000",
        help="Base URL of the live FastAPI backend (default: http://localhost:8000)"
    )
    args = parser.parse_args()

    base_url = args.base_url.rstrip("/")
    chat_url = f"{base_url}/chat"
    health_url = f"{base_url}/health"

    print("==================================================")
    print("      INTELLI DOCS RAG - LIVE AGENT TEST SUITE     ")
    print(f" Target API: {chat_url}")
    print("==================================================\n")

    # Check backend connectivity first
    try:
        health_resp = requests.get(health_url, timeout=5)
        if health_resp.status_code == 200:
            print(f"Backend Health Check: OK ({health_resp.json()})\n")
        else:
            print(f"Backend Health Check returned status {health_resp.status_code}.\n")
    except Exception as e:
        print(f"Warning: Could not connect to backend health check at {health_url}: {e}\n")

    results = []

    # ----------------------------------------------------
    # Scenario 1: Greeting Fast Path
    # ----------------------------------------------------
    def validate_scenario_1(data):
        if data.get("from_cache") is False and "Hello" in data.get("answer", ""):
            return True, "Greeting interceptor triggered instantly (from_cache == false)."
        return False, f"Unexpected response: from_cache={data.get('from_cache')}, answer={data.get('answer')}"

    s1_session = f"eval-s1-{uuid.uuid4().hex[:6]}"
    results.append(run_scenario(
        "Scenario 1: Greeting Fast Path",
        chat_url,
        {"question": "Hello!", "session_id": s1_session},
        validate_scenario_1
    ))

    # ----------------------------------------------------
    # Scenario 2: Multi-Turn Memory
    # ----------------------------------------------------
    s2_session = f"eval-s2-{uuid.uuid4().hex[:6]}"
    # Turn 1
    requests.post(chat_url, json={"question": "What is quicksort?", "session_id": s2_session}, timeout=30)
    
    def validate_scenario_2(data):
        meta = data.get("agent_metadata", {})
        standalone = meta.get("standalone_query", "")
        if standalone and "quicksort" in standalone.lower():
            return True, f"Multi-turn query correctly contextualized to: '{standalone}'"
        return False, f"Standalone query did not resolve pronoun context properly: '{standalone}'"

    results.append(run_scenario(
        "Scenario 2: Multi-Turn Memory",
        chat_url,
        {"question": "What is its worst case complexity?", "session_id": s2_session},
        validate_scenario_2
    ))

    # ----------------------------------------------------
    # Scenario 3: Calculator Tool
    # ----------------------------------------------------
    s3_session = f"eval-s3-{uuid.uuid4().hex[:6]}"
    def validate_scenario_3(data):
        meta = data.get("agent_metadata", {})
        tool = meta.get("tool_choice")
        ans = data.get("answer", "")
        # 85 * 0.4 + 92 * 0.6 = 34 + 55.2 = 89.2
        if tool == "calculate" and ("89.2" in ans or "89.20" in ans):
            return True, f"Calculator tool executed successfully: tool_choice='calculate', answer contains '89.2'"
        return False, f"Calculator validation failed: tool_choice='{tool}', answer='{ans}'"

    results.append(run_scenario(
        "Scenario 3: Calculator Tool",
        chat_url,
        {"question": "Calculate 85 * 0.4 + 92 * 0.6", "session_id": s3_session},
        validate_scenario_3
    ))

    # ----------------------------------------------------
    # Scenario 4: Notes Metadata Tool
    # ----------------------------------------------------
    s4_session = f"eval-s4-{uuid.uuid4().hex[:6]}"
    def validate_scenario_4(data):
        meta = data.get("agent_metadata", {})
        tool = meta.get("tool_choice")
        if tool == "list_docs":
            return True, "Notes metadata tool executed successfully: tool_choice='list_docs'"
        return False, f"Expected tool_choice='list_docs', got '{tool}'"

    results.append(run_scenario(
        "Scenario 4: Notes Metadata Tool",
        chat_url,
        {"question": "What documents do you have available?", "session_id": s4_session},
        validate_scenario_4
    ))

    # ----------------------------------------------------
    # Scenario 5: Document Search & Grounding
    # ----------------------------------------------------
    s5_session = f"eval-s5-{uuid.uuid4().hex[:6]}"
    s5_prompt = "What is the grading policy or syllabus outline?"
    def validate_scenario_5(data):
        meta = data.get("agent_metadata", {})
        guard_passed = meta.get("guard_passed", False)
        sources = data.get("sources", [])
        if guard_passed:
            return True, f"Document search & grounding succeeded (guard_passed=true, sources_count={len(sources)})"
        return False, f"Grounding check failed: guard_passed={guard_passed}"

    results.append(run_scenario(
        "Scenario 5: Document Search & Grounding",
        chat_url,
        {"question": s5_prompt, "session_id": s5_session},
        validate_scenario_5
    ))

    # ----------------------------------------------------
    # Scenario 6: Cache Hit
    # ----------------------------------------------------
    def validate_scenario_6(data):
        if data.get("from_cache") is True:
            return True, "Redis answer cache hit verified (from_cache == true)."
        return False, f"Expected from_cache == true, got {data.get('from_cache')}"

    results.append(run_scenario(
        "Scenario 6: Cache Hit",
        chat_url,
        {"question": s5_prompt, "session_id": s5_session},
        validate_scenario_6
    ))

    # ----------------------------------------------------
    # Summary Report
    # ----------------------------------------------------
    passed_count = sum(1 for r in results if r)
    total_count = len(results)

    print("\n==================================================")
    print(f" VERIFICATION RESULTS: {passed_count}/{total_count} SCENARIOS PASSED")
    print("==================================================")

    if passed_count == total_count:
        print("ALL LIVE AGENT VERIFICATION TESTS PASSED SUCCESSFULLY!")
        sys.exit(0)
    else:
        print("SOME SCENARIOS FAILED. PLEASE CHECK BACKEND/MODEL STATUS.")
        sys.exit(1)


if __name__ == "__main__":
    main()
