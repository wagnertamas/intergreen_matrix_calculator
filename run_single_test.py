print("STARTING TEST RUNNER", flush=True)
from test_automated import SumoRLTester

def main():
    print("In main", flush=True)
    tester = SumoRLTester()
    print("Running Single Test: _test_16_single_agent_traffic", flush=True)
    tester._test_16_single_agent_traffic()
    tester._print_summary()

if __name__ == "__main__":
    main()
