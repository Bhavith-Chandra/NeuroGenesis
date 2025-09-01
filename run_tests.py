#!/usr/bin/env python3
"""
Test runner for NeuroGenesis project.

This script runs all unit tests and provides a summary of results.
"""

import unittest
import sys
import os
import time

# Add the project root to the Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def run_all_tests():
    """Run all tests and return results."""
    print("🧪 Running NeuroGenesis Tests")
    print("=" * 50)
    
    # Discover and run tests
    loader = unittest.TestLoader()
    start_dir = os.path.join(os.path.dirname(__file__), 'tests')
    suite = loader.discover(start_dir, pattern='test_*.py')
    
    # Run tests
    start_time = time.time()
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    end_time = time.time()
    
    # Print summary
    print("\n" + "=" * 50)
    print("📊 TEST SUMMARY")
    print("=" * 50)
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Time taken: {end_time - start_time:.2f} seconds")
    
    if result.failures:
        print("\n❌ FAILURES:")
        for test, traceback in result.failures:
            print(f"  - {test}: {traceback.split('AssertionError:')[-1].strip()}")
    
    if result.errors:
        print("\n🚨 ERRORS:")
        for test, traceback in result.errors:
            print(f"  - {test}: {traceback.split('Exception:')[-1].strip()}")
    
    if result.wasSuccessful():
        print("\n✅ All tests passed!")
        return True
    else:
        print(f"\n❌ {len(result.failures) + len(result.errors)} test(s) failed.")
        return False

def run_specific_test(test_name):
    """Run a specific test module."""
    print(f"🧪 Running specific test: {test_name}")
    print("=" * 50)
    
    # Import and run specific test
    if test_name == 'synapses':
        import tests.test_synapses
        suite = unittest.TestLoader().loadTestsFromModule(tests.test_synapses)
    elif test_name == 'grid_cells':
        import tests.test_grid_cells
        suite = unittest.TestLoader().loadTestsFromModule(tests.test_grid_cells)
    else:
        print(f"❌ Unknown test module: {test_name}")
        print("Available tests: synapses, grid_cells")
        return False
    
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result.wasSuccessful()

if __name__ == '__main__':
    if len(sys.argv) > 1:
        # Run specific test
        test_name = sys.argv[1]
        success = run_specific_test(test_name)
    else:
        # Run all tests
        success = run_all_tests()
    
    # Exit with appropriate code
    sys.exit(0 if success else 1) 