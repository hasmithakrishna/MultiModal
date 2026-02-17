#!/usr/bin/env python3
import requests
import numpy as np
import json
from loguru import logger
API_URL = "http://localhost:8000"
def test_health():
    logger.info("Testing /health endpoint...")
    response = requests.get(f"{API_URL}/health")
    logger.info(f"Status: {response.status_code}")
    logger.info(f"Response: {json.dumps(response.json(), indent=2)}")
    return response.status_code == 200
def test_prediction():
    logger.info("\nTesting /predict endpoint...")    
    sensor_window = np.random.randn(50, 52).tolist()
    request_data = {
        "asset_id": "pump_017",
        "timestamp": "2026-02-14T10:30:00Z",
        "sensor_window": sensor_window,
        "text_docs": [
            {
                "doc_type": "maintenance_request",
                "content": "Excessive vibration detected, bearing temperature elevated to 95°C"
            }
        ]
    }
    response = requests.post(
        f"{API_URL}/predict",
        json=request_data
    )
    logger.info(f"Status: {response.status_code}")
    if response.status_code == 200:
        result = response.json()
        logger.success("Prediction successful!")
        logger.info(f"Response:\n{json.dumps(result, indent=2)}")
        assert "text_analysis" in result
def test_batch_prediction():
    logger.info("\nTesting /predict/batch endpoint...")

    requests_list = []
    for i in range(3):
        sensor_window = np.random.randn(50, 52).tolist()
        requests_list.append({
            "asset_id": f"pump_{i:03d}",
            "timestamp": "2026-02-14T10:30:00Z",
            "sensor_window": sensor_window,
            "text_docs": [
                {
                    "doc_type": "maintenance_request",
                    "content": "Routine check"
                }
            ]
        })

    batch_request = {"requests": requests_list}
    
    response = requests.post(
        f"{API_URL}/predict/batch",
        json=batch_request
    )
    logger.info(f"Status: {response.status_code}")
    
    if response.status_code == 200:
        result = response.json()
        logger.success("Batch prediction successful!")
        logger.info(f"Processed: {result['total_predictions']} predictions")
        logger.info(f"Successful: {result['successful']}")
        logger.info(f"Failed: {result['failed']}")
        logger.info(f"Total time: {result['total_inference_ms']:.2f}ms")
        return result['successful'] > 0
    else:
        logger.error(f"Batch prediction failed: {response.text}")
        return False

def main():
    logger.info("="*50)
    logger.info("Oxmaint API Test Suite")
    logger.info("="*50)
    tests = [
        ("Health Check", test_health),
        ("Single Prediction", test_prediction),
        ("Batch Prediction", test_batch_prediction),
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            success = test_func()
            results.append((test_name, success))
        except Exception as e:
            logger.error(f"{test_name} failed with exception: {e}")
            results.append((test_name, False))    
    logger.info("\n" + "="*50)
    logger.info("Test Summary")
    logger.info("="*50)
    for test_name, success in results:
        status = "✓ PASSED" if success else "✗ FAILED"
        logger.info(f"{test_name}: {status}")
    passed = sum(1 for _, success in results if success)
    total = len(results)
    logger.info(f"\nTotal: {passed}/{total} tests passed")
    if passed == total:
        logger.success("\nAll tests passed!")
    else:
        logger.warning(f"\n{total - passed} test(s) failed")
if __name__ == "__main__":
    main()