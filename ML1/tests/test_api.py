#!/usr/bin/env python3
"""
test_api.py
Примеры использования API для тестирования предсказаний
"""

import requests
import json
from datetime import datetime

BASE_URL = "http://localhost:8000"

def test_health():
    """Test health endpoint"""
    print("Testing health endpoint...")
    response = requests.get(f"{BASE_URL}/health")
    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}\n")

def test_transaction_prediction():
    """Test transaction prediction"""
    print("Testing transaction prediction...")
    
    payload = {
        "transaction": {
            "amount": 5000.0,
            "timestamp": "2025-11-28T14:30:00",
            "src_account_id": "123456",
            "beneficiary_id": "789012"
        }
    }
    
    response = requests.post(
        f"{BASE_URL}/predict/transaction",
        json=payload,
        headers={"Content-Type": "application/json"}
    )
    
    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}\n")

def test_client_activity_prediction():
    """Test client activity prediction"""
    print("Testing client activity prediction...")
    
    payload = {
        "activity": {
            "timestamp": "2025-11-28T14:30:00",
            "src_account_id": "123456",
            "logins_last_7_days": 5,
            "logins_last_30_days": 20,
            "login_frequency_7d": 0.71,
            "login_frequency_30d": 0.67,
            "avg_login_interval_30d": 100000.0,
            "std_login_interval_30d": 50000.0
        }
    }
    
    response = requests.post(
        f"{BASE_URL}/predict/client_activity",
        json=payload,
        headers={"Content-Type": "application/json"}
    )
    
    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}\n")

def test_combined_prediction():
    """Test combined prediction for both datasets"""
    print("Testing combined prediction...")
    
    payload = {
        "transaction": {
            "amount": 5000.0,
            "timestamp": "2025-11-28T14:30:00",
            "src_account_id": "123456",
            "beneficiary_id": "789012"
        },
        "activity": {
            "timestamp": "2025-11-28T14:30:00",
            "src_account_id": "123456",
            "logins_last_7_days": 5,
            "logins_last_30_days": 20,
            "login_frequency_7d": 0.71
        }
    }
    
    response = requests.post(
        f"{BASE_URL}/predict/combined",
        json=payload,
        headers={"Content-Type": "application/json"}
    )
    
    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}\n")

def test_various_amounts():
    """Test with various transaction amounts"""
    print("Testing various transaction amounts...\n")
    
    amounts = [100, 1000, 10000, 50000, 100000]
    
    for amount in amounts:
        payload = {
            "transaction": {
                "amount": float(amount),
                "timestamp": "2025-11-28T14:30:00",
                "src_account_id": "123456",
                "beneficiary_id": "789012"
            }
        }
        
        response = requests.post(
            f"{BASE_URL}/predict/transaction",
            json=payload,
            headers={"Content-Type": "application/json"}
        )
        
        if response.status_code == 200:
            result = response.json()
            print(f"Amount: ${amount:>7} -> Fraud Probability: {result['probability']:.3f} | Action: {result['action']}")
        else:
            print(f"Amount: ${amount:>7} -> Error: {response.status_code}")
    print()

if __name__ == "__main__":
    import sys
    
    print("="*60)
    print("Fraud Detection API Test Suite")
    print("="*60)
    print()
    
    try:
        # Run all tests
        test_health()
        test_transaction_prediction()
        test_client_activity_prediction()
        test_combined_prediction()
        test_various_amounts()
        
        print("="*60)
        print("All tests completed!")
        print("="*60)
        
    except requests.exceptions.ConnectionError:
        print("ERROR: Could not connect to API server!")
        print("Make sure to run the inference service first:")
        print("  python infer_service.py")
        sys.exit(1)
    except Exception as e:
        print(f"ERROR: {e}")
        sys.exit(1)
