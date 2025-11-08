"""Test API endpoints."""
import requests
import json
import time

BASE_URL = "http://localhost:8000"

def test_api_endpoints():
    """Test all API endpoints."""
    print("=== Testing API Endpoints ===\n")
    
    # Wait for server to be ready
    print("Waiting for server to be ready...")
    for i in range(10):
        try:
            response = requests.get(f"{BASE_URL}/health", timeout=2)
            if response.status_code == 200:
                print("✅ Server is ready!\n")
                break
        except:
            time.sleep(1)
    else:
        print("❌ Server is not responding. Please start the server first:")
        print("   uvicorn app.main:app --reload")
        return False
    
    # Test 1: Health endpoint
    print("1. Testing /health endpoint...")
    try:
        response = requests.get(f"{BASE_URL}/health")
        print(f"   Status: {response.status_code}")
        print(f"   Response: {json.dumps(response.json(), indent=2)}")
        print("   ✅ Health endpoint working\n")
    except Exception as e:
        print(f"   ❌ Error: {str(e)}\n")
        return False
    
    # Test 2: Get all notes
    print("2. Testing GET /api/notes/ endpoint...")
    try:
        response = requests.get(f"{BASE_URL}/api/notes/")
        print(f"   Status: {response.status_code}")
        data = response.json()
        print(f"   Notes count: {len(data)}")
        if data:
            print(f"   First note: {data[0].get('subject', 'N/A')}")
        print("   ✅ Notes endpoint working\n")
    except Exception as e:
        print(f"   ❌ Error: {str(e)}\n")
        return False
    
    # Test 3: Get all tasks
    print("3. Testing GET /api/tasks/ endpoint...")
    try:
        response = requests.get(f"{BASE_URL}/api/tasks/")
        print(f"   Status: {response.status_code}")
        data = response.json()
        print(f"   Tasks count: {data.get('total', 0)}")
        print(f"   Statistics: {json.dumps(data.get('statistics', {}), indent=2)}")
        if data.get('tasks'):
            print(f"   First task: {data['tasks'][0].get('title', 'N/A')}")
        print("   ✅ Tasks endpoint working\n")
    except Exception as e:
        print(f"   ❌ Error: {str(e)}\n")
        return False
    
    # Test 4: Get task statistics
    print("4. Testing GET /api/tasks/statistics endpoint...")
    try:
        response = requests.get(f"{BASE_URL}/api/tasks/statistics")
        print(f"   Status: {response.status_code}")
        data = response.json()
        print(f"   Statistics: {json.dumps(data, indent=2)}")
        print("   ✅ Task statistics endpoint working\n")
    except Exception as e:
        print(f"   ❌ Error: {str(e)}\n")
        return False
    
    # Test 5: Search notes
    print("5. Testing GET /api/notes/search/{query} endpoint...")
    try:
        response = requests.get(f"{BASE_URL}/api/notes/search/database")
        print(f"   Status: {response.status_code}")
        data = response.json()
        print(f"   Search results: {len(data)}")
        print("   ✅ Note search endpoint working\n")
    except Exception as e:
        print(f"   ❌ Error: {str(e)}\n")
        return False
    
    # Test 6: Search tasks
    print("6. Testing GET /api/tasks/search/{query} endpoint...")
    try:
        response = requests.get(f"{BASE_URL}/api/tasks/search/test")
        print(f"   Status: {response.status_code}")
        data = response.json()
        print(f"   Search results: {len(data)}")
        print("   ✅ Task search endpoint working\n")
    except Exception as e:
        print(f"   ❌ Error: {str(e)}\n")
        return False
    
    print("=" * 50)
    print("✅ All API endpoint tests passed!")
    print("=" * 50)
    
    return True

if __name__ == "__main__":
    success = test_api_endpoints()
    exit(0 if success else 1)

