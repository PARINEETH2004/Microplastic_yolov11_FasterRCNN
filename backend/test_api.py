import requests


def test_detection_api():
    """Test the detection API with a sample image"""
    url = "http://localhost:5000/api/detect"

    # Test health endpoint first
    health_response = requests.get("http://localhost:5000/api/health")
    print("Health check:", health_response.json())

    # Create a simple test image using PIL
    try:
        from PIL import Image
        import io

        # Create a simple test image
        img = Image.new('RGB', (640, 480), color='white')
        # Add some random shapes to simulate particles
        import random
        for _ in range(10):
            x, y = random.randint(50, 590), random.randint(50, 430)
            size = random.randint(10, 50)
            for i in range(size):
                for j in range(size):
                    if random.random() > 0.3:  # Make it look like particles
                        img.putpixel((x+i, y+j), (random.randint(100, 255),
                                     random.randint(100, 255), random.randint(100, 255)))

        # Save to bytes
        img_bytes = io.BytesIO()
        img.save(img_bytes, format='JPEG')
        img_bytes.seek(0)

        files = {'image': ('test_image.jpg', img_bytes, 'image/jpeg')}
        data = {'mode': 'fast'}

        print("Sending test image to detection API...")
        response = requests.post(url, files=files, data=data)
        print("Detection API Response Status:", response.status_code)
        if response.status_code == 200:
            result = response.json()
            print("Detection Results:")
            print(f"  Total detections: {result['totalCount']}")
            print(f"  Processing time: {result['processingTime']}ms")
            print(f"  Count by type: {result['countByType']}")
            print("  First few detections:")
            for i, det in enumerate(result['detections'][:3]):
                print(
                    f"    {i+1}. {det['particleType']} ({det['confidence']:.2f})")
        else:
            print("Error:", response.text)

    except Exception as e:
        print(f"Test failed: {e}")


if __name__ == "__main__":
    test_detection_api()
