import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from perception.perception_agent import PerceptionAgent


def test_real_model():
    print("=" * 55)
    print("TEST: Real Model Integration")
    print("=" * 55)

    # Load real model
    agent = PerceptionAgent(
        model_path="perception/checkpoints/best.pt",
        run_jensen_gain=True,
        n_elevation=64,
        n_inplane=16,
        n_jensen_rotations=8  # use 8 for speed during testing
    )

    # Test with random image (synthetic)
    print("\nTest 1: Random noise image (should be LOW confidence)")
    image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    output = agent.predict(image)
    print(f"  Jensen Gain: {output.uncertainty.jensen_gain:.2f}°")
    print(f"  Confidence: {output.uncertainty.confidence_label}")
    print(f"  Trustworthy: {output.is_trustworthy}")
    print(f"  Quaternion: {[round(q,3) for q in output.pose.quaternion]}")
    print(f"  Translation: {[round(t,3) for t in output.pose.t]}")
    print(f"  Processing: {output.metadata['processing_time_ms']:.1f}ms")

    # Test with black image
    print("\nTest 2: Black image (no spacecraft visible)")
    black = np.zeros((224, 224, 3), dtype=np.uint8)
    output2 = agent.predict(black)
    print(f"  Jensen Gain: {output2.uncertainty.jensen_gain:.2f}°")
    print(f"  Confidence: {output2.uncertainty.confidence_label}")

    # Test with white image
    print("\nTest 3: White image (overexposed / sun glare)")
    white = np.ones((224, 224, 3), dtype=np.uint8) * 255
    output3 = agent.predict(white)
    print(f"  Jensen Gain: {output3.uncertainty.jensen_gain:.2f}°")
    print(f"  Confidence: {output3.uncertainty.confidence_label}")

    print("\nJSON output sample:")
    print(output.to_json())

    print("\n" + "=" * 55)
    print("REAL MODEL INTEGRATION TEST COMPLETE")
    print("=" * 55)


if __name__ == "__main__":
    test_real_model()