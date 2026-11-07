# Contributing to NASA Mars Rover Terrain Segmentation

Thank you for your interest in contributing to the **NASA Mars Rover Terrain Segmentation & AutoNav System**! This document provides guidelines for bug reports, feature proposals, and pull requests.

---

## 🛠️ Development Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/sushantguri/-Rover.git
   cd -Rover
   ```

2. **Create and activate a virtual environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

4. **Verify environment with the test suite:**
   ```bash
   python -m unittest discover -s tests
   ```

---

## 🧪 Testing Guidelines

Before submitting any Pull Request, ensure all existing and new unit tests pass:
```bash
# Run full automated test suite
python -m unittest discover -s tests

# Test CLI prediction on sample images
python scripts/predict.py --input sample_images/test_rocky_terrain.jpg --output-dir outputs/

# Verify API endpoints
python -m unittest tests/test_api.py
```

---

## 🚀 Contribution Opportunities

We welcome contributions in several key areas:
- **New Rover Camera Profiles:** Adding calibration and masking parameters for **Perseverance Mastcam-Z**, **Curiosity Mastcam-Right**, or **ExoMars Rosalind Franklin PanCam**.
- **Alternative Encoders:** Benchmarking lighter backbones (e.g. `MobileNetV3`, `EfficientNet-B0`) for extreme low-power embedded processors.
- **ROS 2 Nav2 Plugin:** Packaging a native C++ / Python ROS 2 Nav2 costmap recovery layer plugin.
- **Uncertainty Estimation:** Adding Bayesian dropout or Monte Carlo estimation to flag uncertain pixels in low-contrast dust storms.

---

## 📝 Commit Convention

We follow conventional commits for git history:
- `feat(...)`: A new feature or capability.
- `fix(...)`: A bug fix.
- `docs(...)`: Documentation updates.
- `test(...)`: Adding or refactoring tests.
- `refactor(...)`: Code refactoring without changing functionality.
- `ci(...)`: GitHub Actions or CI configuration changes.

---

## 📄 License & Attribution

This project is released under the **MIT License**. Data and mission imagery provided courtesy of NASA Jet Propulsion Laboratory (JPL) and Caltech.

