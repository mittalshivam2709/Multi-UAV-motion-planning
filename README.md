# Multi-UAV Motion Planning using Adversarial Imitation Learning

This repository contains the codebase for **adversarial learning-based motion planning** in **Multi-Unmanned Aerial Vehicle (UAV)** systems. The project integrates classical motion planning strategies with modern imitation learning frameworks to enable decentralized, collision-free navigation in complex environments.

## 📌 Project Overview

This project introduces an adversarial imitation learning framework focused on generating **efficient and collision-free trajectories** for multiple UAVs in **dynamic 2D and 3D environments**.

- ✅ **Hybrid Expert Policy**: Combines **Velocity Obstacle (VO)** and **Rapidly-exploring Random Tree (RRT)** methods in 2D as a baseline.
- 🤖 **Adversarial Imitation Learning**: Implements **Generative Adversarial Imitation Learning (GAIL)** trained on expert demonstrations.
- 🌍 **Simulation in AirSim**: Integrates with **Microsoft AirSim** for high-fidelity 3D navigation experiments.
- 🚁 **Multi-UAV Coordination**: Demonstrates effective decentralized planning and obstacle avoidance.

---

### Prerequisites

- Python 3.8+
- [Microsoft AirSim](https://github.com/microsoft/AirSim)
- PyTorch
- NumPy
- OpenCV
- matplotlib

### Installation

```bash
# Clone the repository
git clone https://github.com/mittalshivam2709/Multi-UAV-motion-planning.git
cd Multi-UAV-motion-planning

```
