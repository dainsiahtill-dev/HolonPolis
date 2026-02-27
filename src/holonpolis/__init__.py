"""HolonPolis - HarborPilot2 Evolution Engine.

A multi-agent system where:
- Each Agent (Holon) has its own isolated LanceDB memory
- Genesis (Evolution Lord) routes requests and spawns new Holons
- Evolution happens through Red-Green-Verify cycles
- No hardcoded Agent classes - all are runtime instances of Blueprints
"""

__version__ = "0.1.0"
