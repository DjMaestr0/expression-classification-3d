import os
import sys

BASE = r"C:\Users\Lenovo\Downloads\thesis_project"
sys.path.insert(0, os.path.join(BASE, "scripts"))
sys.path.insert(0, os.path.join(BASE, "smplx"))

os.environ["EMOTION_PRESET_PATH"] = os.path.join(BASE, "config", "emotion_anchor_boosted.json")
os.environ["EMOTION_PROTOTYPE_PATH"] = os.path.join(BASE, "config", "curated_action_bank_boosted.json")

import expression_anchors
expression_anchors._SIGNATURE_MAX = {}
expression_anchors._SIGNATURE_MIN = {}

import generate_smplx_mesh_vertices
generate_smplx_mesh_vertices.main()
