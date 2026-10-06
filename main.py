"""
GraphPulse AI
FastAPI backend for GCN node classification on the Cora dataset.
"""

import os
import numpy as np

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
import onnxruntime as ort


# ============================================================
# Cora class mapping
# ============================================================

CORA_CLASSES = {
    0: "Case_Based",
    1: "Genetic_Algorithms",
    2: "Neural_Networks",
    3: "Probabilistic_Methods",
    4: "Reinforcement_Learning",
    5: "Rule_Learning",
    6: "Theory",
}


# ============================================================
# Paths
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(
    BASE_DIR,
    "simple_gcn_cora.onnx"
)

STATIC_DIR = os.path.join(
    BASE_DIR,
    "static"
)

CORA_FEATURES_PATH = os.path.join(
    BASE_DIR,
    "cora_features.npy"
)

CORA_EDGE_INDEX_PATH = os.path.join(
    BASE_DIR,
    "cora_edge_index.npy"
)


# ============================================================
# Constants
# ============================================================

FEATURE_DIM = 1433
NUM_CLASSES = 7


# ============================================================
# Load ONNX model
# ============================================================

model_session = ort.InferenceSession(
    MODEL_PATH,
    providers=["CPUExecutionProvider"]
)


# ============================================================
# Load lightweight Cora data
# ============================================================

try:
    cora_features = np.load(
        CORA_FEATURES_PATH,
        mmap_mode="r"
    )

    cora_edge_index = np.load(
        CORA_EDGE_INDEX_PATH,
        mmap_mode="r"
    )

except Exception as error:
    raise RuntimeError(
        f"Failed to load Cora NumPy data: {error}"
    )


# ============================================================
# Validate Cora data
# ============================================================

if cora_features.ndim != 2:
    raise RuntimeError(
        f"Cora features must be 2-dimensional, "
        f"got shape {cora_features.shape}"
    )

if cora_features.shape[1] != FEATURE_DIM:
    raise RuntimeError(
        f"Cora feature dimension must be {FEATURE_DIM}, "
        f"got {cora_features.shape[1]}"
    )

if cora_edge_index.ndim != 2 or cora_edge_index.shape[0] != 2:
    raise RuntimeError(
        f"Cora edge index must have shape [2, num_edges], "
        f"got {cora_edge_index.shape}"
    )


# ============================================================
# FastAPI
# ============================================================

app = FastAPI()


# ============================================================
# Request models
# ============================================================

class GraphPredictRequest(BaseModel):
    node_features: List[List[float]]
    edge_indices: Optional[List[List[int]]] = None


class CoraNodeRequest(BaseModel):
    node_indices: List[int]


# ============================================================
# Utility functions
# ============================================================

def softmax(scores: np.ndarray):
    """
    Numerically stable softmax.
    """
    shifted = scores - scores.max(
        axis=1,
        keepdims=True
    )

    exp_scores = np.exp(shifted)

    return exp_scores / exp_scores.sum(
        axis=-1,
        keepdims=True
    )


def run_model(
    node_features: np.ndarray,
    edge_index: np.ndarray,
    node_indices_to_return
):
    """
    Run ONNX GCN inference.
    """

    output = model_session.run(
        ["logits"],
        {
            "node_features": np.asarray(
                node_features,
                dtype=np.float32
            ),
            "edge_indices": np.asarray(
                edge_index,
                dtype=np.int64
            ),
        },
    )

    logits = output[0]

    probabilities = softmax(logits)

    predicted_classes = logits.argmax(axis=-1)

    results = []

    for i in node_indices_to_return:

        predicted_class_id = int(
            predicted_classes[i]
        )

        results.append({
            "node_index": int(i),

            "predicted_class_id":
                predicted_class_id,

            "predicted_class_name":
                CORA_CLASSES[predicted_class_id],

            "probabilities":
                probabilities[i].tolist(),

            "logits":
                logits[i].tolist(),
        })

    return {
        "num_nodes":
            int(node_features.shape[0]),

        "num_edges":
            int(edge_index.shape[1]),

        "predictions":
            results
    }


# ============================================================
# Home
# ============================================================

@app.get("/")
def home_page():

    index_file = os.path.join(
        STATIC_DIR,
        "index.html"
    )

    if os.path.exists(index_file):

        return FileResponse(index_file)

    return {
        "service": "GraphPulse AI",
        "status": "Running"
    }


# ============================================================
# Health check
# ============================================================

@app.get("/health")
def health_check():

    return {
        "status": "healthy",
        "providers":
            model_session.get_providers()
    }


# ============================================================
# Model information
# ============================================================

@app.get("/info")
def model_info():

    return {
        "Model Name": "SimpleGCN",

        "feature_dimension":
            FEATURE_DIM,

        "num_classes":
            NUM_CLASSES,

        "class_mapping":
            CORA_CLASSES,

        "inputs": [
            {
                "name": inp.name,
                "shape": inp.shape,
                "type": inp.type
            }
            for inp in model_session.get_inputs()
        ],

        "outputs": [
            {
                "name": out.name,
                "shape": out.shape,
                "type": out.type
            }
            for out in model_session.get_outputs()
        ],
    }


# ============================================================
# Custom graph prediction
# ============================================================

@app.post("/predict")
def predict_custom_graph(
    request: GraphPredictRequest
):

    if not request.node_features:

        raise HTTPException(
            status_code=400,
            detail="Node features cannot be empty"
        )

    # Validate feature dimensions
    for feature_vector in request.node_features:

        if len(feature_vector) != FEATURE_DIM:

            raise HTTPException(
                status_code=422,
                detail=(
                    f"Each node's feature vector "
                    f"must exactly have "
                    f"{FEATURE_DIM} features"
                )
            )

    node_features = np.asarray(
        request.node_features,
        dtype=np.float32
    )

    num_nodes = node_features.shape[0]

    # --------------------------------------------------------
    # Edge handling
    # --------------------------------------------------------

    if request.edge_indices:

        edge_index = np.asarray(
            request.edge_indices,
            dtype=np.int64
        )

        if (
            edge_index.ndim != 2
            or edge_index.shape[0] != 2
        ):

            raise HTTPException(
                status_code=422,
                detail=(
                    "edge_indices must have "
                    "shape [2, num_edges]"
                )
            )

        # Validate node IDs
        if edge_index.size > 0:

            if (
                edge_index.min() < 0
                or edge_index.max() >= num_nodes
            ):

                raise HTTPException(
                    status_code=422,
                    detail=(
                        "edge_indices contain "
                        "invalid node IDs"
                    )
                )

    else:

        # Self-loops when no graph is supplied
        node_ids = np.arange(
            num_nodes,
            dtype=np.int64
        )

        edge_index = np.vstack(
            [node_ids, node_ids]
        )

    all_node_indices = list(
        range(num_nodes)
    )

    return run_model(
        node_features,
        edge_index,
        all_node_indices
    )


# ============================================================
# Real Cora node prediction
# ============================================================

@app.post("/predict/cora_node")
def predict_real_cora_nodes(
    request: CoraNodeRequest
):

    # --------------------------------------------------------
    # Validate request
    # --------------------------------------------------------

    if not request.node_indices:

        raise HTTPException(
            status_code=400,
            detail="node_indices cannot be empty"
        )

    num_nodes = cora_features.shape[0]

    largest_valid_index = num_nodes - 1

    invalid_index = [
        i
        for i in request.node_indices
        if i < 0
        or i > largest_valid_index
    ]

    if invalid_index:

        raise HTTPException(
            status_code=400,
            detail=(
                f"node_index out of bounds "
                f"(must be 0 to "
                f"{largest_valid_index})"
            )
        )

    # --------------------------------------------------------
    # Run model using lightweight NumPy Cora data
    # --------------------------------------------------------

    return run_model(
        cora_features,
        cora_edge_index,
        request.node_indices
    )


# ============================================================
# Static frontend
# ============================================================

if os.path.isdir(STATIC_DIR):

    app.mount(
        "/",
        StaticFiles(
            directory=STATIC_DIR,
            html=True
        ),
        name="static"
    )