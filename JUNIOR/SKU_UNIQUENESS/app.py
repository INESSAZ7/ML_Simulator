from typing import Tuple
from sklearn.neighbors import KernelDensity

import os
import numpy as np
import uvicorn
from fastapi import FastAPI
from fastapi_utils.tasks import repeat_every


DIVERSITY_THRESHOLD = 10

app = FastAPI()
embeddings = {}


@app.on_event("startup")
@repeat_every(seconds=10)
def load_embeddings() -> dict:
    """Load embeddings from file."""

    # Load new embeddings each 10 seconds
    path = os.path.join(os.path.dirname(__file__), "embeddings.npy")
    embeddings_raw = np.load(path, allow_pickle=True).item()
    for item_id, embedding in embeddings_raw.items():
        embeddings[item_id] = embedding

    return {}


@app.get("/uniqueness/")
def uniqueness(item_ids: str) -> dict:
    """Calculate uniqueness of each product"""

    # Parse item IDs
    item_ids = [int(item) for item in item_ids.split(",")]

    # Default answer
    item_uniqueness = {item_id: 0.0 for item_id in item_ids}

    # Calculate uniqueness
    item_embeddings = []
    for item_id in item_ids:
        item_embeddings.append(embeddings[item_id])
    uniq = kde_uniqueness(item_embeddings)

    i = 0
    for item_id in item_ids:
        item_uniqueness[item_id] = uniq[i]
        i+=1

    return item_uniqueness


@app.get("/diversity/")
def diversity(item_ids: str) -> dict:
    """Calculate diversity of group of products"""

    # Parse item IDs
    item_ids = [int(item) for item in item_ids.split(",")]

    # Calculate diversity
    response = {"diversity": 0.0, "reject": True}
    
    item_embeddings = []
    for item_id in item_ids:
        item_embeddings.append(embeddings[item_id])
    
    diversity = group_diversity(item_embeddings,DIVERSITY_THRESHOLD)    
    response["diversity"] = float(diversity[1])
    response["reject"] = bool(diversity[0])
    return response


def kde_uniqueness(embeddings: np.ndarray) -> np.ndarray:
    """Estimate uniqueness of each item in item embeddings group. Based on KDE.

    Parameters
    ----------
    embeddings: np.ndarray :
        embeddings group

    Returns
    -------
    np.ndarray
        uniqueness estimates

    """
    # Fit a kernel density estimator to the item embedding space
    kde = KernelDensity().fit(embeddings)

    uniqueness = []
    for item in embeddings:
        uniqueness.append(1 / np.exp(kde.score_samples([item])[0]))

    return np.array(uniqueness)


def group_diversity(embeddings: np.ndarray, threshold: float) -> Tuple[bool, float]:
    """Calculate group diversity based on kde uniqueness.

    Parameters
    ----------
    embeddings: np.ndarray :
        embeddings group
    threshold: float :
       group deversity threshold for reject group

    Returns
    -------
    Tuple[bool, float]
        reject
        group diverstity

    """
    diversity = np.sum(kde_uniqueness(embeddings)) / len(embeddings)
    reject = diversity < threshold
    return reject, diversity


def main() -> None:
    """Run application"""
    uvicorn.run("app:app", host="localhost", port=5000)


if __name__ == "__main__":
    main()
