"""Solution for Similar Items task"""
from typing import Dict
from typing import List
from typing import Tuple

import numpy as np
from scipy.spatial.distance import cosine
from itertools import combinations


class SimilarItems:
    """Similar items class"""

    @staticmethod
    def similarity(embeddings: Dict[int, np.ndarray]) -> Dict[Tuple[int, int], float]:
        """Calculate pairwise similarities between each item
        in embedding.

        Args:
            embeddings (Dict[int, np.ndarray]): Items embeddings.

        Returns:
            Tuple[List[str], Dict[Tuple[int, int], float]]:
            List of all items + Pairwise similarities dict
            Keys are in form of (i, j) - combinations pairs of item_ids
            with i < j.
            Round each value to 8 decimal places.
        """
        pair_sims = {}
        emb_keys = list(embeddings.keys())
        emb_keys.sort()
        emb_sort = {key: embeddings[key] for key in emb_keys}
        comb = combinations(emb_sort, 2)
        for id1, id2 in comb:
            cos_sim = round(1 - cosine(emb_sort[id1], emb_sort[id2]), 8)
            pair_sims[(id1, id2)] = cos_sim  
        
        return pair_sims

    @staticmethod
    def knn(
        sim: Dict[Tuple[int, int], float], top: int
    ) -> Dict[int, List[Tuple[int, float]]]:
        """Return closest neighbors for each item.

        Args:
            sim (Dict[Tuple[int, int], float]): <similarity> method output.
            top (int): Number of top neighbors to consider.

        Returns:
            Dict[int, List[Tuple[int, float]]]: Dict with top closest neighbors
            for each item.
        """
        ids1 = {x[0] for x in sim.keys()}
        ids2 = {x[1] for x in sim.keys()}
        ids = ids1.union(ids2)
        knn_dict = {}
        for i in ids:
            i_list = []
            for keys, value in sim.items():
                if i in keys:
                    i_list.append((keys[0]+keys[1]-i, value))
            knn_dict[i] = sorted(i_list, key=lambda tup: tup[1], reverse=True)[:top]
            
        return knn_dict

    @staticmethod
    def knn_price(
        knn_dict: Dict[int, List[Tuple[int, float]]],
        prices: Dict[int, float],
    ) -> Dict[int, float]:
        """Calculate weighted average prices for each item.
        Weights should be positive numbers in [0, 2] interval.

        Args:
            knn_dict (Dict[int, List[Tuple[int, float]]]): <knn> method output.
            prices (Dict[int, float]): Price dict for each item.

        Returns:
            Dict[int, float]: New prices dict, rounded to 2 decimal places.
        """
        knn_price_dict = {}
        for key, value in knn_dict.items():
            weight_price = 0
            weight = 0
            for item in value:
                weight_price+=prices[item[0]]*(item[1]+1)
                weight+=(item[1]+1)
            knn_price_dict[key] = round(weight_price/weight, 2)
        return knn_price_dict

    @staticmethod
    def transform(
        embeddings: Dict[int, np.ndarray],
        prices: Dict[int, float],
        top: int,
    ) -> Dict[int, float]:
        """Transforming input embeddings into a dictionary
        with weighted average prices for each item.

        Args:
            embeddings (Dict[int, np.ndarray]): Items embeddings.
            prices (Dict[int, float]): Price dict for each item.
            top (int): Number of top neighbors to consider.

        Returns:
            Dict[int, float]: Dict with weighted average prices for each item.
        """
        pair_sims = SimilarItems.similarity(embeddings)
        knn_dict = SimilarItems.knn(pair_sims, top)
        knn_price_dict = SimilarItems.knn_price(knn_dict, prices)
        return knn_price_dict
