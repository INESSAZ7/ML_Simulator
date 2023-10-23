import hashlib
import random
from typing import Tuple, List

class InvalidWeightData(ValueError):
    '''Cutom class for check data and raise error'''
    def __init__(self, message):
        super().__init__(message)
        self.msgfmt = message

class Experiment:
    """Experiment class. Contains the logic for assigning users to groups."""

    def __init__(
        self,
        experiment_id: int,
        groups: Tuple[str] = ("A", "B"),
        group_weights: List[float] = None,
    ):
        self.experiment_id = experiment_id
        self.groups = groups
        self.group_weights = group_weights
        # Define the salt for experiment_id.
        # The salt should be deterministic and unique for each experiment_id.
        self.salt = str(experiment_id)
        # Define the group weights if they are not provided equaly distributed
        # Check input group weights. They must be non-negative and sum to 1.
        if self.group_weights:
            w_sum = 0
            for wieght in group_weights:
                if wieght < 0:
                    raise InvalidWeightData("Wieght is negative")
                w_sum += wieght
            if w_sum != 1:
                raise InvalidWeightData("Sum of weights not equal to 1")

    def group(self, click_id: int) -> Tuple[int, str]:
        """Assigns a click to a group.

        Parameters
        ----------
        click_id: int :
            id of the click

        Returns
        -------
        Tuple[int, str] :
            group id and group name
        """

        # Assign the click to a group randomly based on the group weights
        # Return the group id and group name

        click_id = str(click_id)
        click_hash = hashlib.sha256(str(click_id+self.salt).encode('utf-8')).hexdigest()
        click_hash = int(click_hash, 16)
        if self.group_weights:
            random.seed(click_hash)
            group_ids = range(len(self.groups))
            group_id = random.choices(group_ids, self.group_weights)[0]  
        else:
            group_id = click_hash%len(self.groups)

        return group_id, self.groups[group_id]
