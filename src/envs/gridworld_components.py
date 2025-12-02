import numpy as np
from utils.constants import FruitStatus

class WorldObject():
    def __init__(self, pos: np.typing.NDArray[np.int32]) -> None:
        assert pos.shape == (2,), "position has to be a 1d array."
        
        self.pos = pos


class Human(WorldObject):
    def __init__(self, pos: np.typing.NDArray[np.int32], max_food: float) -> None:
        super().__init__(pos)
        self.max_food = max_food
        self.food = max_food
        self.alive = True

    def eat(self, amount: float) -> None:
        """add amount to total food without exceeding max food.

        Args:
            amount (float): amount to add to food
        """
        if self.food + amount > self.max_food:
            self.food = self.max_food
        else:
            self.food += amount
    
    def check_alive(self) -> bool:
        """checks if object is still alive.

        Returns:
            bool: True if alive
        """
        if self.food <= 0:
            self.alive = False
        return self.alive
    
class Fruit(WorldObject):
    def __init__(self, pos: np.typing.NDArray[np.int32], amount: float, reg_time: int) -> None:
        super().__init__(pos)
        self.amount = amount
        self.reg_time = reg_time
        self.time_until_reg = 0
        self.status = FruitStatus.RIPE

    def tick(self):
        """handle regeneration process.
        """
        if self.time_until_reg <= 0:
            self.status = FruitStatus.RIPE
        else:
            self.time_until_reg += -1

    def harvest(self) -> float:
        """harvest plant.

        Returns:
            float: amount of food on fruit or 0 if fruit isn't ripe
        """
        if self.status == FruitStatus.RIPE:
            self.status = FruitStatus.UNRIPE
            self.time_until_reg = self.reg_time
            return self.amount
        else:
            return 0
        

