from enum import Enum


class Distribution(Enum):
    GAUSSIAN = "gaussian"
    LAPLACIAN = "laplacian"
    
    @classmethod
    def from_string(cls, name: str) -> "Distribution":
        """Return the distribution corresponding to the given name."""
        try:
            for dist in cls:
                if dist.value == name.lower():
                    return dist
        except Exception as err:
            raise ValueError(f"Invalid distribution: {name}") from err
        
    def to_string(self) -> str:
        """Return the name of the distribution."""
        return self.value

