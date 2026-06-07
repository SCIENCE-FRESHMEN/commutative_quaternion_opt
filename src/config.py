"""
Configuration management using dataclasses for type safety and clarity.
"""
from dataclasses import dataclass, field

@dataclass
class Algorithm1Config:
    """Configuration for Algorithm 1 (Multilinear Form Optimization)."""
    num_trials: int = 100
    gamma: float = 0.5  # Parameter for probability bound (0 < gamma < n / ln(n))
    epsilon: float = 0.05  # Failure probability

@dataclass
class Algorithm2Config:
    """Configuration for Algorithm 2 (Homogeneous Polynomial Optimization)."""
    alg1_config: Algorithm1Config = field(default_factory=Algorithm1Config)