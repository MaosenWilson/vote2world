"""Vote2World: consensus-shaped advantage estimation for verifiable world-model RL.

Built on the RLVR-World / iVideoGPT RT-1 single-step world model. The reward
stays a pure RLVR-World verifiable GT signal; intra-group consensus reshapes the
GRPO advantage to denoise that signal. See `dor.rewards`.
"""

__version__ = "0.2.0"
