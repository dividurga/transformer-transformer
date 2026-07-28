import torch
import torch.nn as nn


class LogEncoding(nn.Module):
    def __init__(self, offset: float = 0.01):
        super().__init__()
        self.offset = offset

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.log(x + self.offset)

    def backward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.exp(x) - self.offset


class LogDecoding(nn.Module):
    def __init__(self, offset: float = 0.01):
        super().__init__()
        self.offset = offset

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.exp(x) - self.offset


class SignedLogEncoding(nn.Module):
    """Encode values using sign + log(abs(x) + offset).

    This handles negative values and zeros by:
    1. Storing the sign separately (0 for non-negative, 1 for negative)
    2. Adding an offset before taking log to handle zeros

    Output shape: (..., 2) where [..., 0] is the log magnitude and [..., 1] is the sign.
    """

    def __init__(self, offset: float = 0.01):
        super().__init__()
        self.offset = offset

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode tensor using signed log.

        Args:
            x: Input tensor of shape (..., D)

        Returns:
            Encoded tensor of shape (..., D*2) with [log_mag, sign] pairs
        """
        sign = (x < 0).float()
        log_mag = torch.log(torch.abs(x) + self.offset)
        # Interleave log_mag and sign: [log0, sign0, log1, sign1, ...]
        result = torch.stack([log_mag, sign], dim=-1)
        return result.reshape(*x.shape[:-1], -1)


class SignedLogDecoding(nn.Module):
    """Decode values from sign + log(abs(x) + offset) encoding.

    Reverses SignedLogEncoding by:
    1. Extracting sign (threshold at 0.5)
    2. Exponentiating and subtracting offset
    3. Applying sign
    """

    def __init__(self, offset: float = 0.01):
        super().__init__()
        self.offset = offset

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Decode tensor from signed log encoding.

        Args:
            x: Encoded tensor of shape (..., D*2) with [log_mag, sign] pairs

        Returns:
            Decoded tensor of shape (..., D)
        """
        # Reshape to separate log_mag and sign
        x = x.reshape(*x.shape[:-1], -1, 2)
        log_mag = x[..., 0]
        sign = x[..., 1]
        # Decode magnitude
        mag = torch.exp(log_mag) - self.offset
        # Apply sign (sign >= 0.5 means negative)
        result = torch.where(sign >= 0.5, -mag, mag)
        return result


class BinaryEncoding(nn.Module):
    def __init__(self, num_bits: int, offset: int = 0):
        super().__init__()
        self.num_bits = num_bits
        self.min_value = -offset
        self.max_value = 2**num_bits - 1 - offset
        self.offset = offset

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Convert input tensor to binary encoding.

        Args:
            x: Input tensor of shape (...) with values in [0, 2**num_bits)

        Returns:
            Binary encoded tensor of shape (..., num_bits)
        """
        # Create powers of 2 for binary encoding
        assert x.min() >= self.min_value and x.max() <= self.max_value, (
            f"x.min() {x.min()} >= {self.min_value} and x.max() {x.max()} <= {self.max_value}"
        )
        powers = 2 ** torch.arange(self.num_bits - 1, -1, -1, device=x.device)
        # Expand x and powers for broadcasting
        x_expanded = x.unsqueeze(-1) + self.offset
        powers = powers.view([1] * len(x.shape) + [-1])
        # Compute binary encoding
        binary = (x_expanded // powers) % 2
        return binary.to(x.dtype)


class BinaryDecoding(nn.Module):
    def __init__(self, num_bits: int, offset: int = 0):
        super().__init__()
        self.num_bits = num_bits
        self.offset = offset

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.clip(0, 1)
        return (
            torch.sum(
                torch.round(x)
                * (2 ** torch.arange(self.num_bits - 1, -1, -1, device=x.device)),
                dim=-1,
            ).long()
            + self.offset
        )


class OffsettedEmbedding(nn.Module):
    def __init__(self, embedding: nn.Embedding, offset: int):
        super().__init__()
        self.embedding = embedding
        self.offset = offset

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.embedding(x + self.offset)


if __name__ == "__main__":
    # Test BinaryEncoding/Decoding
    num_bits = 8
    x = torch.randint(0, 2**num_bits, (1000,))
    encoder = BinaryEncoding(num_bits)
    decoder = BinaryDecoding(num_bits)
    assert torch.allclose(x, decoder(encoder(x)))
    print("BinaryEncoding/Decoding test passed!")

    # Test SignedLogEncoding/Decoding
    print("\nTesting SignedLogEncoding/Decoding...")

    # Test with various value ranges
    test_cases = [
        # Positive values (like kp, kv)
        torch.tensor([[60.0, 200.0], [15000.0, 20000.0]]),
        # Negative and positive values (like force_range)
        torch.tensor([[-100.0, 100.0], [-5000.0, 5000.0]]),
        # Zeros and small values (like stiffness, frictionloss)
        torch.tensor([[0.0, 0.2], [0.0, 0.0]]),
        # Mixed signs and magnitudes
        torch.tensor([[-0.01, 0.01], [-1000.0, 0.001]]),
    ]

    for i, x in enumerate(test_cases):
        signed_encoder = SignedLogEncoding(offset=0.01)
        signed_decoder = SignedLogDecoding(offset=0.01)

        encoded = signed_encoder(x)
        decoded = signed_decoder(encoded)

        print(f"\nTest case {i + 1}:")
        print(f"  Original shape: {x.shape}, Encoded shape: {encoded.shape}")
        print(f"  Original: {x}")
        print(f"  Decoded:  {decoded}")

        # Check shape: should double the last dimension
        assert encoded.shape[-1] == x.shape[-1] * 2, (
            f"Encoded shape {encoded.shape} should have 2x last dim of {x.shape}"
        )

        # Check reconstruction (with tolerance for floating point)
        assert torch.allclose(x, decoded, atol=1e-5), (
            f"Reconstruction failed: max diff = {(x - decoded).abs().max()}"
        )
        print(f"  Max reconstruction error: {(x - decoded).abs().max():.2e}")

    print("\nAll SignedLogEncoding/Decoding tests passed!")
