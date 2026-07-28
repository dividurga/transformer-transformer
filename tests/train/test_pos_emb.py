import torch

from t2.model.encoding import BinaryEncoding


def test_pos_emb():
    encoder = BinaryEncoding(num_bits=4)

    # Test case 1: Single number
    x = torch.tensor([5.0])
    out = encoder(x)
    expected = torch.tensor([[0.0, 1.0, 0.0, 1.0]])  # 5 = 0101 in binary
    assert torch.allclose(out, expected), f"Expected {expected}, got {out}"

    # Test case 2: Batch of numbers
    x = torch.tensor([[3.0, 7.0], [2.0, 4.0]])
    out = encoder(x)
    expected = torch.tensor(
        [
            [[0.0, 0.0, 1.0, 1.0], [0.0, 1.0, 1.0, 1.0]],
            [[0.0, 0.0, 1.0, 0.0], [0.0, 1.0, 0.0, 0.0]],
        ]
    )
    assert torch.allclose(out, expected), f"Expected {expected}, got {out}"


if __name__ == "__main__":
    test_pos_emb()
