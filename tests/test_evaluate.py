import pytest
import torch

from src.evaluate import (average_accuracy_by_offset,
                          compute_recall_accuracies,
                          correct_retrieval_counts_by_timestep)
from src.synthetic_datasets import generate_vectors


def test_compute_recall_accuracies_perfect_predictions():
    """Test compute_recall_accuracies with perfect predictions."""
    B, seq_len, value_dim = 2, 3, 4

    # Create values using generate_vectors to get well-separated unit vectors
    values = torch.zeros(B, seq_len, value_dim)
    for b in range(B):
        values[b] = generate_vectors(seq_len, value_dim, correlation=0.0)

    # Create predictions that exactly match the values
    predictions = []
    for t in range(seq_len):
        pred_t = values[:, : t + 1].clone()
        predictions.append(pred_t)

    accuracies = compute_recall_accuracies(predictions, values)

    assert len(accuracies) == seq_len
    for t, acc in enumerate(accuracies):
        assert acc.shape == (B, t + 1)
        # With the exact same vectors, we should get perfect accuracy
        assert torch.allclose(acc, torch.ones_like(acc))


def test_compute_recall_accuracies_with_imperfect_predictions():
    """Test compute_recall_accuracies with imperfect predictions."""
    torch.manual_seed(42)

    B, seq_len, value_dim = 100, 40, 50

    # Create values using generate_vectors
    values = torch.zeros(B, seq_len, value_dim)
    for b in range(B):
        values[b] = generate_vectors(seq_len, value_dim, correlation=0.0)

    # Create imperfect predictions using independent generate_vectors calls
    predictions = []
    for t in range(seq_len):
        pred_t = torch.zeros(B, t + 1, value_dim)
        for b in range(B):
            pred_t[b] = generate_vectors(t + 1, value_dim, correlation=0.0)
        predictions.append(pred_t)

    accuracies = compute_recall_accuracies(predictions, values)

    assert len(accuracies) == seq_len
    for t, acc in enumerate(accuracies):
        assert acc.shape == (B, t + 1)
        # With random predictions, accuracy should be around chance level (1/(t+1))
        chance_level = 1.0 / (t + 1)
        # Allow some tolerance around chance level
        assert abs(acc.mean().item() - chance_level) < 0.1


def test_compute_recall_accuracies_device_handling():
    """Test that compute_recall_accuracies handles device mismatch correctly."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available.")

    B, seq_len, value_dim = 4, 5, 6

    # Create values using generate_vectors
    values_cpu = torch.zeros(B, seq_len, value_dim)
    for b in range(B):
        values_cpu[b] = generate_vectors(seq_len, value_dim, correlation=0.0)

    device = torch.device("cuda", torch.cuda.current_device())
    values_gpu = values_cpu.to(device)

    predictions = []
    for t in range(seq_len):
        pred_t = values_gpu[:, : t + 1].clone()
        predictions.append(pred_t)

    # Should not raise an error and should move values to correct device
    accuracies = compute_recall_accuracies(predictions, values_cpu)
    assert len(accuracies) == seq_len
    assert all(acc.device == device for acc in accuracies)


def test_compute_recall_accuracies_error_cases():
    """Test error handling in compute_recall_accuracies."""
    B, seq_len, value_dim = 20, 30, 40

    # Test empty predictions
    with pytest.raises(ValueError, match="predictions list must not be empty"):
        compute_recall_accuracies([], torch.randn(B, seq_len, value_dim))

    # Test wrong values dimensions
    values = torch.randn(B, seq_len)  # Missing value_dim
    predictions = [torch.randn(B, t + 1, value_dim) for t in range(seq_len)]

    with pytest.raises(
        ValueError, match="values tensor must be 3-dimensional"
    ):
        compute_recall_accuracies(predictions, values)

    # Test sequence length mismatch
    values = torch.randn(B, seq_len + 1, value_dim)  # Wrong sequence length
    with pytest.raises(
        ValueError, match="Number of predictions must match sequence_length"
    ):
        compute_recall_accuracies(predictions, values)


def test_average_accuracy_by_offset_perfect():
    """Test average_accuracy_by_offset with perfect accuracy."""
    B, seq_len = 30, 40

    # Create perfect accuracy history
    accuracy_history = []
    for t in range(seq_len):
        acc_t = torch.ones(B, t + 1)  # Perfect accuracy
        accuracy_history.append(acc_t)

    mean_accuracy, counts = average_accuracy_by_offset(accuracy_history)

    # Should have seq_len offsets (0 to seq_len-1)
    assert len(mean_accuracy) == seq_len
    assert len(counts) == seq_len

    # Perfect accuracy should give mean of 1.0 for all offsets that occur
    valid_mask = ~torch.isnan(mean_accuracy)
    assert torch.allclose(
        mean_accuracy[valid_mask], torch.ones_like(mean_accuracy[valid_mask])
    )
    expected_counts = torch.tensor(
        [(seq_len - i) * B for i in range(seq_len)],
        device=counts.device,
        dtype=counts.dtype,
    )
    assert torch.equal(counts, expected_counts)


def test_average_accuracy_by_offset_mixed_accuracy():
    """Test average_accuracy_by_offset with varying accuracy."""
    B, seq_len = 2, 4

    # Create mixed accuracy history
    accuracy_history = []
    for t in range(seq_len):
        # Create varying accuracy: higher for recent keys, lower for older keys
        acc_t = torch.linspace(0.3, 0.9, t + 1).expand(B, -1)
        accuracy_history.append(acc_t)

    mean_accuracy, counts = average_accuracy_by_offset(accuracy_history)

    assert len(mean_accuracy) == seq_len
    assert len(counts) == seq_len

    # Expected means per offset after flip: [0.75, 0.5333..., 0.4, 0.3]
    expected_mean = torch.tensor([0.75, 0.53333336, 0.4, 0.3])
    assert torch.allclose(mean_accuracy, expected_mean)

    expected_counts = torch.tensor(
        [(seq_len - i) * B for i in range(seq_len)],
        device=counts.device,
        dtype=counts.dtype,
    )
    assert torch.equal(counts, expected_counts)


def test_average_accuracy_by_offset_empty():
    """Test average_accuracy_by_offset with empty input."""
    with pytest.raises(ValueError, match="accuracy_history is empty"):
        average_accuracy_by_offset([])


def test_average_accuracy_by_offset_preserves_device():
    """Ensure average_accuracy_by_offset keeps tensors on the source device."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available.")

    B, seq_len = 2, 3
    device = torch.device("cuda", torch.cuda.current_device())
    accuracy_history = []
    for t in range(seq_len):
        acc_t = torch.ones(B, t + 1, device=device)
        accuracy_history.append(acc_t)

    mean_accuracy, counts = average_accuracy_by_offset(accuracy_history)
    assert mean_accuracy.device == device
    assert counts.device == device


def test_correct_retrieval_counts_by_timestep():
    """Test correct_retrieval_counts_by_timestep."""
    B, seq_len = 20, 30

    # Create accuracy history with known values
    accuracy_history = []
    for t in range(seq_len):
        acc_t = torch.ones(B, t + 1) * 0.5  # 50% accuracy
        accuracy_history.append(acc_t)

    counts = correct_retrieval_counts_by_timestep(accuracy_history)

    assert len(counts) == seq_len
    # Each timestep averages across batch then sums across offsets
    for t in range(seq_len):
        expected_count = (t + 1) * 0.5
        assert torch.allclose(counts[t], torch.tensor(expected_count))


def test_correct_retrieval_counts_variable_accuracy():
    """Test correct_retrieval_counts_by_timestep with variable accuracy."""
    B, seq_len = 30, 40

    # Create accuracy history with different accuracies per timestep
    accuracy_history = []
    expected_counts = []

    for t in range(seq_len):
        # Random accuracy between 0.2 and 0.8
        target_acc = 0.2 + 0.6 * torch.rand(1).item()
        acc_t = torch.ones(B, t + 1) * target_acc
        accuracy_history.append(acc_t)
        expected_counts.append((t + 1) * target_acc)

    counts = correct_retrieval_counts_by_timestep(accuracy_history)

    assert len(counts) == seq_len
    for t in range(seq_len):
        assert torch.allclose(counts[t], torch.tensor(expected_counts[t]), atol=1e-6)


def test_correct_retrieval_counts_pipeline_integration():
    """Ensure counts align with accuracies produced by compute_recall_accuracies."""
    B, seq_len, value_dim = 2, 3, 3
    values = torch.eye(value_dim).unsqueeze(0).expand(B, -1, -1)

    predictions = []
    for t in range(seq_len):
        predictions.append(values[:, : t + 1].clone())

    accuracy_history = compute_recall_accuracies(predictions, values)
    counts = correct_retrieval_counts_by_timestep(accuracy_history)

    expected = torch.tensor(
        [(t + 1) for t in range(seq_len)], dtype=counts.dtype, device=counts.device
    )
    assert torch.allclose(counts, expected)


def test_correct_retrieval_counts_empty_history_error():
    """correct_retrieval_counts_by_timestep should reject empty histories."""
    with pytest.raises(ValueError, match="accuracy_history is empty"):
        correct_retrieval_counts_by_timestep([])
