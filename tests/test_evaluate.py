"""Unit tests for evaluation functions in evaluate.py"""

import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

import torch
from evaluate import (
    compute_recall_accuracies,
    average_accuracy_by_offset,
    correct_retrieval_counts_by_timestep,
)
from synthetic_datasets import generate_vectors


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
        assert torch.allclose(acc, torch.ones_like(acc), atol=1e-5)


def test_compute_recall_accuracies_with_imperfect_predictions():
    """Test compute_recall_accuracies with imperfect predictions."""
    B, seq_len, value_dim = 30, 40, 50

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
        assert abs(acc.mean().item() - chance_level) < 0.3


def test_compute_recall_accuracies_device_handling():
    """Test that compute_recall_accuracies handles device mismatch correctly."""
    if not torch.cuda.is_available():
        print("Skipping device handling test: CUDA not available.")
        return

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
    try:
        compute_recall_accuracies([], torch.randn(B, seq_len, value_dim))
        assert False, "Should have raised ValueError for empty predictions"
    except ValueError as e:
        assert "predictions list must not be empty" in str(e)

    # Test wrong values dimensions
    values = torch.randn(B, seq_len)  # Missing value_dim
    predictions = [torch.randn(B, t + 1, value_dim) for t in range(seq_len)]

    try:
        compute_recall_accuracies(predictions, values)
        assert False, "Should have raised ValueError for wrong values dimensions"
    except ValueError as e:
        assert "values tensor must be 3-dimensional" in str(e)

    # Test sequence length mismatch
    values = torch.randn(B, seq_len + 1, value_dim)  # Wrong sequence length
    try:
        compute_recall_accuracies(predictions, values)
        assert False, "Should have raised ValueError for sequence length mismatch"
    except ValueError as e:
        assert "Number of predictions must match sequence_length" in str(e)


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


def test_average_accuracy_by_offset_mixed_accuracy():
    """Test average_accuracy_by_offset with varying accuracy."""
    B, seq_len = 20, 30

    # Create mixed accuracy history
    accuracy_history = []
    for t in range(seq_len):
        # Create varying accuracy: higher for recent keys, lower for older keys
        acc_t = torch.linspace(0.3, 0.9, t + 1).expand(B, -1)
        accuracy_history.append(acc_t)

    mean_accuracy, counts = average_accuracy_by_offset(accuracy_history)

    assert len(mean_accuracy) == seq_len
    assert len(counts) == seq_len


def test_average_accuracy_by_offset_empty():
    """Test average_accuracy_by_offset with empty input."""
    try:
        average_accuracy_by_offset([])
        assert False, "Should have raised ValueError for empty input"
    except ValueError as e:
        assert "accuracy_history is empty" in str(e)


def test_average_accuracy_by_offset_preserves_device():
    """Ensure average_accuracy_by_offset keeps tensors on the source device."""
    if not torch.cuda.is_available():
        print("Skipping offset device test: CUDA not available.")
        return

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
    # Each timestep should have B * (t+1) * 0.5 correct retrievals
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
    try:
        correct_retrieval_counts_by_timestep([])
        assert False, "Should have raised ValueError for empty accuracy_history"
    except ValueError as e:
        assert "accuracy_history is empty" in str(e)


def run_all_tests():
    """Run all test functions."""
    print("=" * 60)
    print("Running evaluation function tests...")
    print("=" * 60)

    test_compute_recall_accuracies_perfect_predictions()
    print()

    test_compute_recall_accuracies_with_imperfect_predictions()
    print()

    test_compute_recall_accuracies_device_handling()
    print()

    test_compute_recall_accuracies_error_cases()
    print()

    test_average_accuracy_by_offset_perfect()
    print()

    test_average_accuracy_by_offset_mixed_accuracy()
    print()

    test_average_accuracy_by_offset_empty()
    print()

    test_average_accuracy_by_offset_preserves_device()
    print()

    test_correct_retrieval_counts_by_timestep()
    print()

    test_correct_retrieval_counts_variable_accuracy()
    print()

    test_correct_retrieval_counts_pipeline_integration()
    print()

    test_correct_retrieval_counts_empty_history_error()
    print()

    print("=" * 60)
    print("All tests passed! ✓")
    print("=" * 60)


if __name__ == "__main__":
    run_all_tests()
