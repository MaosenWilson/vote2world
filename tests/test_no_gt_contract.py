def test_adaptation_batch_contract_placeholder():
    """Placeholder for the future no-GT leakage test."""
    forbidden_keys = {"ground_truth", "future_frame", "next_frame_gt", "o_t_plus_1"}
    adaptation_batch_keys = {"observation_history", "action"}

    assert forbidden_keys.isdisjoint(adaptation_batch_keys)
