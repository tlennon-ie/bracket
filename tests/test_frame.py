import pytest

from bracket.frame import LossFrame, bucket_timestep


def test_loss_frame_json_roundtrip_required_fields():
    f = LossFrame(
        step=42,
        raw_loss=0.31,
        smoothed_loss=0.305,
        lr=1e-6,
        wall_time=1234567.0,
        timestep=None,
        timestep_bucket=None,
        grad_norm=None,
    )
    j = f.to_json()
    assert j["step"] == 42
    assert j["raw_loss"] == 0.31
    assert j["smoothed_loss"] == 0.305
    assert j["timestep"] is None
    assert j["grad_norm"] is None


def test_bucket_timestep_endpoints():
    assert bucket_timestep(0, num_buckets=10) == 0
    assert bucket_timestep(999, num_buckets=10) == 9
    assert bucket_timestep(500, num_buckets=10) == 5


def test_bucket_timestep_out_of_range_raises():
    with pytest.raises(ValueError):
        bucket_timestep(-1)
    with pytest.raises(ValueError):
        bucket_timestep(1000)


def test_bucket_timestep_uneven_buckets_clamped():
    # Bucket count not evenly dividing max — last value still maps to last bucket
    for ts in (998, 999):
        assert bucket_timestep(ts, num_buckets=7) == 6
