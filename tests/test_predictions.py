import numpy as np
from trackviz.io.predictions import Predictions

def test_dense_basic():
    T = 5
    b = np.full((T,4), np.nan, dtype=np.float32)
    b[2] = [10,10,20,20]
    c = np.full((T,), np.nan, dtype=np.float32)
    c[2] = 0.9
    t = np.full((T,), np.nan, dtype=np.float32)
    t[2] = 3
    p = Predictions(total_frames=T, dense_bbox_xyxy=b, dense_conf=c, dense_track_id=t)
    assert len(p.for_frame(0)) == 0
    d = p.for_frame(2)
    assert len(d) == 1
    assert d[0].track_id == 3
