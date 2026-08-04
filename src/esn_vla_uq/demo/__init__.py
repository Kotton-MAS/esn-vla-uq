"""デモアニメーション (不確実性バーが失敗直前に跳ねる様子を見せる)。

`frames.py` が「何を見せるか」、`animate.py` が「どう描くか」を担う。実 LIBERO の
操作映像が入手できた時点で `frames.py` の映像パネルだけを差し替えれば済むよう
分離してある (`docs/design.md` 6.4 節)。
"""

from esn_vla_uq.demo.animate import write_demo_animation
from esn_vla_uq.demo.frames import DemoFrames, build_demo_frames

__all__ = ["DemoFrames", "build_demo_frames", "write_demo_animation"]
