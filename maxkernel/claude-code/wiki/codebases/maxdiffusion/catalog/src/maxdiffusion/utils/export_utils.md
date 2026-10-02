---
title: 'Module: src/maxdiffusion/utils/export_utils.py'
type: catalog
provenance: extracted
module: src/maxdiffusion/utils/export_utils.py
status: fresh
symbol_base: scip-python python maxdiffusion 0.0.0 `src.maxdiffusion.utils.export_utils`/
symbols:
  export_to_video: export_to_video().
  export_to_video_with_audio: export_to_video_with_audio().
  _legacy_export_to_video: _legacy_export_to_video().
  logger: logger.
  export_to_ply: export_to_ply().
  _write_audio: _write_audio().
  buffered_writer: buffered_writer().
  export_to_gif: export_to_gif().
  export_to_obj: export_to_obj().
  _prepare_audio_stream: _prepare_audio_stream().
  _resample_audio: _resample_audio().
  global_rng: global_rng.
---
# Module: [`src/maxdiffusion/utils/export_utils.py`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/utils/export_utils.py)

## Functions
- `_legacy_export_to_video(video_frames: Union[List[np.ndarray], List[PIL.Image.Image]], output_video_path: str = None, fps: int = 10)` — [`L133`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/utils/export_utils.py#L133)
- `_prepare_audio_stream(container, audio_sample_rate: int)` — [`L234`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/utils/export_utils.py#L234) — Prepare the audio stream for writing.
- `_resample_audio(container, audio_stream, frame_in)` — [`L247`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/utils/export_utils.py#L247)
- `_write_audio(container, audio_stream, samples: Any, audio_sample_rate: int, target_format: str = "s16")` — [`L273`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/utils/export_utils.py#L273)
- `buffered_writer(raw_f)` — [`L42`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/utils/export_utils.py#L42)
- `export_to_gif(image: List[PIL.Image.Image], output_gif_path: str = None)` — [`L48`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/utils/export_utils.py#L48)
- `export_to_obj(mesh, output_obj_path: str = None)` — [`L115`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/utils/export_utils.py#L115)
- `export_to_ply(mesh, output_ply_path: str = None)` — [`L63`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/utils/export_utils.py#L63) — Write a PLY file for a mesh.
- `export_to_video(video_frames: Union[np.ndarray, List[np.ndarray], List[PIL.Image.Image]], output_video_path: str = None, fps: int = 10, quality: float = 5, bitrate: Optional[int] = None, macro_block_size: Optional[int] = 16)` — [`L159`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/utils/export_utils.py#L159) — quality:
- `export_to_video_with_audio(video: Any, fps: int, audio: Optional[Any], audio_sample_rate: Optional[int], output_path: str, audio_format: str = "s16")` — [`L322`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/utils/export_utils.py#L322) — Encodes video (and optionally audio) to a file using PyAV.

## Module values
- `global_rng` — [`L36`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/utils/export_utils.py#L36)
- `logger` — [`L38`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/utils/export_utils.py#L38)

