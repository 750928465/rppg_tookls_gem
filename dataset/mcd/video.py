"""Strict RGB frame audit and conservative recovery of truncated AVI tails."""
import hashlib
from pathlib import Path
import re
import struct
import subprocess


class DataQualityError(ValueError):
    """A record cannot be used reliably; other records may still be audited."""


def inspect_avi_tail(path):
    """Inspect RIFF chunks without loading or rewriting the video.

    Recovery requires exactly one video stream and a final incomplete video
    packet. Sentinel RIFF/LIST sizes alone are not evidence of packet damage.
    """
    size = Path(path).stat().st_size
    packets, problems = [], []
    with Path(path).open('rb') as stream:
        if stream.read(4) != b'RIFF':
            return {'supported': False, 'reason': 'not_riff_avi'}
        stream.seek(8)
        if stream.read(4) != b'AVI ':
            return {'supported': False, 'reason': 'not_riff_avi'}

        def walk(start, end, parents=(), depth=0):
            if depth > 16:
                raise DataQualityError('AVI chunk nesting is excessive.')
            offset = start
            while offset + 8 <= min(end, size):
                stream.seek(offset)
                tag, length = struct.unpack('<4sI', stream.read(8))
                stop = offset + 8 + length
                if tag in (b'RIFF', b'LIST'):
                    kind = stream.read(4)
                    if length < 4:
                        problems.append({'offset': offset, 'reason': 'short_list'})
                    else:
                        walk(offset + 12, min(stop, size), parents + (kind,), depth + 1)
                elif re.fullmatch(rb'\d\d(?:dc|db)', tag) and b'movi' in parents:
                    packets.append({'stream': tag[:2].decode(), 'offset': offset,
                                    'size': length, 'available': max(0, min(stop, size) - offset - 8),
                                    'truncated': stop > size})
                elif stop > size:
                    problems.append({'offset': offset, 'reason': 'truncated_nonvideo_chunk'})
                offset = stop + length % 2

        walk(0, size)
    truncated = [p for p in packets if p['truncated']]
    tail_only = bool(packets and len(truncated) == 1 and truncated[0] == packets[-1]
                     and len({p['stream'] for p in packets}) == 1 and not problems)
    return {'supported': True, 'video_packets': len(packets), 'tail_only': tail_only,
            'video_stream_count': len({p['stream'] for p in packets}),
            'complete_video_packets': sum(not p['truncated'] for p in packets),
            'last_packet': packets[-1] if packets else None, 'problems': problems}


def decode_hashes(path, ffmpeg='ffmpeg', strict=True, timeout=300, log_prefix=None):
    """Decode to RGB frame MD5s, retaining PTS and detecting skipped time slots.

    -vsync 0 matches the training decoder and avoids added duplicate frames.
    Container PTS may skip a slot because of its timebase; acquisition mapping
    is checked separately against AVI packet ordinals and MCD reference files.
    """
    command = [ffmpeg, '-nostdin', '-hide_banner', '-v', 'error']
    if strict:
        command.append('-xerror')
    command += ['-threads', '1', '-i', str(path), '-map', '0:v:0', '-an',
                '-vsync', '0', '-pix_fmt', 'rgb24', '-f', 'framemd5', 'pipe:1']
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired as exc:
        raise DataQualityError(f'Video audit exceeded {timeout} seconds.') from exc
    if log_prefix is not None:
        prefix = str(log_prefix)
        Path(prefix + '.log').write_text(result.stderr, encoding='utf-8')
        Path(prefix + '.framemd5').write_text(result.stdout, encoding='utf-8')
    match = re.search(r'^#dimensions 0:\s*(\d+)x(\d+)\s*$', result.stdout, re.M)
    width, height = map(int, match.groups()) if match else (0, 0)
    rows = []
    for line in result.stdout.splitlines():
        if not line or line.startswith('#'):
            continue
        parts = [s.strip() for s in line.split(',')]
        if len(parts) != 6:
            raise DataQualityError('Malformed frame checksum output.')
        rows.append(parts)
    frame_count = len(rows)
    consecutive = all(int(r[2]) == i and int(r[3]) == 1 for i, r in enumerate(rows))
    monotonic = all(int(b[2]) > int(a[2]) for a, b in zip(rows, rows[1:]))
    correct_sizes = width > 0 and height > 0 and all(int(r[4]) == width * height * 3 for r in rows)
    digest = hashlib.sha256('\n'.join(','.join(r) for r in rows).encode()).hexdigest()
    return {'returncode': result.returncode, 'frames': frame_count, 'width': width, 'height': height,
            'consecutive_pts': consecutive, 'complete_rgb_frames': correct_sizes,
            'monotonic_pts': monotonic,
            'pts_gap_after_frames': [i for i, (a, b) in enumerate(zip(rows, rows[1:]))
                                     if int(b[2]) - int(a[2]) != 1],
            'frame_sequence_sha256': digest, 'stderr': result.stderr, 'command': command,
            '_rows': rows}


def audit_video(path, ffmpeg='ffmpeg', timeout=300, log_dir=None):
    """PASS or verified RECOVERED_PREFIX; no blanket acceptance of decode errors."""
    strict = decode_hashes(path, ffmpeg, True, timeout,
                           Path(log_dir) / 'strict' if log_dir else None)
    if strict['frames'] < 2 or not strict['monotonic_pts'] or not strict['complete_rgb_frames']:
        raise DataQualityError('Decoded frames are empty, incomplete or have nonmonotonic PTS.')
    container = inspect_avi_tail(path)
    # FFmpeg may write errors even with a zero return code; keep them in review.
    if strict['returncode'] == 0 and not strict['stderr'].strip():
        if container.get('supported'):
            if (container['problems'] or container['last_packet'] is None
                    or container['video_stream_count'] != 1
                    or container['video_packets'] != strict['frames']
                    or container['complete_video_packets'] != strict['frames']):
                raise DataQualityError('AVI packet count does not establish one frame per complete video packet.')
        elif not strict['consecutive_pts']:
            raise DataQualityError('PTS gaps without AVI packet evidence require manual mapping review.')
        strict.pop('_rows')
        return {'status': 'PASS', 'strict': strict, 'recovery': None}
    if not container.get('tail_only') or container['complete_video_packets'] != strict['frames']:
        raise DataQualityError('Decode error is not verified as a final truncated AVI packet: '
                               + strict['stderr'][-2000:])
    tolerant = decode_hashes(path, ffmpeg, False, timeout,
                             Path(log_dir) / 'tolerant' if log_dir else None)
    # The tolerant decoder may conceal the final damaged packet as one extra
    # frame. Compare only the verified complete prefix and never use that tail.
    if (tolerant['returncode'] != 0 or not strict['frames'] <= tolerant['frames'] <= strict['frames'] + 1
            or tolerant['_rows'][:strict['frames']] != strict['_rows']
            or not tolerant['monotonic_pts'] or not tolerant['complete_rgb_frames']):
        raise DataQualityError('Tail recovery changed frame pixels, order or count.')
    extra_frames = tolerant['frames'] - strict['frames']
    strict.pop('_rows'); tolerant.pop('_rows')
    return {'status': 'RECOVERED_PREFIX', 'strict': strict,
            'recovery': {'method': 'verified_final_truncated_avi_packet',
                         'container': container, 'tolerant': tolerant,
                         'discarded_tolerant_tail_frames': extra_frames}}
