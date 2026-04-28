from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
from typing import Any


_MASK_64 = (1 << 64) - 1
_RATE_BYTES = 136
_ROTATION_OFFSETS = (
    (0, 36, 3, 41, 18),
    (1, 44, 10, 45, 2),
    (62, 6, 43, 15, 61),
    (28, 55, 25, 21, 56),
    (27, 20, 39, 8, 14),
)
_ROUND_CONSTANTS = (
    0x0000000000000001,
    0x0000000000008082,
    0x800000000000808A,
    0x8000000080008000,
    0x000000000000808B,
    0x0000000080000001,
    0x8000000080008081,
    0x8000000000008009,
    0x000000000000008A,
    0x0000000000000088,
    0x0000000080008009,
    0x000000008000000A,
    0x000000008000808B,
    0x800000000000008B,
    0x8000000000008089,
    0x8000000000008003,
    0x8000000000008002,
    0x8000000000000080,
    0x000000000000800A,
    0x800000008000000A,
    0x8000000080008081,
    0x8000000000008080,
    0x0000000080000001,
    0x8000000080008008,
)
_SOLVER_BINARY: str | None = None


@dataclass(frozen=True, slots=True)
class DeepSeekPowAnswer:
    algorithm: str
    challenge: str
    salt: str
    answer: int
    signature: str
    target_path: str

    def as_header_payload(self) -> dict[str, Any]:
        return {
            "algorithm": self.algorithm,
            "challenge": self.challenge,
            "salt": self.salt,
            "answer": self.answer,
            "signature": self.signature,
            "target_path": self.target_path,
        }


def solve_deepseek_pow(challenge: dict[str, Any]) -> DeepSeekPowAnswer:
    algorithm = str(challenge.get("algorithm") or "")
    if algorithm != "DeepSeekHashV1":
        raise ValueError(f"Unsupported DeepSeek PoW algorithm: {algorithm}")

    challenge_hex = str(challenge.get("challenge") or "")
    salt = str(challenge.get("salt") or "")
    signature = str(challenge.get("signature") or "")
    target_path = str(challenge.get("target_path") or "/api/v0/chat/completion")
    difficulty = int(challenge.get("difficulty") or 0)
    expire_at = int(challenge.get("expire_at") or challenge.get("expireAt") or 0)

    if not challenge_hex or not salt or not signature or not expire_at:
        raise ValueError("DeepSeek PoW challenge is missing required fields")
    if difficulty <= 0:
        raise ValueError("DeepSeek PoW challenge has invalid difficulty")

    answer = _solve_hash_fast(challenge_hex, salt, expire_at, difficulty)
    if answer is None:
        answer = _solve_hash(challenge_hex, salt, expire_at, difficulty)
    if answer is None:
        raise ValueError("Could not solve DeepSeek PoW challenge")

    return DeepSeekPowAnswer(
        algorithm=algorithm,
        challenge=challenge_hex,
        salt=salt,
        answer=answer,
        signature=signature,
        target_path=target_path,
    )


def _solve_hash_fast(
    challenge_hex: str,
    salt: str,
    expire_at: int,
    difficulty: int,
) -> int | None:
    solver = _get_solver_binary()
    if not solver:
        return None

    try:
        completed = subprocess.run(
            [
                solver,
                challenge_hex,
                salt,
                str(expire_at),
                str(difficulty),
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except Exception:
        return None

    if completed.returncode != 0:
        return None

    try:
        return int(completed.stdout.strip())
    except ValueError:
        return None


def _get_solver_binary() -> str | None:
    global _SOLVER_BINARY
    if _SOLVER_BINARY and Path(_SOLVER_BINARY).exists():
        return _SOLVER_BINARY

    configured = os.getenv("DEEPSEEK_POW_SOLVER_PATH", "").strip()
    if configured and Path(configured).exists():
        _SOLVER_BINARY = configured
        return _SOLVER_BINARY

    found = shutil.which("deepseek-pow-solver")
    if found:
        _SOLVER_BINARY = found
        return _SOLVER_BINARY

    source = Path(__file__).with_name("deepseek_pow_solver.c")
    compiler = shutil.which("cc") or shutil.which("gcc") or shutil.which("clang")
    if not source.exists() or not compiler:
        return None

    target = Path(tempfile.gettempdir()) / f"deepseek-pow-solver-{int(source.stat().st_mtime)}"
    if target.exists():
        _SOLVER_BINARY = str(target)
        return _SOLVER_BINARY

    try:
        subprocess.run(
            [
                compiler,
                "-O3",
                "-std=c11",
                str(source),
                "-o",
                str(target),
            ],
            check=True,
            capture_output=True,
            timeout=20,
        )
    except Exception:
        return None

    _SOLVER_BINARY = str(target)
    return _SOLVER_BINARY


def _solve_hash(
    challenge_hex: str,
    salt: str,
    expire_at: int,
    difficulty: int,
) -> int | None:
    expected = bytes.fromhex(challenge_hex)
    if len(expected) != 32:
        raise ValueError("DeepSeek PoW challenge must be a 32-byte hex digest")

    expected_lanes = tuple(
        int.from_bytes(expected[offset : offset + 8], "little")
        for offset in range(0, 32, 8)
    )
    base = bytearray(_RATE_BYTES)
    prefix = f"{salt}_{expire_at}_".encode("utf-8")
    if len(prefix) + len(str(difficulty).encode("utf-8")) >= _RATE_BYTES:
        raise ValueError("DeepSeek PoW prefix is too large for one block")

    base[: len(prefix)] = prefix
    for answer in range(difficulty):
        if _hash_answer_matches(base, len(prefix), answer, expected_lanes):
            return answer

    return None


def _hash_answer_matches(
    base_block: bytearray,
    prefix_len: int,
    answer: int,
    expected_lanes: tuple[int, int, int, int],
) -> bool:
    block = base_block.copy()
    answer_bytes = str(answer).encode("utf-8")
    answer_end = prefix_len + len(answer_bytes)
    block[prefix_len:answer_end] = answer_bytes
    block[answer_end] ^= 0x06
    block[_RATE_BYTES - 1] ^= 0x80

    state = [0] * 25
    for index in range(0, _RATE_BYTES, 8):
        lane_index = index // 8
        state[lane_index] ^= int.from_bytes(block[index : index + 8], "little")

    _keccak_f_deepseek(state)
    return (
        state[0] == expected_lanes[0]
        and state[1] == expected_lanes[1]
        and state[2] == expected_lanes[2]
        and state[3] == expected_lanes[3]
    )


def _deepseek_hash_hex(message: bytes) -> str:
    if len(message) >= _RATE_BYTES:
        raise ValueError("DeepSeek test hash helper only supports one block")

    block = bytearray(_RATE_BYTES)
    block[: len(message)] = message
    block[len(message)] ^= 0x06
    block[_RATE_BYTES - 1] ^= 0x80

    state = [0] * 25
    for index in range(0, _RATE_BYTES, 8):
        lane_index = index // 8
        state[lane_index] ^= int.from_bytes(block[index : index + 8], "little")

    _keccak_f_deepseek(state)
    return b"".join(lane.to_bytes(8, "little") for lane in state[:4]).hex()


def _rol64(value: int, offset: int) -> int:
    offset %= 64
    if offset == 0:
        return value & _MASK_64
    return ((value << offset) & _MASK_64) | (value >> (64 - offset))


def _keccak_f_deepseek(state: list[int]) -> None:
    # DeepSeek's browser worker skips round 0 of the usual Keccak-f permutation.
    for round_index in range(1, 24):
        columns = [
            state[x] ^ state[x + 5] ^ state[x + 10] ^ state[x + 15] ^ state[x + 20]
            for x in range(5)
        ]
        deltas = [
            columns[(x - 1) % 5] ^ _rol64(columns[(x + 1) % 5], 1)
            for x in range(5)
        ]
        for x in range(5):
            delta = deltas[x]
            for y in range(5):
                state[x + 5 * y] ^= delta

        rotated = [0] * 25
        for x in range(5):
            for y in range(5):
                rotated[y + 5 * ((2 * x + 3 * y) % 5)] = _rol64(
                    state[x + 5 * y],
                    _ROTATION_OFFSETS[x][y],
                )

        for x in range(5):
            for y in range(5):
                state[x + 5 * y] = (
                    rotated[x + 5 * y]
                    ^ ((~rotated[(x + 1) % 5 + 5 * y]) & rotated[(x + 2) % 5 + 5 * y])
                ) & _MASK_64

        state[0] ^= _ROUND_CONSTANTS[round_index]
