"""Pure NumPy LSB steganography embedding and extraction.

The public functions in this module operate on an in-memory H x W x 3 RGB
``uint8`` array.  They do not read or write files.  Carrier units are the
interleaved row-major R, G, B values, so a pixel mask is derived only after
unit selection.  Validation, including capacity and legal placement checks,
happens before random draws.  For a successful call, the generator is consumed
in this order: payload bytes or letters, random placement selection, trailing
field padding, and match direction choices.
"""

from dataclasses import dataclass, field
import math

import numpy as np


_LETTERS = np.frombuffer(
    b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz", dtype=np.uint8
)


@dataclass(frozen=True)
class EmbedResult:
    """Metadata describing one embedding operation.

    ``embedded_bits`` includes random padding bits in the final selected unit.
    ``bpp_actual`` describes the requested payload bytes (not that padding).
    The private selected-unit cache is needed to extract a random placement;
    contiguous and strided placements are reconstructed from the public
    placement metadata.
    """

    bit_mode: str
    k: int
    method: str
    placement: str
    start: int | None
    stride: int | None
    payload_kind: str
    payload_bytes: int
    embedded_bits: int
    units_used: int
    units_changed: int
    bpp_actual: float
    _selected_units: np.ndarray | None = field(
        default=None, repr=False, compare=False
    )

    @property
    def units(self) -> np.ndarray | None:
        """Return the selected flat carrier units for random placement.

        For contiguous and strided placement this property is ``None`` because
        the unit indices are fully described by ``start`` and ``stride``.
        """
        return self._selected_units


def _validate_carrier(carrier: np.ndarray) -> tuple[np.ndarray, int, int, int]:
    """Validate an RGB carrier and return it with its dimensions and unit count."""
    if not isinstance(carrier, np.ndarray):
        raise TypeError("carrier must be a numpy array")
    if carrier.dtype != np.uint8 or carrier.ndim != 3 or carrier.shape[2] != 3:
        raise ValueError("carrier must be an H x W x 3 uint8 array")
    height, width = carrier.shape[:2]
    if height < 1 or width < 1:
        raise ValueError("carrier dimensions must be positive")
    return carrier, int(height), int(width), int(carrier.size)


def _validate_generator(rng: np.random.Generator) -> None:
    """Require the caller's NumPy random generator."""
    if not isinstance(rng, np.random.Generator):
        raise TypeError("rng must be a numpy.random.Generator")


def _validate_integer(value: int, name: str, minimum: int | None = None) -> int:
    """Validate an integer parameter and return a native Python integer."""
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name} must be an integer")
    value = int(value)
    if minimum is not None and value < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return value


def _validate_placement(
    total_units: int,
    units_used: int,
    placement: str,
    start: int | None,
    stride: int,
) -> None:
    """Validate placement and all explicit bounds without consuming the RNG."""
    if placement not in {"contiguous", "stride", "random"}:
        raise ValueError("placement must be 'contiguous', 'stride', or 'random'")
    if placement == "random":
        return
    actual_stride = _validate_integer(stride, "stride", 1) if placement == "stride" else 1
    maximum_start = total_units - 1 - (units_used - 1) * actual_stride
    if maximum_start < 0:
        raise ValueError("payload does not fit in the carrier")
    if start is not None:
        actual_start = _validate_integer(start, "start", 0)
        if actual_start > maximum_start:
            raise ValueError("start is outside the legal non-wrapping range")


def _select_units(
    total_units: int,
    units_used: int,
    placement: str,
    start: int | None,
    stride: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, int | None, int | None]:
    """Select sorted, non-wrapping carrier units without a Python unit loop."""
    _validate_placement(total_units, units_used, placement, start, stride)
    if placement == "random":
        # Generator.choice samples a uniform subset without replacement.  The
        # sort is over the selected n units rather than all C units.
        selected = np.asarray(
            rng.choice(total_units, size=units_used, replace=False), dtype=np.intp
        )
        selected.sort()
        selected.setflags(write=False)
        return selected, None, None

    actual_stride = _validate_integer(stride, "stride", 1) if placement == "stride" else 1
    maximum_start = total_units - 1 - (units_used - 1) * actual_stride
    if start is None:
        actual_start = int(rng.integers(0, maximum_start + 1))
    else:
        actual_start = _validate_integer(start, "start", 0)

    selected = actual_start + np.arange(units_used, dtype=np.intp) * actual_stride
    return selected, actual_start, actual_stride if placement == "stride" else None


def _make_payload(
    payload_kind: str,
    payload_bytes: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Create payload bytes and their MSB-first bit sequence."""
    if payload_kind == "random_bytes":
        data = rng.integers(0, 256, size=payload_bytes, dtype=np.uint8)
    elif payload_kind == "letters":
        indices = rng.integers(0, _LETTERS.size, size=payload_bytes)
        data = _LETTERS[indices]
    else:
        raise ValueError("payload must be 'random_bytes' or 'letters'")
    return data, np.unpackbits(data, bitorder="big")


def _target_fields(
    payload_bits: np.ndarray,
    bits_per_unit: int,
    units_used: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Group payload bits into fields, adding random trailing field bits."""
    embedded_bits = units_used * bits_per_unit
    if payload_bits.size < embedded_bits:
        padding = rng.integers(
            0, 2, size=embedded_bits - payload_bits.size, dtype=np.uint8
        )
        bits = np.concatenate((payload_bits, padding))
    else:
        bits = payload_bits
    if bits_per_unit == 8:
        fields = np.packbits(bits, bitorder="big").astype(np.uint8, copy=True)
    else:
        shifts = np.arange(bits_per_unit - 1, -1, -1, dtype=np.uint16)
        fields = np.sum(
            bits.reshape(units_used, bits_per_unit).astype(np.uint16) << shifts,
            axis=1,
            dtype=np.uint16,
        ).astype(np.uint8)
    return fields, bits


def _replace_values(
    values: np.ndarray,
    fields: np.ndarray,
    bit_mode: str,
    k: int,
) -> np.ndarray:
    """Return values with target bits replaced, preserving all other bits."""
    if bit_mode == "lowest":
        field_mask = (1 << k) - 1
        return (
            (values.astype(np.uint16) & (~field_mask & 0xFF))
            | fields.astype(np.uint16)
        ).astype(np.uint8)
    bit_mask = 1 << (k - 1)
    return (
        (values.astype(np.uint16) & (~bit_mask & 0xFF))
        | (fields.astype(np.uint16) * bit_mask)
    ).astype(np.uint8)


def _match_values(
    values: np.ndarray,
    fields: np.ndarray,
    bit_mode: str,
    k: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Return nearest-valid LSB-matched values using vectorized choices.

    In plane mode the deliberate rule is ``v +/- 2**(k - 1)``.  This keeps
    every lower bit plane unchanged; it is not a search for an arbitrary value
    that happens to have the target bit.
    """
    values_i = values.astype(np.int16)
    if bit_mode == "lowest":
        field_mask = (1 << k) - 1
        target = fields.astype(np.int16)
        if k == 8:
            # Every byte has exactly one valid value for a target whole byte.
            return target.astype(np.uint8)
        step = 1 << k
        quotient = np.floor_divide(values_i - target, step)
        below = target + quotient * step
        above = below + step
    else:
        bit_mask = 1 << (k - 1)
        step = bit_mask
        target = fields.astype(np.int16) * bit_mask
        below = values_i - step
        above = values_i + step

    target_matches = (
        (values_i & ((1 << k) - 1)) == fields.astype(np.int16)
        if bit_mode == "lowest"
        else ((values_i & (1 << (k - 1))) == target)
    )
    result = values_i.copy()
    mismatch = ~target_matches
    if np.any(mismatch):
        below_values = below[mismatch]
        above_values = above[mismatch]
        below_valid = below_values >= 0
        above_valid = above_values <= 255
        choose_above = rng.integers(0, 2, size=below_values.size).astype(bool)
        choose_above = np.where(above_valid & ~below_valid, True, choose_above)
        choose_above = np.where(~above_valid & below_valid, False, choose_above)
        result[mismatch] = np.where(choose_above, above_values, below_values)
    return result.astype(np.uint8)


def embed(
    carrier: np.ndarray,
    bit_mode: str,
    k: int,
    method: str,
    placement: str,
    rng: np.random.Generator,
    *,
    start: int | None = None,
    stride: int = 1,
    payload: str = "random_bytes",
    payload_bytes: int | None = None,
    bpp: float | None = None,
) -> tuple[np.ndarray, np.ndarray, EmbedResult]:
    """Embed one generated payload into an RGB carrier.

    Exactly one of ``payload_bytes`` and ``bpp`` must be supplied.  Payload
    bits are MSB-first
    and a partial final field receives random padding bits.  In lowest-bit mode
    with ``k=8``, match is exactly replace because each target whole byte has
    only one valid carrier value.
    """
    carrier, height, width, total_units = _validate_carrier(carrier)
    _validate_generator(rng)
    if bit_mode not in {"lowest", "plane"}:
        raise ValueError("bit_mode must be 'lowest' or 'plane'")
    k = _validate_integer(k, "k", 1)
    if k > 8:
        raise ValueError("k must be between 1 and 8")
    if method not in {"replace", "match"}:
        raise ValueError("method must be 'replace' or 'match'")
    if payload not in {"random_bytes", "letters"}:
        raise ValueError("payload must be 'random_bytes' or 'letters'")

    if (payload_bytes is None) == (bpp is None):
        raise ValueError("provide exactly one of payload_bytes or bpp")
    if payload_bytes is not None:
        payload_count = _validate_integer(payload_bytes, "payload_bytes", 1)
    else:
        if isinstance(bpp, (bool, np.bool_)) or not isinstance(
            bpp, (int, float, np.integer, np.floating)
        ):
            raise TypeError("bpp must be a positive finite number")
        bpp_value = float(bpp)
        if not math.isfinite(bpp_value) or bpp_value <= 0:
            raise ValueError("bpp must be a positive finite number")
        payload_count = max(1, int(round(bpp_value * height * width / 8.0)))

    bits_per_unit = k if bit_mode == "lowest" else 1
    payload_bit_count = payload_count * 8
    capacity_bits = total_units * bits_per_unit
    if payload_bit_count > capacity_bits:
        raise ValueError("payload exceeds carrier capacity")
    units_used = (payload_bit_count + bits_per_unit - 1) // bits_per_unit
    _validate_placement(total_units, units_used, placement, start, stride)

    payload_data, payload_bits = _make_payload(payload, payload_count, rng)
    selected, actual_start, actual_stride = _select_units(
        total_units, units_used, placement, start, stride, rng
    )
    fields, _ = _target_fields(payload_bits, bits_per_unit, units_used, rng)

    stego = carrier.copy()
    flat = stego.reshape(-1)
    before = flat[selected].copy()
    if method == "replace":
        after = _replace_values(before, fields, bit_mode, k)
    else:
        after = _match_values(before, fields, bit_mode, k, rng)
    flat[selected] = after

    mask = np.zeros((height, width), dtype=np.bool_)
    mask.reshape(-1)[selected // 3] = True
    changed = int(np.count_nonzero(before != after))
    selected_cache = selected if placement == "random" else None
    result = EmbedResult(
        bit_mode=bit_mode,
        k=k,
        method=method,
        placement=placement,
        start=actual_start,
        stride=actual_stride,
        payload_kind=payload,
        payload_bytes=payload_count,
        embedded_bits=units_used * bits_per_unit,
        units_used=units_used,
        units_changed=changed,
        bpp_actual=payload_bit_count / float(height * width),
        _selected_units=selected_cache,
    )
    return stego, mask, result


def _units_for_result(
    result: EmbedResult,
    total_units: int,
    units: np.ndarray | None,
) -> np.ndarray:
    """Resolve and validate units used by an embedding result."""
    if not isinstance(result, EmbedResult):
        raise TypeError("result must be an EmbedResult")
    if units is not None:
        if not isinstance(units, np.ndarray) or units.ndim != 1:
            raise ValueError("units must be a one-dimensional numpy array")
        selected = np.asarray(units, dtype=np.intp)
    elif result.placement == "random":
        if result._selected_units is None:
            raise ValueError("random extraction needs the selected units")
        selected = result._selected_units
    else:
        if result.start is None:
            raise ValueError("non-random result has no start")
        actual_stride = result.stride if result.placement == "stride" else 1
        selected = result.start + np.arange(result.units_used, dtype=np.intp) * actual_stride
    if selected.size != result.units_used:
        raise ValueError("units count does not match result")
    if selected.size and (np.any(selected < 0) or np.any(selected >= total_units)):
        raise ValueError("units contain an out-of-range index")
    return selected


def extract(
    carrier: np.ndarray,
    result: EmbedResult,
    *,
    units: np.ndarray | None = None,
) -> bytes:
    """Extract the original payload bytes from a carrier and embed result.

    For random placement, the result retains the selected sorted unit indices;
    callers may instead pass those indices explicitly with ``units``.  The
    result's padding bits are ignored, so the returned value has exactly
    ``result.payload_bytes`` bytes.
    """
    carrier, _, _, total_units = _validate_carrier(carrier)
    selected = _units_for_result(result, total_units, units)
    values = carrier.reshape(-1)[selected]
    if result.bit_mode == "lowest":
        field_mask = (1 << result.k) - 1
        fields = values & field_mask
        if result.k == 8:
            bits = np.unpackbits(fields, bitorder="big")
        else:
            shifts = np.arange(result.k - 1, -1, -1, dtype=np.uint16)
            bits = ((fields.astype(np.uint16)[:, None] >> shifts) & 1).astype(
                np.uint8
            ).reshape(-1)
    elif result.bit_mode == "plane":
        bit_mask = 1 << (result.k - 1)
        bits = ((values & bit_mask) != 0).astype(np.uint8)
    else:
        raise ValueError("result has an unsupported bit mode")
    payload_bit_count = result.payload_bytes * 8
    return np.packbits(bits[:payload_bit_count], bitorder="big").tobytes()


__all__ = ["EmbedResult", "embed", "extract"]
