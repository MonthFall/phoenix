"""Safetensors header parser for Phoenix weight loading (V2).

Identical to V1's implementation — parses the JSON header to extract
tensor metadata without reading tensor data.
"""

import json

import torch

# Mapping from safetensors dtype strings to torch dtypes
SAFETENSORS_DTYPE_TO_TORCH: dict[str, torch.dtype] = {
    "BOOL": torch.bool,
    "U8": torch.uint8,
    "I8": torch.int8,
    "I16": torch.int16,
    "I32": torch.int32,
    "I64": torch.int64,
    "F16": torch.float16,
    "BF16": torch.bfloat16,
    "F32": torch.float32,
    "F64": torch.float64,
    "F8_E4M3": torch.float8_e4m3fn,
    "F8_E8M0": torch.float8_e8m0fnu,
}


def parse_safetensor_header(
    st_file: str,
) -> tuple[dict[str, tuple[torch.dtype, tuple[int, ...], int, int]], int]:
    """Parse safetensors file header.

    Returns:
        A tuple of (tensor_meta, header_size) where tensor_meta maps
        name -> (dtype, shape, start_offset_in_data, nbytes).
        header_size is the total header size (8 + json_len).
    """
    with open(st_file, "rb") as f:
        # First 8 bytes: little-endian uint64 = JSON header length
        header_len = int.from_bytes(f.read(8), "little")
        header_json = f.read(header_len)

    header = json.loads(header_json)

    tensor_meta: dict[str, tuple[torch.dtype, tuple[int, ...], int, int]] = {}
    for name, info in header.items():
        if name == "__metadata__":
            continue
        dtype_str = info["dtype"]
        shape = tuple(info["shape"])
        offsets = info["data_offsets"]  # [start, end] within data section
        nbytes = offsets[1] - offsets[0]

        dtype = SAFETENSORS_DTYPE_TO_TORCH.get(dtype_str)
        if dtype is None:
            raise ValueError(
                f"Unsupported safetensors dtype '{dtype_str}' for tensor "
                f"'{name}'"
            )

        tensor_meta[name] = (dtype, shape, offsets[0], nbytes)

    header_size = 8 + header_len
    return tensor_meta, header_size
