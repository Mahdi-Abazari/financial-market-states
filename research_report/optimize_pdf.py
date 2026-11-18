"""Reduce the size of a compiled PDF while keeping its text searchable.

Requires pypdf and Pillow. Converts embedded RGB chart images to JPEG and
preserves the PDF page content, text, links, and image soft masks.
"""

from io import BytesIO
from pathlib import Path
import sys

from PIL import Image
from pypdf import PdfWriter
from pypdf.generic import NameObject


def optimize(source: Path, destination: Path, quality: int = 88) -> None:
    writer = PdfWriter(clone_from=str(source))
    seen = set()
    for page in writer.pages:
        for reference in page["/Resources"].get("/XObject", {}).values():
            obj = reference.get_object()
            if obj.get("/Subtype") != "/Image" or obj.get("/ColorSpace") != "/DeviceRGB":
                continue
            identity = id(obj)
            if identity in seen:
                continue
            seen.add(identity)
            width, height = int(obj["/Width"]), int(obj["/Height"])
            pixels = obj.get_data()
            if len(pixels) != width * height * 3:
                raise ValueError(f"Unexpected image format: {width}x{height}")
            image = Image.frombytes("RGB", (width, height), pixels)
            buffer = BytesIO()
            image.save(buffer, format="JPEG", quality=quality, optimize=True, subsampling=0)
            obj._data = buffer.getvalue()
            obj[NameObject("/Filter")] = NameObject("/DCTDecode")
            obj.pop(NameObject("/DecodeParms"), None)
    with destination.open("wb") as target:
        writer.write(target)


if __name__ == "__main__":
    if len(sys.argv) not in (3, 4):
        raise SystemExit("Usage: python optimize_pdf.py INPUT.pdf OUTPUT.pdf [JPEG_QUALITY]")
    optimize(Path(sys.argv[1]), Path(sys.argv[2]), int(sys.argv[3]) if len(sys.argv) == 4 else 88)
