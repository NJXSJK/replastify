# backend/app/utils/image_utils.py
import io
from PIL import Image, UnidentifiedImageError
from fastapi import UploadFile, HTTPException

from app.config import settings

MAX_BYTES = settings.max_file_size_mb * 1024 * 1024


async def validate_and_load_image(file: UploadFile) -> Image.Image:
    """
    Validate an uploaded file and return a clean RGB PIL Image.

    Raises HTTPException with appropriate status codes:
      400 — unsupported file type
      413 — file too large
      422 — corrupt or unreadable image
    """
    # 1. Extension check
    ext = (file.filename or "").rsplit(".", 1)[-1].lower()
    if ext not in settings.allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Unsupported file type '.{ext}'. "
                f"Allowed: {', '.join(sorted(settings.allowed_extensions))}"
            ),
        )

    # 2. Read bytes incrementally in 1 MB chunks to prevent memory exhaustion DoS
    contents = bytearray()
    chunk_size = 1024 * 1024  # 1 MB chunk
    while True:
        chunk = await file.read(chunk_size)
        if not chunk:
            break
        contents.extend(chunk)
        if len(contents) > MAX_BYTES:
            raise HTTPException(
                status_code=413,
                detail=f"File too large. Max allowed: {settings.max_file_size_mb} MB",
            )

    # 3. Image validity check
    # PIL's verify() reads to end of stream — must re-open from bytes after calling it
    try:
        image = Image.open(io.BytesIO(contents))
        image.verify()                           # raises on truncated / corrupt files
        image = Image.open(io.BytesIO(contents)) # re-open: verify() exhausts the stream
    except Exception as e:
        raise HTTPException(
            status_code=422,
            detail=f"Cannot process image: {str(e)}",
        )

    # 4. Normalize to RGB
    # Model was trained on 3-channel RGB. Handles: RGBA PNGs, grayscale, CMYK, palette
    image = image.convert("RGB")

    return image
