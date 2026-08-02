"""
Generate simple icons for the Jarvis desktop app.
This creates idle and listening state icons.
"""

from PIL import Image, ImageDraw, ImageFont

from pathlib import Path

# Deterministic, cross-platform icon font.
#
# The icon generator used to resolve the system font (Helvetica on macOS,
# Arial on Windows, PIL's bitmap default on Linux), so the same script
# produced different pixels on every platform — every local build/run
# regenerated the committed assets and git reported them as changed.
#
# DejaVu Sans is bundled in ``fonts/`` (Bitstream Vera license, see
# ``fonts/LICENSE``) and used unconditionally, so the output is byte-
# identical everywhere. ``load_default()`` is only a last-resort guard if
# the file ever goes missing — it must never be the primary path.
_BUNDLED_FONT = Path(__file__).resolve().parent / "fonts" / "DejaVuSans.ttf"


def _load_font(size: int) -> "ImageFont.FreeTypeFont | ImageFont.ImageFont":
    try:
        return ImageFont.truetype(str(_BUNDLED_FONT), size)
    except OSError:
        return ImageFont.load_default()


def create_icon(color: str, filename: str, size: int = 256) -> None:
    """Create a simple circular icon with a 'J' letter."""
    # Create image with transparency
    img = Image.new('RGBA', (size, size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    # Draw circle
    margin = size // 8
    draw.ellipse(
        [(margin, margin), (size - margin, size - margin)],
        fill=color,
        outline=None
    )

    # Draw letter J
    font = _load_font(size // 2)

    text = "J"
    # Get text bounding box
    bbox = draw.textbbox((0, 0), text, font=font)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]

    # Center the text
    x = (size - text_width) // 2 - bbox[0]
    y = (size - text_height) // 2 - bbox[1]

    draw.text((x, y), text, fill='white', font=font)

    # Save in multiple sizes for better cross-platform support
    img.save(filename)

    # Also save smaller versions
    for icon_size in [16, 32, 48, 64, 128]:
        resized = img.resize((icon_size, icon_size), Image.Resampling.LANCZOS)
        resized.save(filename.replace('.png', f'_{icon_size}.png'))

    # Create .ico file for Windows (multiple sizes in one file)
    ico_sizes = [16, 32, 48, 64, 128, 256]
    ico_images = [img.resize((s, s), Image.Resampling.LANCZOS) for s in ico_sizes]
    ico_filename = filename.replace('.png', '.ico')
    # Save ICO with multiple sizes - PIL handles multi-size ICO via append_images
    ico_images[-1].save(
        ico_filename,
        format='ICO',
        append_images=ico_images[:-1]
    )


if __name__ == '__main__':
    import os
    import sys
    from pathlib import Path

    # Fix Windows console encoding for emojis
    if sys.platform == 'win32':
        try:
            # Try to set UTF-8 encoding for Windows console
            import io
            sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
        except Exception:
            pass

    # Get the directory where this script is located
    script_dir = Path(__file__).parent

    # Create idle icon (gray)
    create_icon('#9E9E9E', str(script_dir / 'icon_idle.png'))
    print("Created icon_idle.png")

    # Create listening icon (green)
    create_icon('#4CAF50', str(script_dir / 'icon_listening.png'))
    print("Created icon_listening.png")

    print("\nIcon generation complete!")

