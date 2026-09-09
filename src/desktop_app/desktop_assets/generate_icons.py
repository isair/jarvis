"""
Generate simple icons for the Talkie Toaster (Toustovač) desktop app.
Code-native vector drawing via PIL — idle and listening state toaster icons.
"""

from PIL import Image, ImageDraw, ImageFont

from pathlib import Path

# Deterministic, cross-platform icon font.
#
# The icon generator historically resolved the system font (Helvetica on
# macOS, Arial on Windows, PIL's bitmap default on Linux), so the same script
# produced different pixels on every platform — every local build/run
# regenerated the committed assets and git reported them as changed.
#
# DejaVu Sans is bundled in ``fonts/`` (Bitstream Vera license, see
# ``fonts/LICENSE``) and used unconditionally for the optional letter mark,
# so the output is byte-identical everywhere. ``load_default()`` is only a
# last-resort guard if the file ever goes missing.
_BUNDLED_FONT = Path(__file__).resolve().parent / "fonts" / "DejaVuSans.ttf"


def _load_font(size: int) -> "ImageFont.FreeTypeFont | ImageFont.ImageFont":
    try:
        return ImageFont.truetype(str(_BUNDLED_FONT), size)
    except OSError:
        return ImageFont.load_default()


def create_icon(color: str, filename: str, size: int = 256) -> None:
    """Create a flat vector toaster icon (transparent background)."""
    # Create image with transparency
    img = Image.new('RGBA', (size, size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    s = size / 256.0

    def R(*vals):
        return tuple(int(v * s) for v in vals)

    # Toast slices (behind the body top edge) — two golden rounds.
    toast = "#e8b96b"
    toast_edge = "#b0782f"
    draw.rounded_rectangle(R(74, 52, 110, 96), radius=int(8 * s), fill=toast, outline=toast_edge, width=max(1, int(3 * s)))
    draw.rounded_rectangle(R(146, 52, 182, 96), radius=int(8 * s), fill=toast, outline=toast_edge, width=max(1, int(3 * s)))

    # Toaster body — polished rounded rectangle in the state colour.
    draw.rounded_rectangle(R(48, 86, 208, 208), radius=int(22 * s), fill=color, outline="#26262b", width=max(1, int(4 * s)))

    # Two bread slots on the top surface.
    slot = "#14171e"
    draw.rounded_rectangle(R(74, 96, 118, 110), radius=int(6 * s), fill=slot)
    draw.rounded_rectangle(R(138, 96, 182, 110), radius=int(6 * s), fill=slot)

    # Heating glow hint between the slots.
    draw.rounded_rectangle(R(122, 98, 134, 108), radius=int(4 * s), fill="#fcd34d")

    # Front face: two dot eyes + tiny smile line (integrated, minimal).
    eye = "#0a0b0f"
    draw.ellipse(R(104, 130, 116, 142), fill=eye)
    draw.ellipse(R(140, 130, 152, 142), fill=eye)
    draw.arc(R(112, 140, 144, 158), start=0, end=180, fill=eye, width=max(1, int(3 * s)))

    # Lever on the right edge: track + knob.
    draw.line(R(200, 108, 200, 188), fill="#26262b", width=max(1, int(3 * s)))
    draw.ellipse(R(192, 138, 210, 156), fill=color, outline="#26262b", width=max(1, int(3 * s)))

    # Base plate.
    draw.rounded_rectangle(R(44, 202, 212, 216), radius=int(6 * s), fill="#26262b")

    # Save in multiple sizes for better cross-platform support
    img.save(filename)

    # Also save smaller versions
    for icon_size in [16, 32, 48, 64, 128]:
        resized = img.resize((icon_size, icon_size), Image.Resampling.LANCZOS)
        resized.save(filename.replace('.png', f'_{icon_size}.png'))

    # Create .ico file for Windows (multiple sizes in one file)
    ico_sizes = [16, 32, 48, 64, 128, 256]
    ico_images = [img.resize((sz, sz), Image.Resampling.LANCZOS) for sz in ico_sizes]
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
