"""
Generate simple icons for the Jarvis desktop app.
This creates idle and listening state icons.
"""

from PIL import Image, ImageDraw, ImageFont


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
    try:
        # Try to use a nice font
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", size // 2)
    except:
        try:
            font = ImageFont.truetype("arial.ttf", size // 2)
        except:
            # Fallback to default
            font = ImageFont.load_default()

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

