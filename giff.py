from PIL import Image

# Load the two images
img1 = Image.open(r"F:\U1PAGE1.png")
img2 = Image.open(r"F:\UI1PAGE.png")

# Ensure both images are the same size
img2 = img2.resize(img1.size)

# Create frame sequence for smooth loop (forward and reverse)
frames = [img1, img2, img1]

# Save as GIF
frames[0].save(
    r"F:\gifs\1smooth_loop.gif",
    save_all=True,
    append_images=frames[1:],
    duration=500,  # duration in ms
    loop=0
)

from PIL import Image

# Load images with RGBA mode for blending
img1 = Image.open(r"F:\ui_back4.png").convert('RGBA')
img2 = Image.open(r"F:\ui_5.png").convert('RGBA')

# Resize second image to match the first
img2 = img2.resize(img1.size)

# Number of intermediate blending frames
num_transitions = 10
frames = []

# Generate blended frames from img1 to img2
for i in range(num_transitions + 1):
    alpha = i / num_transitions
    blended = Image.blend(img1, img2, alpha)
    frames.append(blended)

# Add reverse blended frames for smooth looping back to img1
for i in range(num_transitions - 1, -1, -1):
    alpha = i / num_transitions
    blended = Image.blend(img1, img2, alpha)
    frames.append(blended)

# Save as GIF with smooth transition and looping
frames[0].save(
    r"F:\gifs\7smooth_loop.gif",
    save_all=True,
    append_images=frames[1:],
    duration=500,
    loop=0
)

