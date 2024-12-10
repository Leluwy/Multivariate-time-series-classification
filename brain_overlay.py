from PIL import Image, ImageDraw

brain_image = Image.open("/Users/leluwy/Downloads/brain_image.jpg")

# Create a blank image
overlay = Image.new("RGBA", brain_image.size)

# the first two entries are the coordinates of the corresponding electrode, the other two entries are the importance
# values for the wavelet features- and power features, respectively
electrode_relevance = {
    "FP1": [450, 150, 5, 5],
    "FP2": [700, 150, 5, 5],
    "F3": [400, 400, 4.27, 4.43],
    "F4": [750, 400, 4.26, 4.46],
    "FZ": [575, 395, 4.08, 4.91],
    "F7": [250, 320, 4.11, 4.26],
    "F8": [900, 320, 4.33, 4.41],
    "CZ": [575, 680, 4.05, 4.19],
    "C3": [325, 680, 3.90, 4.11],
    "C4": [825, 680, 3.58, 3.97],
    "PZ": [575, 950, 3.75, 4.07],
    "P3": [375, 950, 3.63, 3.95],
    "P4": [775, 950, 3.67, 4.05],
    "O1": [400, 1200, 3.80, 4.07],
    "O2": [750, 1200, 3.74, 4.03],
    "A1": [75, 600, 4.44, 4.37],
    "A2": [1075, 600, 4.58, 4.03],
    "T3": [175, 750, 4.52, 4.46],
    "T4": [975, 750, 4.07, 4.15],
    "T5": [200, 1000, 4.2, 4.1],
    "T6": [950, 1000, 3.71, 4.1],

}

# Define the color
electrode_color = (255, 0, 0, 128)  # red color with transparency
electrode_size = 170  # define the base size of each electrode

# Draw circles on the overlay image for each electrode
draw = ImageDraw.Draw(overlay)
for electrode, data in electrode_relevance.items():

    relevance = data[3]-3.5  # adapt the relevance levels

    electrode_x = data[0]
    electrode_y = data[1]

    electrode_radius = relevance * electrode_size  # the radius of the electrode

    alpha = int(relevance/1.5 * 255)  # how bright the electrode will be

    draw.ellipse(
        [
            (electrode_x - electrode_radius, electrode_y - electrode_radius),
            (electrode_x + electrode_radius, electrode_y + electrode_radius),
        ],
        fill=(alpha, 0, 0, 128)
    )


final_image = Image.alpha_composite(brain_image.convert("RGBA"), overlay)

# Save
final_image.save("/Users/leluwy/Downloads/brain_image_with_overlay.png")
