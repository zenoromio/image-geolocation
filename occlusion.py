from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import torch
import numpy as np
import matplotlib.pyplot as plt


model = CLIPModel.from_pretrained("geolocal/StreetCLIP", attn_implementation="eager")
processor = CLIPProcessor.from_pretrained("geolocal/StreetCLIP")

# choose and resize image
#image = Image.open("PATH_TO_IMAGE")
image = Image.open("images/Chile/Chile_0b32c6d6.jpg")
image_336 = image.resize((336, 336))

# all 193 available countries
choices = [
    "Afghanistan", "Albania", "Algeria", "Andorra", "Angola", "Antigua and Barbuda", "Argentina", "Armenia", "Aruba",
    "Australia", "Austria", "Azerbaijan", "Bahamas", "Bahrain", "Bangladesh", "Barbados", "Belarus", "Belgium",
    "Belize", "Benin", "Bhutan", "Bolivia", "Bosnia and Herzegovina", "Botswana", "Brazil", "Brunei", "Bulgaria",
    "Burkina Faso", "Burundi", "Cabo Verde", "Cambodia", "Cameroon", "Canada", "Central African Republic", "Chad",
    "Chile", "China", "Colombia", "Comoros", "Costa Rica", "Croatia", "Cuba", "Cyprus", "Czech Republic",
    "Côte d'Ivoire", "Democratic Republic of the Congo", "Denmark", "Djibouti", "Dominica", "Dominican Republic",
    "East Timor", "Ecuador", "Egypt", "El Salvador", "Equatorial Guinea", "Eritrea", "Estonia", "Ethiopia", "Fiji",
    "Finland", "France", "Gabon", "Gambia", "Georgia", "Germany", "Ghana", "Greece", "Greenland", "Grenada",
    "Guatemala", "Guinea", "Guinea-Bissau", "Guyana", "Haiti", "Honduras", "Hungary", "Iceland", "India",
    "Indonesia", "Iran", "Iraq", "Ireland", "Israel", "Italy", "Jamaica", "Japan", "Jordan", "Kazakhstan",
    "Kenya", "Kuwait", "Kyrgyzstan", "Laos", "Latvia", "Lebanon", "Lesotho", "Liberia", "Libya", "Liechtenstein",
    "Lithuania", "Luxembourg", "Madagascar", "Malawi", "Malaysia", "Mali", "Malta", "Marshall Islands", "Mauritania",
    "Mauritius", "Mexico", "Micronesia", "Moldova", "Mongolia", "Montenegro", "Morocco", "Mozambique", "Myanmar",
    "Namibia", "Nauru", "Nepal", "Netherlands", "New Zealand", "Nicaragua", "Niger", "Nigeria", "North Korea",
    "North Macedonia", "Norway", "Oman", "Pakistan", "Palau", "Palestine", "Panama", "Papua New Guinea", "Paraguay",
    "Peru", "Philippines", "Poland", "Portugal", "Qatar", "Republic of the Congo", "Romania", "Russia", "Rwanda",
    "Saint Kitts and Nevis", "Saint Lucia", "Saint Vincent and the Grenadines", "Samoa", "San Marino", "Saudi Arabia",
    "Senegal", "Serbia", "Seychelles", "Sierra Leone", "Singapore", "Slovakia", "Slovenia", "Solomon Islands",
    "Somalia", "South Africa", "South Korea", "South Sudan", "Spain", "Sri Lanka", "Sudan", "Suriname", "Eswatini",
    "Sweden", "Switzerland", "Syria", "Taiwan", "Tajikistan", "Tanzania", "Thailand", "Togo", "Tonga",
    "Trinidad and Tobago", "Tunisia", "Turkey", "Turkmenistan", "Tuvalu", "Uganda", "Ukraine", "United Arab Emirates",
    "United Kingdom", "Uruguay", "Uzbekistan", "Vanuatu", "Venezuela", "Vietnam", "Yemen", "Zambia", "Zimbabwe",
    "United States"
]

# foward pass to get the score
def get_score(img_pil):
    inputs = processor(text=choices, images=img_pil, return_tensors="pt", padding=True)
    with torch.no_grad():
        logits = model(**inputs).logits_per_image
    return logits.softmax(dim=1)[0]


baseline_probs = get_score(image_336)

# print top prediction
ranked = sorted(zip(choices, baseline_probs.tolist()), key=lambda x: x[1], reverse=True)
print("Top 5 predictions:")
for choice, prob in ranked[:5]:
    print(f"  {choice}: {prob:.4f}")

# begin occlusion process
target_idx = baseline_probs.argmax().item()
baseline_score = baseline_probs[target_idx].item()
print(f"\nTracking: '{choices[target_idx]}' (score: {baseline_score:.4f})")

def occlusion_heatmap(image_pil, target_idx, grid_size=7, mask_color=(128, 128, 128)):
    W, H = image_pil.size
    patch_w = W // grid_size
    patch_h = H // grid_size
    importance = np.zeros((grid_size, grid_size))
    baseline = get_score(image_pil)[target_idx].item()

    total = grid_size * grid_size
    for i in range(grid_size):
        for j in range(grid_size):
            img_copy = image_pil.copy()
            pixels = img_copy.load()
            for y in range(i * patch_h, (i + 1) * patch_h):
                for x in range(j * patch_w, (j + 1) * patch_w):
                    pixels[x, y] = mask_color

            masked_score = get_score(img_copy)[target_idx].item()
            importance[i, j] = baseline - masked_score

            done = i * grid_size + j + 1
            print(f"  [{done}/{total}] patch ({i},{j}) → drop: {importance[i,j]:+.4f}")

    return importance

print(f"\nRunning {7*7} occlusion passes...")
heatmap = occlusion_heatmap(image_336, target_idx, grid_size=7)

fig, ax = plt.subplots(figsize=(6, 6))
ax.imshow(image_336)

# CREATE IMPORTANCE MAP
grid_size = 7
patch_w = patch_h = 336 // grid_size

for i in range(grid_size):
    for j in range(grid_size):
        val = heatmap[i, j]
        color = 'red' if val > 0 else 'blue'
        rect = plt.Rectangle((j*patch_w, i*patch_h), patch_w, patch_h,
                              linewidth=1, edgecolor=color,
                              facecolor=color, alpha=min(abs(val)*3, 0.7))
        ax.add_patch(rect)
        ax.text(j*patch_w + patch_w//2, i*patch_h + patch_h//2,
                f"{val:+.2f}", color='white', fontsize=6,
                ha='center', va='center', fontweight='bold')

ax.axis("off")
plt.tight_layout()
plt.show()
