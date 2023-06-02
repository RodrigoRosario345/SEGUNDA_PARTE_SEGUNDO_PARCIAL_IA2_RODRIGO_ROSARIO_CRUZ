import numpy as np
import time
import matplotlib.pyplot as plt
import cv2
import sys
import io
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
import torch
import json
import cv2
import numpy as np
# Parámetros
model_size = "medium"  # small, medium, large
device = torch.device("cpu")  # cuda:0, cpu
image_name = "fotos papeles basura ia2 Rodrigo Rosario/papel (1).jpg"  # Nombre de la imagen 
image_name =  "image_examples/prueba.jpg"

# Cargar la imagen
image = cv2.imread(image_name)

# Aplicar ajuste de brillo y contraste
alpha = 1.5  # Factor de brillo (ajustable)
beta = 50  # Factor de contraste (ajustable)
enhanced_image = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)

# Aplicar filtro de mejora de nitidez
sharpened_image = cv2.filter2D(enhanced_image, -1, kernel=np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]]))

# Mostrar la imagen original y mejorada
# cv2.imshow("Imagen original", image)
# cv2.imshow("Imagen mejorada", sharpened_image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

image_name = sharpened_image



if torch.cuda.is_available():
    current_device = torch.cuda.current_device()
    print(f"El código se está ejecutando en la GPU: {torch.cuda.get_device_name(current_device)}")
else:
    print("El código se está ejecutando en la CPU.")

# Cargar la imagen
image = image_name
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Métodos de visualización
def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.array([0/255, 255/255, 0/255, 0.6])
    else:
        color = np.array([0/255, 255/255, 0/255, 0.6])  # Verde por defecto
    
    # Aumentar la intensidad del color multiplicando por un factor
    intensity_factor = 1.5
    color *= intensity_factor
    
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)
    
def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels == 1]
    neg_points = coords[labels == 0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)   
    
def show_box(box, ax):
    x0, y0, w, h = box
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0, 0, 0, 0), lw=2))  

# Ruta del archivo JSON
json_path = 'papelesLabels/LabelOriginal.json'

# Cargar el archivo JSON
with open(json_path) as f:
    data = json.load(f)

# Obtener los bounding boxes del JSON
bboxes = torch.empty((0, 4), dtype=torch.float32)
for obj in data[0]['annotation']['object']:
    bbox = obj['bndbox']
    x, y, w, h = int(bbox['left']), int(bbox['top']), int(bbox['width']), int(bbox['height'])
    bboxes = torch.cat((bboxes, torch.tensor([[x, y, w, h]])), dim=0)
bboxes = bboxes.to(device=device)



# get middle point of image
input_boxes = torch.tensor([
    [15, 400, 269, 525],
    [300, 40, 435, 395],
], device=device)

sys.path.append("..")

# Descargar el modelo checkpoint (https://github.com/facebookresearch/segment-anything#model-checkpoints)
if model_size == "small":
    sam_checkpoint = "model_checkpoint/sam_vit_b_01ec64.pth"
    model_type = "vit_b"

if model_size == "medium":
    sam_checkpoint = "model_checkpoint/sam_vit_l_0b3195.pth"
    model_type = "vit_l"

if model_size == "large":
    sam_checkpoint = "model_checkpoint/sam_vit_h_4b8939.pth"
    model_type = "vit_h"

# Cargar el modelo SAM
sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device=device)

# Crear el predictor SAM
predictor = SamPredictor(sam)
predictor.set_image(image)

# Obtener el tiempo de inicio
start = time.time()

# Ampliar los bounding boxes desde el centro
centers = bboxes[:, :2] + bboxes[:, 2:] / 2
delta = (bboxes[:, 2:] / 2) * 1.2  # Factor de expansión (ajustable)
expanded_bboxes = torch.cat((centers - delta, 2 * delta), dim=1)

# Ajustar las coordenadas al formato (left, top, width, height)
adjusted_boxes = torch.zeros_like(input_boxes)
adjusted_boxes[:, 0] = input_boxes[:, 0] - input_boxes[:, 2] / 2  # left
adjusted_boxes[:, 1] = input_boxes[:, 1] - input_boxes[:, 3] / 2  # top
adjusted_boxes[:, 2] = input_boxes[:, 2]  # width
adjusted_boxes[:, 3] = input_boxes[:, 3]  # height

expanded_bboxes = expanded_bboxes.to(device=device)


# Aplicar la transformación a los bounding boxes ampliados
transformed_boxes = predictor.transform.apply_boxes_torch(adjusted_boxes, image.shape[:2])

# Predecir la máscara
masks, _, _ = predictor.predict_torch(
    point_coords=None,
    point_labels=None,
    boxes=transformed_boxes,
    multimask_output=False,
)

# Mostrar la imagen original con los bounding boxes y máscaras ampliados
plt.figure(figsize=(10, 10))
plt.imshow(image)
for mask in masks:
    show_mask(mask.cpu().numpy(), plt.gca(), random_color=True)
for bbox in expanded_bboxes:
    show_box(bbox.cpu().numpy(), plt.gca())
plt.axis('off')
plt.show()  

# Obtener el tiempo de finalización
end = time.time()
print('Tiempo transcurrido = ' + str((end - start)*1000) + ' ms')
