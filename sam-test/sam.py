from segment_anything import SamPredictor, SamAutomaticMaskGenerator, sam_model_registry
import cv2
import numpy as np

img = cv2.imread("test.png")
mask = cv2.imread("out.png/test/1.png", 0)
mask = cv2.threshold(mask, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

img[mask == 0] = [0, 0, 0]
cv2.imwrite("saved.png", img)
# sam = sam_model_registry["vit_h"](checkpoint="sam_vit_h_4b8939.pth")
# mask_gen = SamAutomaticMaskGenerator(sam)
# masks = mask_gen.generate(img)
# color_mask = np.zeros_like(img)
# for mask in masks:
#     color_mask[mask > 0.5] = [255, 255, 255]
#     masked_img = cv2.addWeighted(img, 0.6, color_mask, 0.4, 0)
# cv2.imwrite('masked_img.png', cv2.cvtColor(masked_img, cv2.COLOR_RGB2BGR))
# print(masks)

# cv2.imshow('image', img)
# cv2.imshow('mask', masks)
# cv2.waitKey()
# predictor = SamPredictor(sam)
# predictor.set_image(img)
# masks, _, _ = predictor.predict(<input_prompts>)