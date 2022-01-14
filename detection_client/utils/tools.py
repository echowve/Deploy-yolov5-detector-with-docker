import numpy as np
import cv2
import json
names = [ 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
         'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
         'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
         'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
         'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
         'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
         'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
         'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear',
         'hair drier', 'toothbrush' ]
OBJECT_STRINGS = \
{1: {'id': 1, 'name': u'person'},
 2: {'id': 2, 'name': u'bicycle'},
 3: {'id': 3, 'name': u'car'},
 4: {'id': 4, 'name': u'motorcycle'},
 5: {'id': 5, 'name': u'airplane'},
 6: {'id': 6, 'name': u'bus'},
 7: {'id': 7, 'name': u'train'},
 8: {'id': 8, 'name': u'truck'},
 9: {'id': 9, 'name': u'boat'},
 10: {'id': 10, 'name': u'traffic light'},
 11: {'id': 11, 'name': u'fire hydrant'},
 13: {'id': 13, 'name': u'stop sign'},
 14: {'id': 14, 'name': u'parking meter'},
 15: {'id': 15, 'name': u'bench'},
 16: {'id': 16, 'name': u'bird'},
 17: {'id': 17, 'name': u'cat'},
 18: {'id': 18, 'name': u'dog'},
 19: {'id': 19, 'name': u'horse'},
 20: {'id': 20, 'name': u'sheep'},
 21: {'id': 21, 'name': u'cow'},
 22: {'id': 22, 'name': u'elephant'},
 23: {'id': 23, 'name': u'bear'},
 24: {'id': 24, 'name': u'zebra'},
 25: {'id': 25, 'name': u'giraffe'},
 27: {'id': 27, 'name': u'backpack'},
 28: {'id': 28, 'name': u'umbrella'},
 31: {'id': 31, 'name': u'handbag'},
 32: {'id': 32, 'name': u'tie'},
 33: {'id': 33, 'name': u'suitcase'},
 34: {'id': 34, 'name': u'frisbee'},
 35: {'id': 35, 'name': u'skis'},
 36: {'id': 36, 'name': u'snowboard'},
 37: {'id': 37, 'name': u'sports ball'},
 38: {'id': 38, 'name': u'kite'},
 39: {'id': 39, 'name': u'baseball bat'},
 40: {'id': 40, 'name': u'baseball glove'},
 41: {'id': 41, 'name': u'skateboard'},
 42: {'id': 42, 'name': u'surfboard'},
 43: {'id': 43, 'name': u'tennis racket'},
 44: {'id': 44, 'name': u'bottle'},
 46: {'id': 46, 'name': u'wine glass'},
 47: {'id': 47, 'name': u'cup'},
 48: {'id': 48, 'name': u'fork'},
 49: {'id': 49, 'name': u'knife'},
 50: {'id': 50, 'name': u'spoon'},
 51: {'id': 51, 'name': u'bowl'},
 52: {'id': 52, 'name': u'banana'},
 53: {'id': 53, 'name': u'apple'},
 54: {'id': 54, 'name': u'sandwich'},
 55: {'id': 55, 'name': u'orange'},
 56: {'id': 56, 'name': u'broccoli'},
 57: {'id': 57, 'name': u'carrot'},
 58: {'id': 58, 'name': u'hot dog'},
 59: {'id': 59, 'name': u'pizza'},
 60: {'id': 60, 'name': u'donut'},
 61: {'id': 61, 'name': u'cake'},
 62: {'id': 62, 'name': u'chair'},
 63: {'id': 63, 'name': u'couch'},
 64: {'id': 64, 'name': u'potted plant'},
 65: {'id': 65, 'name': u'bed'},
 67: {'id': 67, 'name': u'dining table'},
 70: {'id': 70, 'name': u'toilet'},
 72: {'id': 72, 'name': u'tv'},
 73: {'id': 73, 'name': u'laptop'},
 74: {'id': 74, 'name': u'mouse'},
 75: {'id': 75, 'name': u'remote'},
 76: {'id': 76, 'name': u'keyboard'},
 77: {'id': 77, 'name': u'cell phone'},
 78: {'id': 78, 'name': u'microwave'},
 79: {'id': 79, 'name': u'oven'},
 80: {'id': 80, 'name': u'toaster'},
 81: {'id': 81, 'name': u'sink'},
 82: {'id': 82, 'name': u'refrigerator'},
 84: {'id': 84, 'name': u'book'},
 85: {'id': 85, 'name': u'clock'},
 86: {'id': 86, 'name': u'vase'},
 87: {'id': 87, 'name': u'scissors'},
 88: {'id': 88, 'name': u'teddy bear'},
 89: {'id': 89, 'name': u'hair drier'},
 90: {'id': 90, 'name': u'toothbrush'}}

np.random.seed(10)
COLORS = np.random.randint(0, 255, [1000, 3])


class Colors:
    # Ultralytics color palette https://ultralytics.com/
    def __init__(self):
        # hex = matplotlib.colors.TABLEAU_COLORS.values()
        hex = ('FF3838', 'FF9D97', 'FF701F', 'FFB21D', 'CFD231', '48F90A', '92CC17', '3DDB86', '1A9334', '00D4BB',
               '2C99A8', '00C2FF', '344593', '6473FF', '0018EC', '8438FF', '520085', 'CB38FF', 'FF95C8', 'FF37C7')
        self.palette = [self.hex2rgb('#' + c) for c in hex]
        self.n = len(self.palette)

    def __call__(self, i, bgr=False):
        c = self.palette[int(i) % self.n]
        return (c[2], c[1], c[0]) if bgr else c

    @staticmethod
    def hex2rgb(h):  # rgb order (PIL)
        return tuple(int(h[1 + i:1 + i + 2], 16) for i in (0, 2, 4))

def xyxy2xywh(x):
    # Convert nx4 boxes from [x1, y1, x2, y2] to [x, y, w, h] where xy1=top-left, xy2=bottom-right
    y = np.copy(x)
    y[:, 0] = (x[:, 0] + x[:, 2]) / 2  # x center
    y[:, 1] = (x[:, 1] + x[:, 3]) / 2  # y center
    y[:, 2] = x[:, 2] - x[:, 0]  # width
    y[:, 3] = x[:, 3] - x[:, 1]  # height
    return y

def plot_one_box(x, im, color=(128, 128, 128), label=None, line_thickness=3):
    # Plots one bounding box on image 'im' using OpenCV
    assert im.data.contiguous, 'Image not contiguous. Apply np.ascontiguousarray(im) to plot_on_box() input image.'
    tl = line_thickness or round(0.002 * (im.shape[0] + im.shape[1]) / 2) + 1  # line/font thickness
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    cv2.rectangle(im, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
    if label:
        tf = max(tl - 1, 1)  # font thickness
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        cv2.rectangle(im, c1, c2, color, -1, cv2.LINE_AA)  # filled
        cv2.putText(im, label, (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)



def visualize_detection_results(img_np, active_actors, prob_dict, OBJECT_STRINGS, COLORS):
    score_th = 0.30
    action_th = 0.20

    # copy the original image first
    disp_img = np.copy(img_np)
    H, W, C = img_np.shape
    #for ii in range(len(active_actors)):
    # active_actors = list(cur_actor.keys())
    print('active_actors', active_actors)
    print('prob_dict', prob_dict)

    print('len(active_actors)', len(active_actors))
    for ii in range(len(active_actors)):
        cur_actor = active_actors[ii]
        actor_id = str(cur_actor['actor_id'])
        print('actor_id', actor_id)
        print(prob_dict)
        cur_act_results =eval(prob_dict[actor_id]) if actor_id in prob_dict else []
        try:
            cur_box, cur_score, cur_class = cur_actor['all_boxes'][0], cur_actor['all_scores'][0], 1
        except IndexError:
            continue
        
        if cur_score < score_th: 
            continue

        top, left, bottom, right = cur_box



        left = int(W * left)
        right = int(W * right)

        top = int(H * top)
        bottom = int(H * bottom)

        conf = cur_score
        #label = bbox['class_str']
        # label = 'Class_%i' % cur_class

        actor_id = int(actor_id)

        label =OBJECT_STRINGS[cur_class]['name']
        message = '%s_%i: %% %.2f' % (label, actor_id,conf)
        action_message_list = ["%s:%.3f" % (actres[0][0:7], actres[1]) for actres in cur_act_results if actres[1]>action_th]
        print('action_message_list', action_message_list)
        # action_message = " ".join(action_message_list)

        color = COLORS[actor_id].tolist()

        cv2.rectangle(disp_img, (left,top), (right,bottom), color, 3)

        font_size =  max(0.5,(right - left)/50.0/float(len(message)))
        cv2.rectangle(disp_img, (left, top-int(font_size*40)), (right,top), color, -1)
        cv2.putText(disp_img, message, (left, top-12), 0, font_size, (np.array((255,255,255))-np.array(color)).tolist(), 1)

        #action message writing
        cv2.rectangle(disp_img, (left, top), (right,top+10*len(action_message_list)), color, -1)
        for aa, action_message in enumerate(action_message_list):
            offset = aa*10
            cv2.putText(disp_img, action_message, (left, top+5+offset), 0, 0.5, (np.array((255,255,255))-np.array(color)).tolist(), 1)

    return disp_img


