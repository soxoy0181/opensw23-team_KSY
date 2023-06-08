# python 3.6
"""diffuses target images to context images with In-domain GAN Inversion.

Basically, this script first copies the central region from the target image to
the context image, and then performs in-domain GAN inversion on the stitched
image. Different from `intert.py`, masked reconstruction loss is used in the
optimization stage.

NOTE: This script will diffuse every image from `target_image_list` to every
image from `context_image_list`.
"""

import os
import argparse
from tqdm import tqdm
import numpy as np
import dlib
from PIL import Image
import scipy


from utils.inverter import StyleGANInverter
from utils.logger import setup_logger
from utils.visualizer import HtmlPageVisualizer
from utils.visualizer import load_image, resize_image

MODEL_DIR = os.path.join('models', 'pretrain')

LANDMARK_MODEL_NAME = 'shape_predictor_68_face_landmarks.dat'
LANDMARK_MODEL_PATH = os.path.join(MODEL_DIR, LANDMARK_MODEL_NAME)
LANDMARK_MODEL_URL = f'http://dlib.net/files/{LANDMARK_MODEL_NAME}.bz2'

class FaceLandmarkDetector(object):
  """Class of face landmark detector."""

  def __init__(self, align_size=256, enable_padding=True):
    """Initializes face detector and landmark detector.

  Args:
    align_size: Size of the aligned face if performing face alignment.
    (default: 1024)
    enable_padding: Whether to enable padding for face alignment (default:
    True)
  """
    # Download models if needed.
    if not os.path.exists(LANDMARK_MODEL_PATH):
      data = requests.get(LANDMARK_MODEL_URL)
      data_decompressed = bz2.decompress(data.content)
      with open(LANDMARK_MODEL_PATH, 'wb') as f:
        f.write(data_decompressed)

    self.face_detector = dlib.get_frontal_face_detector()
    self.landmark_detector = dlib.shape_predictor(LANDMARK_MODEL_PATH)
    self.align_size = align_size
    self.enable_padding = enable_padding

  def detect(self, image_path):
    """Detects landmarks from the given image.

  This function will first perform face detection on the input image. All
  detected results will be grouped into a list. If no face is detected, an
  empty list will be returned.

  For each element in the list, it is a dictionary consisting of `image_path`,
  `bbox` and `landmarks`. `image_path` is the path to the input image. `bbox`
  is the 4-element bounding box with order (left, top, right, bottom), and
  `landmarks` is a list of 68 (x, y) points.

  Args:
    image_path: Path to the image to detect landmarks from.

  Returns:
    A list of dictionaries, each of which is the detection results of a
    particular face.
  """
    results = []

    # image_ = np.array(image)
    images = dlib.load_rgb_image(image_path)
    # Face detection (1 means to upsample the image for 1 time.)
    bboxes = self.face_detector(images, 1)
    # Landmark detection
    for bbox in bboxes:
      landmarks = []
      for point in self.landmark_detector(images, bbox).parts():
        landmarks.append((point.x, point.y))
      results.append({
          'image_path': image_path,
          'bbox': (bbox.left(), bbox.top(), bbox.right(), bbox.bottom()),
          'landmarks': landmarks,
      })
    return results

  def align(self, face_info):
    """Aligns face based on landmark detection.

  The face alignment process is borrowed from
  https://github.com/NVlabs/ffhq-dataset/blob/master/download_ffhq.py,
  which only supports aligning faces to square size.

  Args:
    face_info: Face information, which is the element of the list returned by
    `self.detect()`.

  Returns:
    A `np.ndarray`, containing the aligned result. It is with `RGB` channel
    order.
  """
    img = Image.open(face_info['image_path'])

    landmarks = np.array(face_info['landmarks'])
    eye_left = np.mean(landmarks[36: 42], axis=0)
    eye_right = np.mean(landmarks[42: 48], axis=0)
    eye_middle = (eye_left + eye_right) / 2
    eye_to_eye = eye_right - eye_left
    mouth_middle = (landmarks[48] + landmarks[54]) / 2
    eye_to_mouth = mouth_middle - eye_middle

    # Choose oriented crop rectangle.
    x = eye_to_eye - np.flipud(eye_to_mouth) * [-1, 1]
    x /= np.hypot(*x)
    x *= max(np.hypot(*eye_to_eye) * 2.0, np.hypot(*eye_to_mouth) * 1.8)
    y = np.flipud(x) * [-1, 1]
    c = eye_middle + eye_to_mouth * 0.1
    quad = np.stack([c - x - y, c - x + y, c + x + y, c + x - y])
    qsize = np.hypot(*x) * 2

    # Shrink.
    shrink = int(np.floor(qsize / self.align_size * 0.5))
    if shrink > 1:
      rsize = (int(np.rint(float(img.size[0]) / shrink)),
               int(np.rint(float(img.size[1]) / shrink)))
      img = img.resize(rsize, Image.ANTIALIAS)
      quad /= shrink
      qsize /= shrink

    # Crop.
    border = max(int(np.rint(qsize * 0.1)), 3)
    crop = (int(np.floor(min(quad[:, 0]))), int(np.floor(min(quad[:, 1]))),
            int(np.ceil(max(quad[:, 0]))), int(np.ceil(max(quad[:, 1]))))
    crop = (max(crop[0] - border, 0),
            max(crop[1] - border, 0),
            min(crop[2] + border, img.size[0]),
            min(crop[3] + border, img.size[1]))
    if crop[2] - crop[0] < img.size[0] or crop[3] - crop[1] < img.size[1]:
      img = img.crop(crop)
      quad -= crop[0:2]

    # Pad.
    pad = (int(np.floor(min(quad[:, 0]))), int(np.floor(min(quad[:, 1]))),
           int(np.ceil(max(quad[:, 0]))), int(np.ceil(max(quad[:, 1]))))
    pad = (max(-pad[0] + border, 0),
           max(-pad[1] + border, 0),
           max(pad[2] - img.size[0] + border, 0),
           max(pad[3] - img.size[1] + border, 0))
    if self.enable_padding and max(pad) > border - 4:
      pad = np.maximum(pad, int(np.rint(qsize * 0.3)))
      img = np.pad(np.float32(img),
                   ((pad[1], pad[3]), (pad[0], pad[2]), (0, 0)),
                   'reflect')
      h, w, _ = img.shape
      y, x, _ = np.ogrid[:h, :w, :1]
      mask = np.maximum(1.0 - np.minimum(np.float32(x) / pad[0],
                                         np.float32(w - 1 - x) / pad[2]),
                        1.0 - np.minimum(np.float32(y) / pad[1],
                                         np.float32(h - 1 - y) / pad[3]))
      blur = qsize * 0.02
      blurred_image = scipy.ndimage.gaussian_filter(img, [blur, blur, 0]) - img
      img += blurred_image * np.clip(mask * 3.0 + 1.0, 0.0, 1.0)
      img += (np.median(img, axis=(0, 1)) - img) * np.clip(mask, 0.0, 1.0)
      img = Image.fromarray(np.uint8(np.clip(np.rint(img), 0, 255)), 'RGB')
      quad += pad[:2]

    # Transform.
    img = img.transform((self.align_size * 4, self.align_size * 4), Image.QUAD,
                        (quad + 0.5).flatten(), Image.BILINEAR)
    img = img.resize((self.align_size, self.align_size), Image.ANTIALIAS)

    return np.array(img)


def align_face(image_path, align_size=256):
  """Aligns a given face."""
  model = FaceLandmarkDetector(align_size)
  face_infos = model.detect(image_path)
  face_infos = face_infos[0]
  img = model.align(face_infos)
  return img


def build_inverter(model_name, iteration=100, regularization_loss_weight=2):
  """Builds inverter"""
  inverter = StyleGANInverter(
      model_name,
      learning_rate=0.01,
      iteration=iteration,
      reconstruction_loss_weight=1.0,
      perceptual_loss_weight=5e-5,
      regularization_loss_weight=regularization_loss_weight)
  return inverter


def get_generator(model_name):
  """Gets model by name"""
  return build_generator(model_name)


def align(inverter, image_path):
  """Aligns an unloaded image."""
  aligned_image = align_face(image_path,
                             align_size=inverter.G.resolution)
  return aligned_image


def parse_args():
  """Parses arguments."""
  parser = argparse.ArgumentParser()
  parser.add_argument('model_name', type=str, help='Name of the GAN model.')
  parser.add_argument('target_list', type=str,
                      help='List of target images to diffuse from.')
  parser.add_argument('context_list', type=str,
                      help='List of context images to diffuse to.')
  parser.add_argument('-o', '--output_dir', type=str, default='',
                      help='Directory to save the results. If not specified, '
                           '`./results/diffusion` will be used by default.')
  parser.add_argument('-s', '--crop_size', type=int, default=110,
                      help='Crop size. (default: 110)')
  parser.add_argument('-x', '--center_x', type=int, default=125,
                      help='X-coordinate (column) of the center of the cropped '
                           'patch. This field should be adjusted according to '
                           'dataset and image size. (default: 125)')
  parser.add_argument('-y', '--center_y', type=int, default=145,
                      help='Y-coordinate (row) of the center of the cropped '
                           'patch. This field should be adjusted according to '
                           'dataset and image size. (default: 145)')
  parser.add_argument('--learning_rate', type=float, default=0.01,
                      help='Learning rate for optimization. (default: 0.01)')
  parser.add_argument('--num_iterations', type=int, default=100,
                      help='Number of optimization iterations. (default: 100)')
  parser.add_argument('--num_results', type=int, default=5,
                      help='Number of intermediate optimization results to '
                           'save for each sample. (default: 5)')
  parser.add_argument('--loss_weight_feat', type=float, default=5e-5,
                      help='The perceptual loss scale for optimization. '
                           '(default: 5e-5)')
  parser.add_argument('--viz_size', type=int, default=256,
                      help='Image size for visualization. (default: 256)')
  parser.add_argument('--gpu_id', type=str, default='0',
                      help='Which GPU(s) to use. (default: `0`)')
  parser.add_argument('--batch_size', type=int, default=1,
                      help='Batch size. (default: 4)')
  return parser.parse_args()


def main():
  """Main function."""
  args = parse_args()
  os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
  assert os.path.exists(args.target_list)
  target_list_name = os.path.splitext(os.path.basename(args.target_list))[0]
  assert os.path.exists(args.context_list)
  context_list_name = os.path.splitext(os.path.basename(args.context_list))[0]
  output_dir = args.output_dir or f'results/diffusion'
  job_name = f'{target_list_name}_TO_{context_list_name}'
  logger = setup_logger(output_dir, f'{job_name}.log', f'{job_name}_logger')

  logger.info(f'Loading model.')
  inverter = StyleGANInverter(
      args.model_name,
      learning_rate=args.learning_rate,
      iteration=args.num_iterations,
      reconstruction_loss_weight=1.0,
      perceptual_loss_weight=args.loss_weight_feat,
      regularization_loss_weight=0.0,
      logger=logger)
  image_size = inverter.G.resolution

  # Load image list.
  logger.info(f'Loading target images and context images.')
  target_list = []
  with open(args.target_list, 'r') as f:
    for line in f:
      target_list.append(line.strip())
  num_targets = len(target_list)
  context_list = []
  with open(args.context_list, 'r') as f:
    for line in f:
      context_list.append(line.strip())
  num_contexts = len(context_list)
  num_pairs = num_targets * num_contexts

  # Initialize visualizer.
  save_interval = args.num_iterations // args.num_results
  headers = ['Target Image', 'Context Image', 'Stitched Image',
             'Encoder Output']
  for step in range(1, args.num_iterations + 1):
    if step == args.num_iterations or step % save_interval == 0:
      headers.append(f'Step {step:06d}')
  viz_size = None if args.viz_size == 0 else args.viz_size
  visualizer = HtmlPageVisualizer(
      num_rows=num_pairs, num_cols=len(headers), viz_size=viz_size)
  visualizer.set_headers(headers)

  # Diffuse images.
  logger.info(f'Start diffusion.')
  latent_codes = []
  for target_idx in tqdm(range(num_targets), desc='Target ID', leave=False):
    # Load target.
    #target_image = resize_image(load_image(target_list[target_idx]),
    #                            (image_size, image_size))
    target_image = align(inverter, target_list[target_idx])
    if target_image.shape[2] == 4: # in case of image have four channels
        target_image = target_image[:, :, :3]
    visualizer.set_cell(target_idx * num_contexts, 0, image=target_image)
    for context_batch_idx in tqdm(range(0, num_contexts, args.batch_size),
                            desc='Context ID', leave=False):
      context_images = []
      for it in range(args.batch_size):
        context_idx = context_batch_idx + it
        if context_idx >= num_contexts:
          continue
        row_idx = target_idx * num_contexts + context_idx
        context_image = resize_image(load_image(context_list[context_idx]),
                                     (image_size, image_size))
        context_images.append(context_image)        
        visualizer.set_cell(row_idx, 1, image=context_image)
      
      code, viz_results = inverter.easy_diffuse(target=target_image,
                                                context=np.asarray(context_images),
                                                center_x=args.center_x,
                                                center_y=args.center_y,
                                                crop_x=args.crop_size,
                                                crop_y=args.crop_size,
                                                num_viz=args.num_results)
      for key, values in viz_results.items():
        context_idx = context_batch_idx + key
        row_idx = target_idx * num_contexts + context_idx
        for viz_idx, viz_img in enumerate(values):
          visualizer.set_cell(row_idx, viz_idx + 2, image=viz_img)
      latent_codes.append(code)

  # Save results.
  os.system(f'cp {args.target_list} {output_dir}/target_list.txt')
  os.system(f'cp {args.context_list} {output_dir}/context_list.txt')
  np.save(f'{output_dir}/{job_name}_inverted_codes.npy',
          np.concatenate(latent_codes, axis=0))
  visualizer.save(f'{output_dir}/{job_name}.html')


if __name__ == '__main__':
  main()
