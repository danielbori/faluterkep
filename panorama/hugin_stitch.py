import subprocess
import os
import re

import PIL
from tqdm import tqdm
import pandas as pd
from PIL import Image, ImageEnhance
import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.ndimage import distance_transform_edt
import cv2

def subprocess_run(command, *args, **kwargs):
    with open(r'C:/hugin_panorama/hugin_process.log', "a+") as file:
        file.write(f"{'-'*30} {command[0]}:  {command[1::]} {'-'*30}")
        subprocess.run(command , *args, stdout=file, **kwargs)

def generate_project(images, project_file):
    command = ['pto_gen', '-o', project_file] + images
    subprocess_run(command, check=True)

def find_control_points(project_file):
    global output_dir
    command = ['cpfind', '--multirow', '--celeste', '-o', project_file, project_file]
    subprocess_run(command, check=True)
def clean_control_points(project_file):
    command = ['cpclean', '-o', project_file, project_file]
    subprocess_run(command, check=True)
def optimize_panorama(project_file):
    command = ['autooptimiser', '-a', '-m', '-l', '-s', '-o', project_file, project_file]
    subprocess_run(command, check=True)


def render_panorama(project_file, output_file):
    command = ['hugin_executor', '--stitching', '--prefix', output_file, project_file]
    subprocess_run(command, check=True)


def stitch_panorama(images, project_file, output_file):
    generate_project(images, project_file)
    find_control_points(project_file)
    clean_control_points(project_file)
    optimize_panorama(project_file)
    render_panorama(project_file, output_file)

def fill_sky(image_path, output_folder = None, gauss_kernel_size = 29, blur_line_height = 200, break_offset = 5):
    """

    Identify no data values by alpha channel, and fill it from nearest (probably sky) values line by line,
    then use a simple gaussian filter to blur the lines

    """
    if not output_folder:
        output_folder = os.path.split(image_path)[0]
    try:
        img = Image.open(image_path)
    except PIL.UnidentifiedImageError as e:
        print(f"ERROR COULD NOT OPEN IMAGE: {repr(2)}")
        return None
    if os.path.exists(os.path.join(output_folder, os.path.split(image_path)[1])):
        print(f"Image {image_path} already exists, skipping")
        return None

    img_arr = np.array(img)
    diff_360 = img_arr.shape[1] - (img_arr.shape[0]*2)
    img_arr = img_arr[:, :-diff_360, :]

    alpha = img_arr[:,:,3]

    break_line = np.argmax(alpha, axis=0)
    break_line = break_line + break_offset

    ref_line = img_arr[break_line, np.arange(img_arr.shape[1])]
    black_rows = (alpha == 0).any(axis=1)


    meeting_line = []
    for i in range(np.unique(black_rows, return_counts=True)[1][1],
                   np.unique(black_rows, return_counts=True)[1][1] + blur_line_height):
        meeting_line.append(i)

    orig_line = img_arr[meeting_line, :, :][:,:,(0,1,2)].copy()
    img_arr[black_rows] = ref_line
    black_rows[meeting_line] = True
    blurred_image_array = np.zeros_like(img_arr[black_rows])

    for i in tqdm(range(3), desc='apply gaussian_filter'):
        # Apply Gaussian Filter to image
        blurred_image_array[:, :, i] = cv2.GaussianBlur(img_arr[black_rows, :, i], (gauss_kernel_size, 1), sigmaX=0, sigmaY=0)

    new_line = blurred_image_array[meeting_line, :, :][:,:,(0,1,2)].copy()

    merged_line = np.zeros_like(orig_line)
    for i in tqdm(range(3), desc='applying gradient on meeting line'):
        merged_line[:,:,i] = ((orig_line[:,:,i] * np.array([np.linspace(0,1, orig_line.shape[0])] * alpha.shape[1]).swapaxes(1,0))
                              + (new_line[:,:,i] * np.array([np.linspace(1,0, orig_line.shape[0])] * alpha.shape[1]).swapaxes(1,0)))
    for i in range(3):
        blurred_image_array[meeting_line,:,i] = merged_line[:,:,i]

    # Step 3: Convert the blurred numpy array back to an image
    blurred_image = Image.fromarray(blurred_image_array)
    img_arr[black_rows] = blurred_image
    img_result = Image.fromarray(img_arr[:,:,(0,1,2)])
    img_result.save(os.path.join(output_folder, f"{os.path.split(image_path)[1]}_sky_filled.jpg"))
    converter = ImageEnhance.Color(img_result)
    img2 = converter.enhance(1.2)

    img2.resize( (6000, 3000), Image.LANCZOS).save(os.path.join(output_folder, f"{os.path.split(image_path)[1]}_sky_filled_resized.jpg"),
                                                     optimize=True, quality=55)

def rename_images_to_web(root_folder):
    if not root_folder:
        root_folder = r'path_to_panorama/sky_filled'
    for file in os.listdir(root_folder):
        groups = re.match(r'.*_(\d{4})_.*.tif_sky_filled_resized.jpg', file)
        if groups is not None:
            print(f"Low res: {file}")
            try:
                os.rename(os.path.join(root_folder, file), os.path.join(root_folder, 'low', f'DJI_{groups[1]}.JPG'))
            except FileExistsError:
                pass
        groups = re.match(r'.*_(\d{4})_.*.tif_sky_filled.jpg$', file)
        if groups is not None:
            print(f"High res: {file}")
            try:
                os.rename(os.path.join(root_folder, file), os.path.join(root_folder, 'full', f'DJI_{groups[1]}.JPG'))
            except FileExistsError:
                pass


if __name__ == "__main__":
    root_dir = r"J:\\"
    root_dir_list = []
    output_dir = r'path_to_panorama'
    hugin_path = r'C:\Program Files\Hugin\bin'
    os.chdir(hugin_path)

    file_list_csv = r"path_to_panorama/panoramas.csv"
    df = pd.read_csv(file_list_csv)

    df['full_path'] = [os.path.join(x, y) for x, y in zip(df['Path'], df['Name'])]
    df = df[~df['Path'].str.startswith('G')]

    # Merge panorama images
    for path in tqdm(df['full_path'].to_list(), desc='Merging 360panoramas'):
        images = [os.path.join(path, file) for file in os.listdir(path) if re.match(r'PANO\d{4}.(JPG|jpg)', file)]
        if len(images) > 40 or len(images) < 10:
            print(f"Number of images ({len(images)}) seems off, skipping pano merge folder: {path}")
            continue
        filename = f"{os.path.split(path)[1]}_{os.path.split(os.path.split(path)[0])[1]}"
        output_file = os.path.join(output_dir, filename)
        if os.path.exists(f"{output_file}.tif"):
            print(f"Output file {output_file} already exists, skipping")
            continue
        project_file = os.path.join(root_dir, os.path.join(output_dir, f'{filename}_project.pto'))
        stitch_panorama(images, project_file, output_file)

    # fill_sky
    for file in [os.path.join(root_dir, x) for x in os.listdir(output_dir) if x.endswith('tif')]:
        fill_sky(file, os.path.join(root_dir, 'sky_filled'), gauss_kernel_size=151)
