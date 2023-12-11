import os
import argparse

def download_shapenet(args):
    shapenet_dir = args.shapenet_dir
    output_dir = args.output_dir

    os.makedirs(output_dir, exist_ok=True)

    categories_dict = {
        '02691156': 'airplane',
        '02958343': 'car',
        '03001627': 'chair',
        '04379243': 'table',
    }

    for category_id, category_name in categories_dict.items():
        print(f'category: {category_name}')

        folder_name = "data_shapenet_" + category_name

        # unzip the .zip files to category_dir
        zip_file = os.path.join(shapenet_dir, category_id + '.zip')
        os.system(f'unzip {zip_file} -d {output_dir}')
        os.system(f'mv {output_dir}/{category_id} {output_dir}/{folder_name}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description=("Script for getting assets from ShapeNet.")
    )
    parser.add_argument(
        "--shapenet_dir", type=str,
        help="Path to the ShapeNet directory"
    )
    parser.add_argument(
        "--output_dir", type=str,
        help="Output directory to save the extracted .obj files"
    )
    args = parser.parse_args()

    download_shapenet(args)
