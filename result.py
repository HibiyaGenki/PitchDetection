import os
import json
import sys

# def process_images_with_json(base_folder, json_file, output_file):
#     with open(json_file, 'r') as file:
#         data = json.load(file)  # JSONを読み込む

#     # JSONのキーから拡張子を除去
#     data = {key.replace('.mp4', ''): value for key, value in data.items()}

#     subfolders = [
#         os.path.join(base_folder, subfolder) for subfolder in os.listdir(base_folder)
#         if os.path.isdir(os.path.join(base_folder, subfolder))
#     ]

#     with open(output_file, 'w') as output:
#         for subfolder in subfolders:
#             folder_name = os.path.basename(subfolder)  # フォルダ名を取得 (例: test_HvsW)
#             if folder_name in data:
#                 numbers = data[folder_name]
#                 image_paths = sorted(
#                     [os.path.join(subfolder, img) for img in os.listdir(subfolder) if img.lower().endswith(('jpg'))]
#                 )

#                 # 各画像と対応する数値を出力
#                 for image_path, number in zip(image_paths, numbers):
#                     output_line = f"Folder: {folder_name}, Image: {image_path}, Number: {number}\n"
#                     output.write(output_line)
#                     print(output_line.strip())

# if __name__ == "__main__":
#     if len(sys.argv) != 4:
#         print("Usage: python3 result.py <base_folder> <json_file> <output_file>")
#         sys.exit(1)

#     base_folder = sys.argv[1]
#     json_file = sys.argv[2]
#     output_file = sys.argv[3]
#     process_images_with_json(base_folder, json_file, output_file)



def process_images_with_json_to_html(base_folder, json_file, output_html):
    with open(json_file, 'r') as file:
        data = json.load(file)  # JSONを読み込む

    # JSONのキーから拡張子を除去
    data = {key.replace('.mp4', ''): value for key, value in data.items()}

    subfolders = [
        os.path.join(base_folder, subfolder) for subfolder in os.listdir(base_folder)
        if os.path.isdir(os.path.join(base_folder, subfolder))
    ]

    with open(output_html, 'w') as html_file:
        html_file.write("<html><body>\n")
        for subfolder in subfolders:
            folder_name = os.path.basename(subfolder)
            if folder_name in data:
                numbers = data[folder_name]
                image_paths = sorted(
                    [os.path.join(subfolder, img) for img in os.listdir(subfolder) if img.lower().endswith(('jpg'))]
                )

                html_file.write(f"<h2>Folder: {folder_name}</h2>\n")
                for image_path, number in zip(image_paths, numbers):
                    html_file.write(f"<div>\n")
                    html_file.write(f"<p>Number: {number}</p>\n")
                    html_file.write(f"<img src=\"{image_path}\" style=\"max-width:300px;\">\n")
                    html_file.write(f"</div>\n")
        html_file.write("</body></html>\n")
if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python3 result.py <base_folder> <json_file> <output_file>")
        sys.exit(1)

    base_folder = sys.argv[1]
    json_file = sys.argv[2]
    output_file = sys.argv[3]
    process_images_with_json_to_html(base_folder, json_file, output_file)
