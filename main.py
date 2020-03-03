from sudoku import Sudoku
from utils.utils import parse_json,load_image
import argparse

args = argparse.ArgumentParser()
args.add_argument('--image',help='the path to your test image',default='data/test.jpeg')

def test_image(image_path):
    sudoku_config = parse_json("config/sudoku.json")
    image_processor_config = parse_json("config/image_processor.json")
    sudoku = Sudoku(sudoku_config=sudoku_config,preprocess_config=image_processor_config)
    image = load_image(image_path)
    sudoku.run_pipeline(image)

if __name__ == '__main__':
    arg = args.parse_args()
    image_path = arg.image
    test_image(image_path)