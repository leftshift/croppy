import sys
import subprocess
import os.path
from find_imgs import ImgFinder, Rect

CONVERT_DEPTH_FLAGS = [
    "-depth", "8"
]

class CropManager:

    def __init__(self, first_photo_index: int):
        self.tmp = 'tmp'
        self.scanfiles_fifo_path = os.path.join(self.tmp, 'scanfiles')
        self.photo_index = first_photo_index

        self.img_finder = ImgFinder()

    def gen_low_depth(self, in_path: str, out_path: str):
        print("Generating low bit depth version…")
        res = subprocess.run([
            "convert", in_path,
             *CONVERT_DEPTH_FLAGS, out_path
        ])

        res.check_returncode()

    def crop(self, in_path, rects: list[Rect]):
        print("Cropping file…")
        for r in rects:
            crop_string = f"{r.w}x{r.h}+{r.x}+{r.y}"
            out_path = f"{self.photo_index:0>2}.tiff"
            print(f"\tGenerating {out_path}")
            res = subprocess.run([
                "convert", in_path,
                "-crop", crop_string,
                out_path
            ])
            res.check_returncode()
            self.photo_index +=1

    def convert_crop(self, scan_path: str):
        scan_basename = os.path.basename(scan_path)
        scan_name = os.path.splitext(scan_basename)[0]
        scan_low_file = f"{scan_name}.low.tiff"
        scan_low_path = os.path.join(self.tmp, scan_low_file)

        self.gen_low_depth(scan_path, scan_low_path)

        self.img_finder.finish_callback = lambda r: self.crop(scan_path, r)
        self.img_finder.load_img(scan_low_path)
        self.img_finder.run()

    def crop_multiple(self):
        scanfiles_fifo = open(self.scanfiles_fifo_path, 'r')

        for path in scanfiles_fifo:
            self.convert_crop(path.strip())

def main():
    photo_index = 0
    res = input("Enter number of first image [0]:")
    try:
        photo_index = int(res)
    except ValueError:
        pass

    m = CropManager(photo_index)
    if len(sys.argv) > 1:
        print("using supplied path as infile")
        m.convert_crop(sys.argv[1])
    else:
        print("using paths from tmp/scanfiles")
        m.crop_multiple()

if __name__ == '__main__':
    main()

