import subprocess
import os.path
from find_imgs import ImgFinder, Rect

SCANIMG_FLAGS = [
    "--device-name", "epkowa:interpreter:002:009",
    "--source", "Transparency Unit",
    "--resolution", "3200dpi",
    "--depth", "16",
    "--format", "tiff",
    "--progress",
    "--wait-for-button",
    "-t", "5mm",
    "-y", "170mm"
]

CONVERT_DEPTH_FLAGS = [
    "-depth", "8"
]

class ScanManager:

    def __init__(self):
        self.tmp = 'tmp'
        self.scan_index: int = 0
        self.photo_index: int = 0

        self.img_finder = ImgFinder()

    def scan_image(self, out_path: str):
        res = subprocess.run(
            ["scanimage", *SCANIMG_FLAGS,
             "--output-file", out_path]
        )
        print(res)
        res.check_returncode()

    def gen_low_depth(self, in_path: str, out_path: str):
        res = subprocess.run([
            "convert", in_path,
             *CONVERT_DEPTH_FLAGS, out_path
        ])

        res.check_returncode()

    def scan_cut(self):
        slide_file = f"{self.scan_index}.tiff"
        slide_path = os.path.join(self.tmp, slide_file)
        slide_low_file = f"{self.scan_index}.low.tiff"
        slide_low_path = os.path.join(self.tmp, slide_low_file)

        self.scan_image(slide_path)
        self.gen_low_depth(slide_path, slide_low_path)

        self.img_finder.finish_callback = lambda r: self.crop(slide_path, r)
        self.img_finder.load_img(slide_low_path)
        self.img_finder.run()

    def crop(self, in_path, rects: list[Rect]):
        for r in rects:
            crop_string = f"{r.w}x{r.h}+{r.x}+{r.y}"
            out_path = f"{self.photo_index:0>2}.tiff"
            res = subprocess.run([
                "convert", in_path,
                "-crop", crop_string,
                out_path
            ])
            res.check_returncode()
            self.photo_index +=1
        self.scan_index += 1

    def run(self):
        res = input("Enter number of first image [0]:")
        try:
            self.photo_index = int(res)
        except ValueError:
            pass
        self.scan_cut()

def main():
    m = ScanManager()
    m.run()

if __name__ == '__main__':
    main()
