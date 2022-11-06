import subprocess
import os
import os.path

SCANIMG_FLAGS = [
    "--device-name", "epkowa:interpreter:002:017",
    "--source", "Transparency Unit",
    "--resolution", "3200dpi",
    "--depth", "16",
    "--format", "tiff",
    "--progress",
    "--wait-for-button",
    "-t", "5mm",
    "-y", "200mm" #"170mm"
]

class ScanManager:

    def __init__(self):
        self.tmp = 'tmp'
        self.scanfiles_fifo_path = os.path.join(self.tmp, 'scanfiles')
        self.scan_index: int = 0

    def call_scanimage(self, out_path: str):
        print(f"Scanning to {out_path}")
        print("Press button on scanner to start scanning")
        res = subprocess.run(
            ["scanimage", *SCANIMG_FLAGS,
             "--output-file", out_path]
        )
        print(res)
        res.check_returncode()


    def scan(self):
        slide_file = f"{self.scan_index}.tiff"
        slide_path = os.path.join(self.tmp, slide_file)

        self.call_scanimage(slide_path)
        self.scanfiles_fifo.write(slide_path + "\n")
        self.scanfiles_fifo.flush()
        self.scan_index +=1

    def run(self):
        os.makedirs(self.tmp, exist_ok=True)
        try:
            os.mkfifo(self.scanfiles_fifo_path)
        except FileExistsError:
            pass

        print("opening fifo tmp/scanfiles, please start crop.py…")
        self.scanfiles_fifo = open(self.scanfiles_fifo_path, 'w')
        while True:
            self.scan()

def main():
    m = ScanManager()
    m.run()

if __name__ == '__main__':
    main()
