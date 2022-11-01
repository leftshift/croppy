import sys
from collections import namedtuple
from dataclasses import dataclass, field, fields
from abc import ABC, abstractmethod
import cv2
import numpy as np

Rect = namedtuple('Rect', ['x', 'y', 'w', 'h'])

@dataclass
class Stage(ABC):
    @abstractmethod
    def process(self, prev_res):
        pass

    @abstractmethod
    def preview(self, res, img: cv2.Mat) -> cv2.Mat:
        pass

@dataclass
class Binarize(Stage):
    threshold: int = field(metadata={
        'expose': True,
        'max': 255
        }, default=138)

    def process(self, img):
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return cv2.threshold(gray_img, self.threshold, 255, cv2.THRESH_BINARY)[1]

    def preview(self, res, img):
        return res

@dataclass
class Morph(Stage):
    closing_kernel_x: int = field(default=10, metadata={
        'expose': True,
        'max': 100
        })
    closing_kernel_y: int = field(default=50, metadata={
        'expose': True,
        'max': 100
        })

    opening_kernel_x: int = field(default=25, metadata={
        'expose': True,
        'max': 100
        })
    opening_kernel_y: int = field(default=10, metadata={
        'expose': True,
        'max': 100
        })

    def process(self, img):
        # close to get rid of small light areas
        kernel = np.ones((self.closing_kernel_x, self.closing_kernel_y), np.uint8)
        closing = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel, iterations=2)

        # open to get rid of black lines in light areas
        kernel2 = np.ones((self.opening_kernel_x, self.opening_kernel_y), np.uint8)
        opening = cv2.morphologyEx(closing, cv2.MORPH_CLOSE, kernel2, iterations=1)

        return opening

    def preview(self, res, img):
        return res

@dataclass
class Contours(Stage):
    def process(self, img):
        contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        return contours

    def preview(self, res, img):
        cv2.drawContours(img, res, -1, (255,0,0), 3)
        return img


def rect_sort_keyfn(rect: Rect):
    return ((round(rect.x/1000),rect.y))


@dataclass
class ContourRects(Stage):
    def process(self, contours):
        spacer_rects = list(
            map(lambda contour: Rect._make(cv2.boundingRect(contour)), contours)
        )
        spacer_rects.sort(key=rect_sort_keyfn)
        return spacer_rects

    def preview(self, res, img):
        for i, r in enumerate(res):
            cv2.rectangle(img, (r.x,r.y),(r.x+r.w,r.y+r.h),(0,0,255),4)
            cv2.putText(img, str(i), (r.x, r.y+r.h), cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 4, (0,0,255), 4)
        return img

@dataclass
class ImgRects(Stage):
    def process(self, spacer_rects):
        # find pictures between spacers
        picture_rects = []
        prev_rect = None
        for rect in spacer_rects:
            if not prev_rect:
                # first spacer of first row
                prev_rect = rect
                continue
            if prev_rect.y + prev_rect.h > rect.y:
                # reached first spacer of second row
                prev_rect = rect
                continue
            x = min(prev_rect.x, rect.x)
            y = prev_rect.y + prev_rect.h
            w = max(prev_rect.x + prev_rect.w, rect.x + rect.w) - x
            h = rect.y - y
            picture = Rect(x,y,w,h)
            picture_rects.append(picture)

            prev_rect = rect
        return picture_rects
    
    def preview(self, picture_rects, img):
        for i, r in enumerate(picture_rects):
            cv2.rectangle(img, (r.x,r.y),(r.x+r.w,r.y+r.h),(0,255,0),4)
            cv2.putText(img, str(i), (r.x, r.y+r.h), cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 4, (0,255,0), 4)
        return img


class ImgFinder:
    stages = [
        Binarize(),
        Morph(),
        Contours(),
        ContourRects(),
        ImgRects()
    ]

    def __init__(self, img: cv2.Mat):
        self.current_stage_index = 0
        self.img = img
        self.results = [img, None, None, None, None, None]

    @property
    def stage(self):
        return self.stages[self.current_stage_index]

    # prev_res
    #current_res

    def trackbarCallback(self, field: str):
        def callback(val):
            setattr(self.stage, field, val)
            self.update_stage()
        return callback

    def add_trackbars(self):
        for field in fields(self.stage):
            m = field.metadata
            if m['expose']:
                cv2.createTrackbar(
                        field.name, window, field.default, m['max'],
                        self.trackbarCallback(field.name))

    def update_stage(self):
        img_copy = img.copy()
        
        prev_res = self.results[self.current_stage_index]
        res = self.stage.process(prev_res)

        self.results[self.current_stage_index + 1] = res

        preview = self.stage.preview(res, img_copy)
        cv2.imshow(window, preview)

    def show_stage(self):
        cv2.destroyAllWindows()
        cv2.namedWindow(window, cv2.WINDOW_GUI_EXPANDED)

        self.update_stage()
        self.add_trackbars()

    def run(self):
        self.show_stage()

    def progress_stage(self):
        self.current_stage_index += 1
        self.show_stage()

    def handle_event(self, event: int):
        if event == 13:
            self.progress_stage()

window = 'main'
img = cv2.imread(sys.argv[1], cv2.IMREAD_UNCHANGED)

def main():
    finder = ImgFinder(img)
    finder.run()
    while True:
        code = cv2.waitKeyEx()
        print(code)
        if code == 113 or code == 27:
            break
        finder.handle_event(code)

main()