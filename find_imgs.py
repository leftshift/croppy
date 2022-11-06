from typing import Callable, Any
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
        'max': 500
        })
    opening_kernel_y: int = field(default=10, metadata={
        'expose': True,
        'max': 500
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
    return ((round(rect.x/5000),rect.y))


@dataclass
class ContourRects(Stage):
    min_height: int = field(default=50, metadata={
        'expose': True,
        'max': 5000
        })
    min_width: int = field(default=500, metadata={
        'expose': True,
        'max': 5000
        })
    def process(self, contours):
        spacer_rects = list(
            map(lambda contour: Rect._make(
                cv2.boundingRect(contour)
                ), contours)
        )
        spacer_rects = list(
            filter(lambda r: r.h >= self.min_height and r.w >= self.min_width, spacer_rects)
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
    grow_width: int = field(default=50, metadata={
        'expose': True,
        'max': 500
        })
    grow_height: int = field(default=50, metadata={
        'expose': True,
        'max': 500
        })
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

            x -= self.grow_width//2
            x = max(x, 0)
            y -= self.grow_height//2
            y = max(y, 0)
            w += self.grow_width
            h += self.grow_height

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

    def __init__(self):
        self.stages: list[tuple[Stage, bool]] =  [
            (Binarize(), True),
            (Close(), False),
            (Open(), False),
            (Contours(), False),
            (ContourRects(), False),
            (ImgRects(), True)
        ]

        self.current_stage_index = 0
        self.img: cv2.Mat
        self.results: list[Any] = []
        self.finish_callback: Callable[[list[Rect]], Any] = lambda x: None

    def load_img(self, path):
        self.img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        self.current_stage_index = 0
        self.results = [self.img]

    @property
    def stage(self) -> Stage:
        return self.stages[self.current_stage_index][0]

    def trackbarCallback(self, field: str):
        def callback(val):
            prev_val = getattr(self.stage, field, val)
            if prev_val == val:
                return
            setattr(self.stage, field, val)
            self.update_stage()
        return callback

    def add_trackbars(self):
        for field in fields(self.stage):
            m = field.metadata
            if m['expose']:
                current_value = getattr(self.stage, field.name)
                cv2.createTrackbar(
                        field.name, window, current_value, m['max'],
                        self.trackbarCallback(field.name))

    def get_result(self, stage_index: int, refresh: bool = False):
        print(f"get_result of stage {stage_index}, refresh {refresh}")
        result_index = stage_index + 1

        if result_index < len(self.results) and not refresh:
            return self.results[result_index]
        else:
            print(f"generating result of stage {stage_index}")
            prev_res = self.get_result(stage_index - 1)
            res = self.stages[stage_index][0].process(prev_res)

            # clear everything but previous results
            self.results = self.results[:result_index]
            self.results.append(res)
            return res

    def update_stage(self):
        img_copy = self.img.copy()
        
        res = self.get_result(self.current_stage_index, True)
        preview, hint = self.stage.preview(res, img_copy)
        cv2.imshow(window, preview)

    def show_stage(self):
        cv2.destroyAllWindows()
        cv2.namedWindow(window, cv2.WINDOW_GUI_EXPANDED)

        self.update_stage()
        self.add_trackbars()

    def get_rects(self):
        return self.results[-1]

    def set_stage(self, index) -> bool:
        max_stage = len(self.stages)

        if index >= max_stage:
            self.output_result()
            return False

        index = max(0, index)
        self.current_stage_index = index
        self.show_stage()
        return True

    def progress_stage(self, forward: bool, skip: bool = False) -> bool:
        increment = 1 if forward else -1

        if not skip:
            return self.set_stage(self.current_stage_index + increment)
        else:
            i = self.current_stage_index + increment
            while not self.stages[i][1]:
                i += increment
            return self.set_stage(i)

    def output_result(self):
        self.finish_callback(self.get_rects())

    def handle_event(self, event: int) -> bool:
        if event == 13: # Enter
            return self.progress_stage(True, True)
        elif event == 8: # Backspace
            return self.progress_stage(False, True)
        elif event == 110: # n
            return self.progress_stage(True, False)
        elif event == 98: # b
            return self.progress_stage(False, False)
        elif event >= 49 and event <= 57: # 1-9
            stage = event - 49
            return self.set_stage(stage)
        return True

    def handle_events(self):
        while True:
            code = cv2.waitKeyEx()
            print(code)
            if code == 113 or code == 27:
                break

            should_continue = self.handle_event(code)

            if not should_continue:
                cv2.destroyAllWindows()
                break

    def run(self):
        self.show_stage()
        self.handle_events()


window = 'main'

def main():
    finder = ImgFinder()
    finder.load_img(sys.argv[1])
    finder.run()


if __name__ == '__main__':
    main()
