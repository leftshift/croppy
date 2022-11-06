# Croppy

Collection of tools for scanning sheets of film negatives and cropping out single images.

Right now, this is a slightly odd collection that works well for the Epson Perfection V600 Photo, your mileage may vary.

I'd generally be open to the idea of adding a mechanism for presets for different scanners, both to the scanning and cropping tools, and in general making this a bit easier to use.

## scan.py

This is a thin wrapper around `scanimg`. It will repeatedly wait for the scan button (on the v600, that's the rightmost button) to be pressed and save the result with incrementing file names into `./tmp` in the current directory.

You'll need to adjust the device name at the top of the file in the flags. Find the name of your scanner by running `scanimage --list-devices`.

It will also optionally (TODO: make this actually optional) create a named pipe `./tmp/scanfiles` where it will write the path of each scanned .tiff once finished.

## crop.py
This is used for cropping the single photos out of the scan tiffs.

It can either be invoked with a scanned tiff as its first argument to just crop out images from a single scan. If it is started without arguments, it listenes to the named pipe in `tmp` and lets you process multiple scans as they come in.

First, you are asked for the number of the first image. This should correspond to the numbers listed on the film to make identifying single photos easier along the way.

It uses imagemagick's `convert` to generate a low bit depth version of the scanned tiff. This is needed because otherwise, the file is apparently too large for opencv.

It then invokes the interactive image finding of `find_imgs.py`.

Once `find_imgs.py` is done, it again uses `convert` to crop out the single photos out of the scan tiff and saves them to incrementing file names in the current directory.

## find_imgs.py

This uses opencv to process the scanned image in multiple stages and finally determine the rectangles around the photos on the film strips. It's usage is described in a bit more detail below.

# Quick start
* Open terminal, go to target directory for your photos
* Open second terminal in same folder
* Start scan.py in first terminal
* Start crop.py

## Using `find_imgs.py`
Once a photo comes into crop.py, an opencv window will open and guide you through the stages. Most stages have a hint text and some expose sliders to adjust some values of the current stage.

I'd suggest to go through the `Threshold`, `Close` and `Open` stages for the first scan and adjust accordingly. The parameters are retained for subsequent scans in the same run, so for the second, you may try hitting enter and going back to adjust if neccessary.

Keyboard shortcuts:
* press `n` to progress to the next stage
* press `b` to go to the previous stage
* press `enter` to go to the next significant stage
    * press `enter` in the last stage ImgRects to accept the result
* press `backspace` to go to the previous significant stage
* press a number key to directly go to to the corresponding stage

# Screenshots
![OpenCV window showing black rectangle with white areas, a slider at the bottom of the window](/screenshots/binarize.png)
![OpenCV window showing two negative film strips. Each photo has a green rectangle around it](/screenshots/final.png)
![Previous picture zoomed in. There is a corner of one photo with a green outline and the number 6](/screenshots/final_closeup.png)

# How does it even work?
It works by trying to identify the light areas of film between the images.

In the first step, the image is binarized based on a lightness threshold. Then come two steps of morphological operations to supress unwanted areas and clean up the dividers. Then, the bounding boxes of the dividers are determined. In the last step, the images are whatever is between two bounding boxes.
