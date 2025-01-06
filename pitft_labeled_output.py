# SPDX-FileCopyrightText: 2021 Limor Fried/ladyada for Adafruit Industries
# SPDX-FileCopyrightText: 2021 Melissa LeBlanc-Williams for Adafruit Industries
#
# SPDX-License-Identifier: MIT

import time
import logging
import argparse
import pygame
import os
import subprocess
import sys
import numpy as np
import signal

CONFIDENCE_THRESHOLD = 0.5   # At what confidence level do we say we detected a thing
PERSISTANCE_THRESHOLD = 0.25  # What percentage of the time we have to have seen a thing

def dont_quit(signal, frame):
    print(f"Caught signal {signal}, exiting...")
    pygame.quit()  # Quit pygame to properly close the display
    sys.exit(0)   # Exit program

signal.signal(signal.SIGHUP, dont_quit)
signal.signal(signal.SIGINT, dont_quit)  # Handle SIGINT (Ctrl + C)
signal.signal(signal.SIGTERM, dont_quit)  # Handle termination signals

from rpi_vision.agent.capturev2 import PiCameraStream
from rpi_vision.models.mobilenet_v2 import MobileNetV2Base

logging.basicConfig()
logging.getLogger().setLevel(logging.INFO)

# Initialize the display
pygame.init()
screen = pygame.display.set_mode((800, 600))  # Default windowed mode
capture_manager = None

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--include-top', type=bool,
                        dest='include_top', default=True,
                        help='Include fully-connected layer at the top of the network.')

    parser.add_argument('--tflite',
                        dest='tflite', action='store_true', default=False,
                        help='Convert base model to TFLite FlatBuffer, then load model into TFLite Python Interpreter')

    parser.add_argument('--rotation', type=int, choices=[0, 90, 180, 270],
                        dest='rotation', action='store', default=0,
                        help='Rotate everything on the display by this amount')
    args = parser.parse_args()
    return args

last_seen = [None] * 10
last_spoken = None

def main(args):
    global last_spoken, capture_manager, screen  # Declare screen as global

    capture_manager = PiCameraStream(preview=False)

    # Initialize screen buffer for rendering
    if args.rotation in (0, 180):
        buffer = pygame.Surface((screen.get_width(), screen.get_height()))
    else:
        buffer = pygame.Surface((screen.get_height(), screen.get_width()))

    pygame.mouse.set_visible(False)
    screen.fill((0, 0, 0))

    try:
        splash = pygame.image.load(os.path.dirname(sys.argv[0]) + '/bchatsplash.bmp')
        splash = pygame.transform.rotate(splash, args.rotation)
        splash = pygame.transform.scale(splash, (min(screen.get_width(), screen.get_height()), min(screen.get_width(), screen.get_height())))
        screen.blit(splash, ((screen.get_width() - splash.get_width()) // 2, (screen.get_height() - splash.get_height()) // 2))
    except pygame.error:
        pass
    pygame.display.update()

    scale = max(buffer.get_height() // capture_manager.resolution[1], 1)
    scaled_resolution = tuple([x * scale for x in capture_manager.resolution])

    smallfont = pygame.font.Font(None, 24 * scale)
    medfont = pygame.font.Font(None, 36 * scale)
    bigfont = pygame.font.Font(None, 48 * scale)

    model = MobileNetV2Base(include_top=args.include_top)

    capture_manager.start()
    is_fullscreen = False  # Track fullscreen state

    while not capture_manager.stopped:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                print("User requested exit.")
                capture_manager.stop()
                pygame.quit()
                sys.exit(0)
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_q:  # Press 'q' to quit
                    print("Quit key pressed.")
                    capture_manager.stop()
                    pygame.quit()
                    sys.exit(0)
                elif event.key == pygame.K_f:  # Press 'f' to toggle fullscreen
                    is_fullscreen = not is_fullscreen
                    if is_fullscreen:
                        screen = pygame.display.set_mode((0, 0), pygame.FULLSCREEN)
                    else:
                        screen = pygame.display.set_mode((800, 600))

        if capture_manager.frame is None:
            continue
        buffer.fill((0, 0, 0))
        frame = capture_manager.read()
        previewframe = np.ascontiguousarray(capture_manager.frame)
        img = pygame.image.frombuffer(previewframe, capture_manager.resolution, 'RGB')
        img = pygame.transform.scale(img, scaled_resolution)

        cropped_region = (
            (img.get_width() - buffer.get_width()) // 2,
            (img.get_height() - buffer.get_height()) // 2,
            buffer.get_width(),
            buffer.get_height()
        )

        buffer.blit(img, (0, 0), cropped_region)

        timestamp = time.monotonic()
        if args.tflite:
            prediction = model.tflite_predict(frame)[0]
        else:
            prediction = model.predict(frame)[0]

        delta = time.monotonic() - timestamp
        logging.info("%s inference took %d ms, %0.1f FPS" % ("TFLite" if args.tflite else "TF", delta * 1000, 1 / delta))

        for p in prediction:
            label, name, conf = p
            if conf > CONFIDENCE_THRESHOLD:
                print("Detected", name)
                last_seen.append(name)
                last_seen.pop(0)

                persistant_obj = last_seen.count(name) / len(last_seen) > PERSISTANCE_THRESHOLD

                if persistant_obj and last_spoken != name:
                    subprocess.call(f"echo {name} | festival --tts &", shell=True)
                    last_spoken = name
                break
        else:
            last_seen.append(None)
            last_seen.pop(0)
            if all(x is None for x in last_seen):
                last_spoken = None

        screen.blit(pygame.transform.rotate(buffer, args.rotation), (0, 0))
        pygame.display.update()

if __name__ == "__main__":
    args = parse_args()
    try:
        main(args)
    except KeyboardInterrupt:
        print("Program interrupted by user. Exiting...")
        capture_manager.stop()
        pygame.quit()
        sys.exit(0)
