#!/usr/bin/env python3
"""
Remote Joystick Client for unitree_mujoco NetworkJoystick
Reads a local gamepad via pygame and sends binary UDP packets
to unitree_mujoco running with joystick_type: "network".

Packet format (18 bytes, little-endian):
  uint16_t  buttons   - BtnUnion bit field (see below)
  float     lx        - Left stick X  [-1.0, 1.0]
  float     ly        - Left stick Y  [-1.0, 1.0]
  float     rx        - Right stick X [-1.0, 1.0]
  float     ry        - Right stick Y [-1.0, 1.0]

BtnUnion bit layout (matches unitree_joystick.hpp):
  bit 0:  R1 / RB          bit 8:  A
  bit 1:  L1 / LB          bit 9:  B
  bit 2:  Start             bit 10: X
  bit 3:  Select / Back     bit 11: Y
  bit 4:  R2 / RT           bit 12: Up
  bit 5:  L2 / LT           bit 13: Right
  bit 6:  F1                bit 14: Down
  bit 7:  F2                bit 15: Left

Usage:
  pip install pygame
  python3 remote_joystick_client.py --host <GPU_WORKSTATION_IP> --port 5000
"""

import pygame
import socket
import struct
import sys
import argparse
import time
import platform


# -- Xbox-style button index -> BtnUnion bit --
# Standard SDL/pygame Xbox mapping:
#   0=A, 1=B, 2=X, 3=Y, 4=LB, 5=RB, 6=Back, 7=Start
XBOX_BUTTON_MAP = {
    0: 8,   # A  -> bit 8
    1: 9,   # B  -> bit 9
    2: 10,  # X  -> bit 10
    3: 11,  # Y  -> bit 11
    4: 1,   # LB -> bit 1
    5: 0,   # RB -> bit 0
    6: 3,   # Back/Select -> bit 3
    7: 2,   # Start -> bit 2
}

# D-pad (hat) -> BtnUnion bits
DPAD_UP    = 12
DPAD_RIGHT = 13
DPAD_DOWN  = 14
DPAD_LEFT  = 15

# D-pad as buttons (common on Windows Xbox controllers with 16+ buttons)
# Maps pygame button index -> BtnUnion bit
DPAD_BUTTON_MAP = {
    11: DPAD_UP,
    12: DPAD_DOWN,
    13: DPAD_LEFT,
    14: DPAD_RIGHT,
}


# Axis layout definitions:
#   Xbox:  0=LX, 1=LY, 2=LT, 3=RX, 4=RY, 5=RT
#   PS:    0=LX, 1=LY, 2=RX, 3=RY, 4=LT, 5=RT
AXIS_LAYOUTS = {
    'xbox': {'lx': 0, 'ly': 1, 'rx': 3, 'ry': 4, 'lt': 2, 'rt': 5},
    'ps':   {'lx': 0, 'ly': 1, 'rx': 2, 'ry': 3, 'lt': 4, 'rt': 5},
}

def detect_layout(controller_name):
    """Auto-detect axis layout from controller name and OS.
    pygame/SDL normalizes ALL controllers to the same axis order on
    every platform (Linux, Windows, macOS):
      0=LX, 1=LY, 2=RX, 3=RY, 4=LT, 5=RT
    This is the 'ps' layout.  The 'xbox' XInput-native order
    (LT=2, RX=3, RY=4) does NOT apply when going through pygame.
    """
    # pygame/SDL always uses this order — return 'ps' unconditionally.
    return 'ps'


def calibrate(device_id=0):
    """Interactive calibration mode — prints raw axis/button events."""
    pygame.init()
    pygame.display.set_mode((400, 200))
    pygame.display.set_caption('Joystick Calibration')
    pygame.joystick.init()
    if pygame.joystick.get_count() == 0:
        print('No joystick found!'); return
    js = pygame.joystick.Joystick(device_id)
    js.init()
    print(f'Controller: {js.get_name()}')
    print(f'Axes: {js.get_numaxes()}  Buttons: {js.get_numbuttons()}  Hats: {js.get_numhats()}')
    print('\nMove sticks and press buttons to see raw events.')
    print('Press Ctrl+C or close window to exit.\n')
    try:
        while True:
            for ev in pygame.event.get():
                if ev.type == pygame.QUIT:
                    return
                elif ev.type == pygame.JOYAXISMOTION:
                    if abs(ev.value) > 0.15:
                        print(f'  AXIS {ev.axis:2d}  = {ev.value:+.3f}')
                elif ev.type == pygame.JOYBUTTONDOWN:
                    print(f'  BUTTON {ev.button} DOWN')
                elif ev.type == pygame.JOYBUTTONUP:
                    print(f'  BUTTON {ev.button} UP')
                elif ev.type == pygame.JOYHATMOTION:
                    print(f'  HAT {ev.hat} = {ev.value}')
            pygame.time.wait(10)
    except KeyboardInterrupt:
        pass
    finally:
        pygame.quit()


class RemoteJoystickClient:
    def __init__(self, server_host, server_port=5000, device_id=0, layout='auto'):
        self.server_host = server_host
        self.server_port = server_port
        self.device_id = device_id
        self.layout_name = layout
        self.deadzone = 0.15

        # State
        self.buttons = 0          # uint16 bit field
        self.lx = 0.0
        self.ly = 0.0
        self.rx = 0.0
        self.ry = 0.0

        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    # -- helpers ---
    def _apply_deadzone(self, val):
        return 0.0 if abs(val) < self.deadzone else val

    def _set_bit(self, bit, on):
        if on:
            self.buttons |= (1 << bit)
        else:
            self.buttons &= ~(1 << bit)

    def _pack(self):
        """Pack state into 18-byte little-endian packet."""
        return struct.pack('<H4f',
                           self.buttons,
                           self.lx, self.ly,
                           self.rx, self.ry)

    def _send(self):
        self.sock.sendto(self._pack(), (self.server_host, self.server_port))

    # -- main loop ---
    def run(self):
        pygame.init()
        screen = pygame.display.set_mode((480, 360))
        pygame.display.set_caption("Remote Joystick -> unitree_mujoco")

        pygame.joystick.init()
        if pygame.joystick.get_count() == 0:
            print("[Client] No joystick found!  Plug one in and retry.")
            return
        js = pygame.joystick.Joystick(self.device_id)
        js.init()

        # Detect or use specified layout
        if self.layout_name == 'auto':
            self.layout_name = detect_layout(js.get_name())
        self.axes = AXIS_LAYOUTS[self.layout_name]

        num_buttons = js.get_numbuttons()
        num_hats = js.get_numhats()
        num_axes = js.get_numaxes()

        print(f"[Client] Joystick : {js.get_name()}")
        print(f"[Client] Axes={num_axes}  Buttons={num_buttons}  Hats={num_hats}")
        print(f"[Client] Layout  : {self.layout_name} (LX={self.axes['lx']} LY={self.axes['ly']} RX={self.axes['rx']} RY={self.axes['ry']} LT={self.axes['lt']} RT={self.axes['rt']})")
        print(f"[Client] Trigger threshold: 0.2  (works for Linux -1..+1 and Windows 0..+1)")
        if num_buttons > 11:
            print(f"[Client] D-pad buttons enabled: {list(DPAD_BUTTON_MAP.keys())}")
        print(f"[Client] Sending to {self.server_host}:{self.server_port}")
        print(f"[Client] Ready -- move sticks / press buttons!")

        clock = pygame.time.Clock()
        font = pygame.font.SysFont("monospace", 16)
        last_send = 0.0
        SEND_HZ = 100  # 100 Hz send rate
        TRIGGER_THRESHOLD = 0.2  # Works for both Linux (-1..+1) and Windows (0..+1)

        running = True
        while running:
            # --- Process events (only for window close) ---
            for ev in pygame.event.get():
                if ev.type == pygame.QUIT:
                    running = False

            # --- Poll all inputs each frame (more reliable than events) ---
            self.buttons = 0  # Reset each frame, rebuild from polled state

            # Standard buttons (A, B, X, Y, LB, RB, Back, Start)
            for btn_idx, bit in XBOX_BUTTON_MAP.items():
                if btn_idx < num_buttons and js.get_button(btn_idx):
                    self.buttons |= (1 << bit)

            # D-pad from hat
            if num_hats > 0:
                hx, hy = js.get_hat(0)
                if hy > 0:  self.buttons |= (1 << DPAD_UP)
                if hy < 0:  self.buttons |= (1 << DPAD_DOWN)
                if hx < 0:  self.buttons |= (1 << DPAD_LEFT)
                if hx > 0:  self.buttons |= (1 << DPAD_RIGHT)

            # D-pad from buttons (fallback — some Windows Xbox setups use buttons 11-14)
            for btn_idx, bit in DPAD_BUTTON_MAP.items():
                if btn_idx < num_buttons and js.get_button(btn_idx):
                    self.buttons |= (1 << bit)

            # Sticks
            ax = self.axes
            self.lx = self._apply_deadzone(js.get_axis(ax['lx']))
            self.ly = -self._apply_deadzone(js.get_axis(ax['ly']))
            self.rx = self._apply_deadzone(js.get_axis(ax['rx']))
            self.ry = -self._apply_deadzone(js.get_axis(ax['ry']))

            # Triggers — use > 0.2 threshold (cross-platform)
            #   Linux:   released = -1.0, pressed = +1.0  → pressed > 0.2 ✓
            #   Windows: released =  0.0, pressed = +1.0  → pressed > 0.2 ✓
            lt_raw = js.get_axis(ax['lt']) if ax['lt'] < num_axes else -1.0
            rt_raw = js.get_axis(ax['rt']) if ax['rt'] < num_axes else -1.0
            if lt_raw > TRIGGER_THRESHOLD:
                self.buttons |= (1 << 5)   # LT -> bit 5
            if rt_raw > TRIGGER_THRESHOLD:
                self.buttons |= (1 << 4)   # RT -> bit 4

            # Send at fixed rate
            now = time.time()
            if now - last_send >= 1.0 / SEND_HZ:
                self._send()
                last_send = now

            # -- HUD --
            screen.fill((30, 30, 30))
            lines = [
                f"Server: {self.server_host}:{self.server_port}",
                f"Buttons: {self.buttons:016b}",
                f"LX: {self.lx:+.2f}   LY: {self.ly:+.2f}",
                f"RX: {self.rx:+.2f}   RY: {self.ry:+.2f}",
                f"LT raw: {lt_raw:+.2f}   RT raw: {rt_raw:+.2f}",
                "",
                "Button states:",
                f"  A={bool(self.buttons>>8&1)}  B={bool(self.buttons>>9&1)}  "
                f"X={bool(self.buttons>>10&1)}  Y={bool(self.buttons>>11&1)}",
                f"  LB={bool(self.buttons>>1&1)}  RB={bool(self.buttons>>0&1)}  "
                f"LT={bool(self.buttons>>5&1)}  RT={bool(self.buttons>>4&1)}",
                f"  Start={bool(self.buttons>>2&1)}  Back={bool(self.buttons>>3&1)}",
                f"  Up={bool(self.buttons>>12&1)}  Down={bool(self.buttons>>14&1)}  "
                f"Left={bool(self.buttons>>15&1)}  Right={bool(self.buttons>>13&1)}",
            ]
            for i, line in enumerate(lines):
                surf = font.render(line, True, (0, 255, 100))
                screen.blit(surf, (10, 10 + i * 22))

            pygame.display.flip()
            clock.tick(120)

        pygame.quit()
        print("[Client] Closed.")


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description='Remote Joystick Client for unitree_mujoco')
    ap.add_argument('--host', default=None, help='IP of the machine running unitree_mujoco')
    ap.add_argument('--port', type=int, default=5000, help='UDP port (default 5000)')
    ap.add_argument('--device', type=int, default=0, help='Joystick device index')
    ap.add_argument('--layout', default='auto', choices=['auto', 'xbox', 'ps'],
                    help='Axis layout: auto (detect by OS), xbox, or ps')
    ap.add_argument('--calibrate', action='store_true',
                    help='Run calibration mode to see raw axis/button events')
    args = ap.parse_args()

    if args.calibrate:
        calibrate(args.device)
    else:
        if not args.host:
            ap.error('--host is required (unless using --calibrate)')
        RemoteJoystickClient(args.host, args.port, args.device, args.layout).run()
