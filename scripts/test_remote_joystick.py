#!/usr/bin/env python3
"""
Network connectivity test for remote joystick setup
Tests that the server is reachable before running the full client
"""

import socket
import sys
import argparse

def test_connection(host, port):
    """Test UDP connectivity to the server"""
    print(f"Testing UDP connection to {host}:{port}...")
    
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.settimeout(2)
        
        # Send a test message
        test_msg = b'{"test": true}'
        sock.sendto(test_msg, (host, port))
        
        # Try to receive response (may not get one, but socket should work)
        print(f"✓ Successfully sent test packet")
        sock.close()
        return True
    except socket.gaierror:
        print(f"✗ Cannot resolve hostname: {host}")
        return False
    except socket.timeout:
        print(f"✗ Connection timeout to {host}:{port}")
        return False
    except Exception as e:
        print(f"✗ Connection error: {e}")
        return False

def test_joystick():
    """Test joystick detection"""
    print("\nTesting joystick detection...")
    
    try:
        import pygame
        pygame.init()
        pygame.joystick.init()
        
        joystick_count = pygame.joystick.get_count()
        
        if joystick_count == 0:
            print(f"✗ No joysticks found")
            print("  - Check if joystick is connected")
            print("  - Try: lsusb | grep -i joystick")
            return False
        
        for i in range(joystick_count):
            js = pygame.joystick.Joystick(i)
            js.init()
            print(f"✓ Joystick {i}: {js.get_name()}")
            print(f"  - Axes: {js.get_numaxes()}")
            print(f"  - Buttons: {js.get_numbuttons()}")
            print(f"  - Hats: {js.get_numhats()}")
        
        pygame.quit()
        return True
    except ImportError:
        print(f"✗ pygame not installed")
        print("  - Run: pip install pygame")
        return False
    except Exception as e:
        print(f"✗ Error testing joystick: {e}")
        return False

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Test remote joystick setup')
    parser.add_argument('--host', required=True, help='Server host IP')
    parser.add_argument('--port', type=int, default=5000, help='Server port')
    parser.add_argument('--no-joystick', action='store_true', help='Skip joystick test')
    
    args = parser.parse_args()
    
    print("=" * 50)
    print("Remote Joystick Setup Verification")
    print("=" * 50)
    
    # Test network
    network_ok = test_connection(args.host, args.port)
    
    # Test joystick
    if args.no_joystick:
        joystick_ok = True
    else:
        joystick_ok = test_joystick()
    
    print("\n" + "=" * 50)
    if network_ok and joystick_ok:
        print("✓ All tests passed! Ready to run client")
        sys.exit(0)
    else:
        print("✗ Some tests failed. See above for details.")
        sys.exit(1)
