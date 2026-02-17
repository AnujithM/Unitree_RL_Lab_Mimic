// NetworkJoystick for g1_ctrl — receives joystick input over UDP
// from remote_joystick_client.py running on a separate machine.
//
// Usage in g1_ctrl:
//   ./g1_ctrl --network eth0 --remote-joystick 5000
//
// Packet format (18 bytes, little-endian):
//   uint16_t buttons   - BtnUnion bit field
//   float    lx, ly, rx, ry   - stick axes [-1.0, 1.0]

#pragma once

#include <unitree/dds_wrapper/common/unitree_joystick.hpp>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <unistd.h>
#include <fcntl.h>
#include <cstring>
#include <iostream>

class NetworkJoystick
{
public:
    static constexpr size_t PACKET_SIZE = 2 + 4 * 4;  // 18 bytes

    NetworkJoystick() = default;

    bool start(int port)
    {
        port_ = port;
        sock_fd_ = socket(AF_INET, SOCK_DGRAM, 0);
        if (sock_fd_ < 0) {
            std::cerr << "[NetworkJoystick] Error: Failed to create socket." << std::endl;
            return false;
        }

        // Non-blocking
        int flags = fcntl(sock_fd_, F_GETFL, 0);
        fcntl(sock_fd_, F_SETFL, flags | O_NONBLOCK);

        // Allow reuse
        int opt = 1;
        setsockopt(sock_fd_, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));

        // Bind to all interfaces
        struct sockaddr_in addr;
        memset(&addr, 0, sizeof(addr));
        addr.sin_family = AF_INET;
        addr.sin_addr.s_addr = INADDR_ANY;
        addr.sin_port = htons(port_);

        if (bind(sock_fd_, (struct sockaddr*)&addr, sizeof(addr)) < 0) {
            std::cerr << "[NetworkJoystick] Error: Failed to bind to port " << port_ << std::endl;
            close(sock_fd_);
            sock_fd_ = -1;
            return false;
        }

        active_ = true;
        std::cout << "[NetworkJoystick] Listening on UDP port " << port_ << std::endl;
        std::cout << "[NetworkJoystick] Waiting for remote joystick client..." << std::endl;
        return true;
    }

    ~NetworkJoystick()
    {
        if (sock_fd_ >= 0) {
            close(sock_fd_);
        }
    }

    bool isActive() const { return active_; }

    /**
     * Read all pending UDP packets (keep latest), then apply to the
     * given UnitreeJoystick reference — overriding whatever was there.
     * Returns true if new data was applied.
     */
    bool applyTo(unitree::common::UnitreeJoystick& joystick)
    {
        if (!active_) return false;

        uint8_t buf[PACKET_SIZE];
        bool got_data = false;
        struct sockaddr_in sender_addr;
        socklen_t sender_len = sizeof(sender_addr);

        // Drain all pending packets, keep the latest
        while (true) {
            ssize_t n = recvfrom(sock_fd_, buf, PACKET_SIZE, 0,
                                 (struct sockaddr*)&sender_addr, &sender_len);
            if (n <= 0) break;
            if (n == (ssize_t)PACKET_SIZE) {
                got_data = true;
                memcpy(latest_packet_, buf, PACKET_SIZE);

                if (!client_connected_) {
                    char ip_str[INET_ADDRSTRLEN];
                    inet_ntop(AF_INET, &sender_addr.sin_addr, ip_str, sizeof(ip_str));
                    std::cout << "[NetworkJoystick] Client connected from "
                              << ip_str << ":" << ntohs(sender_addr.sin_port) << std::endl;
                    client_connected_ = true;
                }
            }
        }

        if (!got_data && !client_connected_) {
            return false;  // No data yet
        }

        // Parse packet
        uint16_t buttons;
        float axes[4];
        memcpy(&buttons, latest_packet_, 2);
        memcpy(axes, latest_packet_ + 2, 16);

        // Apply buttons
        joystick.RB((buttons >> 0) & 1);
        joystick.LB((buttons >> 1) & 1);
        joystick.start((buttons >> 2) & 1);
        joystick.back((buttons >> 3) & 1);
        joystick.RT(float((buttons >> 4) & 1));
        joystick.LT(float((buttons >> 5) & 1));
        joystick.F1((buttons >> 6) & 1);
        joystick.F2((buttons >> 7) & 1);
        joystick.A((buttons >> 8) & 1);
        joystick.B((buttons >> 9) & 1);
        joystick.X((buttons >> 10) & 1);
        joystick.Y((buttons >> 11) & 1);
        joystick.up((buttons >> 12) & 1);
        joystick.right((buttons >> 13) & 1);
        joystick.down((buttons >> 14) & 1);
        joystick.left((buttons >> 15) & 1);

        // Apply axes
        joystick.lx(axes[0]);
        joystick.ly(axes[1]);
        joystick.rx(axes[2]);
        joystick.ry(axes[3]);

        return true;
    }

private:
    int port_ = 5000;
    int sock_fd_ = -1;
    bool active_ = false;
    bool client_connected_ = false;
    uint8_t latest_packet_[PACKET_SIZE] = {0};
};
