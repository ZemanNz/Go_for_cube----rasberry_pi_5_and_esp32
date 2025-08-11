#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <iostream>
#include <fcntl.h>
#include <termios.h>
#include <unistd.h>
#include <cmath>
#include <vector>
#include <string>
#include <algorithm>

const float Kp = 0.001f;
const float v_const = 0.5f; // základní rychlost vpřed (0..1)
const float smoothing = 0.1f; // pro PWM změny

// UART nastavení
// int setup_uart(const char* device) {
//     int fd = open(device, O_RDWR | O_NOCTTY | O_SYNC);
//     if (fd < 0) {
//         std::cerr << "Nelze otevřít UART port!" << std::endl;
//         return -1;
//     }
//     struct termios tty;
//     memset(&tty, 0, sizeof tty);
//     if (tcgetattr(fd, &tty) != 0) {
//         std::cerr << "Chyba při čtení atributů portu!" << std::endl;
//         close(fd);
//         return -1;
//     }
//     cfsetospeed(&tty, B115200);
//     cfsetispeed(&tty, B115200);
//     tty.c_cflag = (tty.c_cflag & ~CSIZE) | CS8;     // 8-bit chars
//     tty.c_iflag &= ~IGNBRK;         // disable break processing
//     tty.c_lflag = 0;                // no signaling chars, no echo, no canonical processing
//     tty.c_oflag = 0;                // no remapping, no delays
//     tty.c_cc[VMIN]  = 0;            // read doesn't block
//     tty.c_cc[VTIME] = 5;            // 0.5 seconds read timeout
//     tty.c_iflag &= ~(IXON | IXOFF | IXANY); // shut off xon/xoff ctrl
//     tty.c_cflag |= (CLOCAL | CREAD);// ignore modem controls, enable reading
//     tty.c_cflag &= ~(PARENB | PARODD);      // shut off parity
//     tty.c_cflag &= ~CSTOPB;
//     tty.c_cflag &= ~CRTSCTS;
//     if (tcsetattr(fd, TCSANOW, &tty) != 0) {
//         std::cerr << "Chyba při nastavování portu!" << std::endl;
//         close(fd);
//         return -1;
//     }
//     return fd;
// }

// void send_uart(int fd, const std::string& msg) {
//     write(fd, msg.c_str(), msg.size());
// }

std::pair<int, int> compute_pwms(float error, int old_pwm_l, int old_pwm_r) {
    float omega = Kp * error;
    float vl = v_const - omega;
    float vr = v_const + omega;
    vl = std::max(0.0f, std::min(1.0f, vl));
    vr = std::max(0.0f, std::min(1.0f, vr));
    int pwm_l = static_cast<int>(vl * 255);
    int pwm_r = static_cast<int>(vr * 255);
    pwm_l = static_cast<int>(old_pwm_l + (pwm_l - old_pwm_l) * smoothing);
    pwm_r = static_cast<int>(old_pwm_r + (pwm_r - old_pwm_r) * smoothing);
    return {pwm_l, pwm_r};
}

float get_horizontal_error(int img_width, int center_x) {
    float output = img_width / 2.0f - center_x;
    std::cout << "Horizontal error: " << output << std::endl;
    return output;
}

int main() {
    // UART
    // int uart_fd = setup_uart("/dev/serial0");
    // if (uart_fd < 0) return 1;
    // send_uart(uart_fd, "Ahoj ESP32!\n");
    std::cout << "Odesláno: Ahoj ESP32!" << std::endl;

    // Kamera
    cv::VideoCapture cap(0);
    cap.set(cv::CAP_PROP_FRAME_WIDTH, 640);
    cap.set(cv::CAP_PROP_FRAME_HEIGHT, 480);

    // Model
    cv::dnn::Net net = cv::dnn::readNetFromONNX("best.onnx");

    int old_pwm_l = 0, old_pwm_r = 0;

    while (true) {
        cv::Mat img;
        cap >> img;
        if (img.empty()) {
            std::cerr << "Camera read failed." << std::endl;
            break;
        }
        int img_height = img.rows;
        int img_width = img.cols;

        // Inference
        cv::Mat blob = cv::dnn::blobFromImage(img, 1.0/255.0, cv::Size(640, 480), cv::Scalar(), true, false);
        net.setInput(blob);
        std::vector<cv::Mat> outputs;
        net.forward(outputs, net.getUnconnectedOutLayersNames());

        // Výstupní zpracování (předpokládá se YOLOv8/ONNX formát)
        std::vector<int> objects_ids;
        std::vector<cv::Point> objects_centers;
        std::vector<float> cube_distances;

        // Pro jednoduchost předpokládáme, že výstup je v outputs[0]
        // Každý řádek: [x1, y1, x2, y2, conf, class_id]
        float conf_threshold = 0.5f;
        for (int i = 0; i < outputs[0].rows; ++i) {
            float* data = (float*)outputs[0].ptr(i);
            float conf = data[4];
            if (conf < conf_threshold) continue;
            int cube_id = static_cast<int>(data[5]);
            int x1 = static_cast<int>(data[0]);
            int y1 = static_cast<int>(data[1]);
            int x2 = static_cast<int>(data[2]);
            int y2 = static_cast<int>(data[3]);
            int width = x2 - x1;
            int height = y2 - y1;

            // validace kostky podle poměru stran a velikosti
            float aspect_ratio = height != 0 ? (float)width / height : 0;
            int min_side = std::min(width, height);
            if (!(0.65f <= aspect_ratio && aspect_ratio <= 1.35f && min_side >= 30))
                continue;

            // vykreslení
            cv::rectangle(img, cv::Point(x1, y1), cv::Point(x2, y2), cv::Scalar(0,255,0), 2);
            int center_x = x1 + width / 2;
            int center_y = y1 + height / 2;
            cv::Point center(center_x, center_y);
            objects_centers.push_back(center);
            objects_ids.push_back(cube_id);

            // popisek
            std::string color_name = "UNKNOWN";
            if (cube_id == 2) color_name = "RED";
            else if (cube_id == 1) color_name = "GREEN";
            else if (cube_id == 0) color_name = "BLUE";
            cv::circle(img, center, 5, cv::Scalar(0,255,0), -1);
            cv::putText(img, color_name, cv::Point(center.x+10, center.y), cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0,255,0), 2);

            // vzdálenost od středu obrazu
            float konstanta1 = std::abs(img_width/2.0f - center_x);
            float cube_distance = ((konstanta1*konstanta1 + (img_height-center_y)*(img_height-center_y)))*0.5f;
            cube_distances.push_back(cube_distance);
        }

        // Výpis detekovaných kostek
        std::cout << "Detekované kostky:" << std::endl;
        for (size_t i = 0; i < objects_ids.size(); ++i) {
            std::string color_name = "UNKNOWN";
            if (objects_ids[i] == 2) color_name = "RED";
            else if (objects_ids[i] == 1) color_name = "GREEN";
            else if (objects_ids[i] == 0) color_name = "BLUE";
            std::cout << "  barva=" << color_name
                      << ", pozice=(" << objects_centers[i].x << "," << objects_centers[i].y << ")"
                      << ", vzdálenost=" << cube_distances[i] << std::endl;
        }

        // Najdi nejbližší kostku
        float error = 0;
        if (!cube_distances.empty()) {
            auto min_it = std::min_element(cube_distances.begin(), cube_distances.end());
            int a = std::distance(cube_distances.begin(), min_it);
            int x = objects_centers[a].x;
            int y = objects_centers[a].y;
            std::string color_name = "UNKNOWN";
            if (objects_ids[a] == 2) color_name = "RED";
            else if (objects_ids[a] == 1) color_name = "GREEN";
            else if (objects_ids[a] == 0) color_name = "BLUE";
            std::cout << "Nejbližší kostka je " << color_name << " na pozici (" << x << "," << y << "), vzdálenost=" << cube_distances[a] << std::endl;
            cv::circle(img, cv::Point(x, y), 5, cv::Scalar(255,255,255), -1);
            error = get_horizontal_error(img_width, x);
        }

        // Výpočet PWM a odeslání
        auto [pwm_l, pwm_r] = compute_pwms(error, old_pwm_l, old_pwm_r);
        old_pwm_l = pwm_l;
        old_pwm_r = pwm_r;
        char msg[32];
        snprintf(msg, sizeof(msg), "%d,%d\n", pwm_l, pwm_r);
        // send_uart(uart_fd, msg);

        // Zobrazení výsledků
        cv::imshow("Detected Cubes", img);
        if (cv::waitKey(1) == 'q') break;
        usleep(500000); // 0.5s = cca 2 FPS
    }

    // close(uart_fd);
    cap.release();
    cv::destroyAllWindows();
    return 0;
}
