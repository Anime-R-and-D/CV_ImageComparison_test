#include <opencv2/opencv.hpp>
#include <type_traits>
#include <iostream>
#include <chrono>
#include <vector>

template<typename T> void calculateMax(const cv::Mat& output, std::vector<double>& max, std::vector<std::vector<std::pair<int, int>>>& max_pos, int channels) {
    for (int y = 0; y < output.rows; y++) {
        const T* row_ptr = output.ptr<T>(y);
        for (int x = 0; x < output.cols; x++) {
            for (int k = 0; k < channels; k++) {
                double val = static_cast<double>(row_ptr[x][k]);
                if (val > max[k]) {
                    max[k] = val;
                    max_pos[k].clear();
                    max_pos[k].emplace_back(y, x);
                }
                else if (val == max[k]) {
                    max_pos[k].emplace_back(y, x);
                }
            }
        }
    }
}

template<typename T> void printPixelValues(const cv::Mat& image1, const cv::Mat& image2, const std::vector<std::vector<std::pair<int, int>>>& max_pos, int channels) {
    for (int k = 0; k < channels; k++) {
        std::cout << "Input pixel values at max positions for channel " << k << ":" << std::endl;
        for (const auto& pos : max_pos[k]) {
            T p1 = image1.at<T>(pos.first, pos.second);
            T p2 = image2.at<T>(pos.first, pos.second);
            std::cout << "Position (" << pos.first << ", " << pos.second << "): ";
            std::cout << "Input Image1: [";
            for (int c = 0; c < channels; c++) {
                std::cout << static_cast<double>(p1[c]) << ' ';
            }
            std::cout << "] ";
            std::cout << "Input Image2: [";
            for (int c = 0; c < channels; c++) {
                std::cout << static_cast<double>(p2[c]) << ' ';
               
            }
            std::cout << "]" << std::endl;
        }
    }
}

int main(int argc, char* argv[]) {
    std::vector<int> formats;

    _putenv("OPENCV_IO_ENABLE_OPENEXR=1");

    cv::Mat image1 = cv::imread(argv[1], cv::IMREAD_UNCHANGED);
    cv::Mat image2 = cv::imread(argv[2], cv::IMREAD_UNCHANGED);

    if (image1.empty()) {
        std::cout << "Could not open or find the image1" << std::endl;
        return -1;
    }

    if (image2.empty()) {
        std::cout << "Could not open or find the image2" << std::endl;
        return -1;
    }

    cv::Mat output;

    cv::absdiff(image1, image2, output);

    int channels = output.channels();
    int depth = output.depth();

    std::vector<double> max(channels, 0);
    std::vector<std::vector<std::pair<int, int>>> max_pos(channels);

    if (depth == CV_8U) {
        calculateMax<cv::Vec3b>(output, max, max_pos, channels);
        printPixelValues<cv::Vec3b>(image1, image2, max_pos, channels);
    }
    else if (depth == CV_16U) {
        calculateMax<cv::Vec3w>(output, max, max_pos, channels);
        printPixelValues<cv::Vec3w>(image1, image2, max_pos, channels);
    }
    else if (depth == CV_32F) {
        calculateMax<cv::Vec3f>(output, max, max_pos, channels);
        printPixelValues<cv::Vec3f>(image1, image2, max_pos, channels);
    }

    cv::Scalar mean_values = cv::mean(output);
    std::cout << "Mean: " << mean_values << std::endl;

    std::cout << "Max: ";
    for (int i = 0; i < channels; i++) {
        std::cout << max[i] << " ";
    }
    std::cout << std::endl;

    for (int k = 0; k < channels; k++) {
        std::cout << "Max positions for channel " << k << ": ";
        for (const auto& pos : max_pos[k]) {
            std::cout << "(" << pos.first << ", " << pos.second << ") ";
        }
        std::cout << std::endl;
    }

    cv::imshow("Display Window", output);
    cv::waitKey(0);

    return 0;
}
