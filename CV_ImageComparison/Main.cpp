#include <opencv2/opencv.hpp>
#include <type_traits>
#include <iostream>
#include <chrono>
#include <vector>
#include <windows.h>
#include <thread>
#include <omp.h>

template<typename T> void calculateMax(const cv::Mat& output, std::vector<double>& max, std::vector<std::vector<std::pair<int, int>>>& max_pos, int channels) {
    auto start = std::chrono::high_resolution_clock::now();

    std::vector<std::vector<double>> local_max(omp_get_max_threads(), std::vector<double>(channels, 0));
    std::vector<std::vector<std::vector<std::pair<int, int>>>> local_max_pos(omp_get_max_threads(), std::vector<std::vector<std::pair<int, int>>>(channels));

#pragma omp parallel
    {
        int thread_id = omp_get_thread_num();

        // Find local max values and positions
#pragma omp for
        for (int y = 0; y < output.rows; y++) {
            const T* row_ptr = output.ptr<T>(y);
            for (int x = 0; x < output.cols; x++) {
                for (int k = 0; k < channels; k++) {
                    double val = static_cast<double>(row_ptr[x][k]);
                    if (val > local_max[thread_id][k]) {
                        local_max[thread_id][k] = val;
                        local_max_pos[thread_id][k].clear();
                        local_max_pos[thread_id][k].emplace_back(y, x);
                    }
                    else if (val == local_max[thread_id][k]) {
                        local_max_pos[thread_id][k].emplace_back(y, x);
                    }
                }
            }
        }
    }

    // Combine local max values and positions
    for (int k = 0; k < channels; k++) {
        for (int t = 0; t < omp_get_max_threads(); t++) {
            if (local_max[t][k] > max[k]) {
                max[k] = local_max[t][k];
                max_pos[k] = local_max_pos[t][k];
            }
            else if (local_max[t][k] == max[k]) {
                max_pos[k].insert(max_pos[k].end(), local_max_pos[t][k].begin(), local_max_pos[t][k].end());
            }
        }
    }

    auto end = std::chrono::high_resolution_clock::now();
    double elapsed_ms = std::chrono::duration<double, std::milli>(end - start).count();
    printf("Elapsed time: %.2f ms\n", elapsed_ms);
}

template<typename T> void printPixelValues(const cv::Mat& image1, const cv::Mat& image2, const std::vector<std::vector<std::pair<int, int>>>& max_pos, int channels) {
    for (int k = 0; k < channels; k++) {
        printf("Input pixel values at max positions for channel %d:\n", k);
        for (const auto& pos : max_pos[k]) {
            T p1 = image1.at<T>(pos.first, pos.second);
            T p2 = image2.at<T>(pos.first, pos.second);
            printf("Position (%d, %d): ", pos.first, pos.second);
            printf("Input Image1: [");
            for (int c = 0; c < channels; c++) {
                printf("%.2f ", static_cast<double>(p1[c]));
            }
            printf("] ");
            printf("Input Image2: [");
            for (int c = 0; c < channels; c++) {
                printf("%.2f ", static_cast<double>(p2[c]));
            }
            printf("]\n");
        }
    }
}

int main(int argc, char* argv[]) {
    auto start = std::chrono::high_resolution_clock::now();
    std::vector<int> formats;

    _putenv("OPENCV_IO_ENABLE_OPENEXR=1");

    cv::Mat image1 = cv::imread(argv[1], cv::IMREAD_UNCHANGED);
    cv::Mat image2 = cv::imread(argv[2], cv::IMREAD_UNCHANGED);


    if (image1.empty()) {
        printf("Could not open or find the image1\n");
        return -1;
    }

    if (image2.empty()) {
        printf("Could not open or find the image2\n");
        return -1;
    }

    cv::Mat output;

    cv::absdiff(image1, image2, output);

    int channels = output.channels();
    int depth = output.depth();

    std::vector<double> max1(channels, 0);
    std::vector<std::vector<std::pair<int, int>>> max_pos(channels);

    if (depth == CV_8U) {
        calculateMax<cv::Vec3b>(output, max1, max_pos, channels);
        printPixelValues<cv::Vec3b>(image1, image2, max_pos, channels);
    }
    else if (depth == CV_16U) {
        calculateMax<cv::Vec3w>(output, max1, max_pos, channels);
        printPixelValues<cv::Vec3w>(image1, image2, max_pos, channels);
    }
    else if (depth == CV_32F) {
        calculateMax<cv::Vec3f>(output, max1, max_pos, channels);
        printPixelValues<cv::Vec3f>(image1, image2, max_pos, channels);
    }

    cv::Scalar mean_values = cv::mean(output);
    printf("Mean: [%.2f, %.2f, %.2f, %.2f]\n", mean_values[0], mean_values[1], mean_values[2], mean_values[3]);

    printf("Max: ");
    for (int i = 0; i < channels; i++) {
        printf("%.2f ", max1[i]);
    }
    printf("\n");

    for (int k = 0; k < channels; k++) {
        printf("Max positions for channel %d: ", k);
        for (const auto& pos : max_pos[k]) {
            printf("(%d, %d) ", pos.first, pos.second);
        }
        printf("\n");
    }

    auto end = std::chrono::high_resolution_clock::now();
    double elapsed_ms = std::chrono::duration<double, std::milli>(end - start).count();
    printf("Elapsed time: %.2f ms\n", elapsed_ms);

    cv::imshow("Display Window", output);
    cv::waitKey(0);

    return 0;
}
