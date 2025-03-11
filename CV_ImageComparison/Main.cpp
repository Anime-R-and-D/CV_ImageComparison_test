#include <opencv2/opencv.hpp>
#include <iostream>


int main() {
    //two image inputs
    cv::Mat image1 = cv::imread("image1.png", cv::IMREAD_UNCHANGED);
    cv::Mat image2 = cv::imread("image2.png", cv::IMREAD_UNCHANGED);

	

    if (image1.empty()) {
        std::cout << "Could not open or find the image1" << std::endl;
        return -1;
    }

    if (image2.empty()) {
        std::cout << "Could not open or find the image2" << std::endl;
        return -1;
    }

	cv::Mat output;

	//computes the absolute difference between the two images
	cv::absdiff(image1, image2, output);


	
    int channels = output.channels();
    int depth = output.depth();

    std::vector<double> max(channels, 0);
    std::vector<double> avg(channels, 0);

	// Calculate max and avg, pixel is Vec3b, Vec3w, or Vec3f based on depth
    for (int i = 0; i < output.rows; i++) {
        for (int j = 0; j < output.cols; j++) {
            if (depth == CV_8U) {
                cv::Vec3b pixel = output.at<cv::Vec3b>(i, j);
                for (int k = 0; k < channels; k++) {
                    avg[k] += pixel[k];
                    if (pixel[k] > max[k]) {
                        max[k] = pixel[k];
                    }
                }
            }
            else if (depth == CV_16U) {
                cv::Vec3w pixel = output.at<cv::Vec3w>(i, j);
                for (int k = 0; k < channels; k++) {
                    avg[k] += pixel[k];
                    if (pixel[k] > max[k]) {
                        max[k] = pixel[k];
                    }
                }
            }
            else if (depth == CV_32F) {
                cv::Vec3f pixel = output.at<cv::Vec3f>(i, j);
                for (int k = 0; k < channels; k++) {
                    avg[k] += pixel[k];
                    if (pixel[k] > max[k]) {
                        max[k] = pixel[k];
                    }
                }
            }
        }
    }
	for (int i = 0; i < channels; i++) {
		avg[i] /= output.rows * output.cols;
	}


    std::cout << "Max: ";
    for (int i = 0; i < channels; i++) {
        std::cout << max[i] << " ";
    }
    std::cout << std::endl;

    std::cout << "Avg: ";
    for (int i = 0; i < channels; i++) {
        std::cout << avg[i] << " ";
    }
    std::cout << std::endl;

	
	/*int t = 0;
	
        if (output.depth() == CV_8U) {
        t = 255;
        }
        else if (output.depth() == CV_16U) {
        t = 65535;
        }
        else if (output.depth() == CV_32F) {
        t = 1;
        }
		
	
	std::cout << t << std::endl;*/
	
     

	cv::imshow("Display Window", output);
    cv::waitKey(0);

    return 0;
}
