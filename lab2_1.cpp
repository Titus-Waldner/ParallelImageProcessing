//--------------------------------
//Code by Titus Waldner 7881218
// Lab2_Part1
// Parallel Processing - ECE 4530
//10/13/2023
//--------------------------------
//Code takes an image in and applys a blur to it
//--------------------------------

#include<iostream>
#include<vector>
#include<opencv4/opencv2/core/core.hpp>
#include<opencv4/opencv2/highgui/highgui.hpp>

//--------------------------
// apply blur to an image
//--------------------------
void blurImage(cv::Mat &image, int blurLevel) 
{
    cv::Mat processedImage(image.rows, image.cols, image.type());

    for (int iy = 0; iy < image.rows; iy++) 
    {
        for (int ix = 0; ix < image.cols; ix++) 
        {
            int ymin = std::max(0, iy - blurLevel);
            int ymax = std::min(iy + blurLevel, image.rows - 1);

            int xmin = std::max(0, ix - blurLevel);
            int xmax = std::min(ix + blurLevel, image.cols - 1);

            double valB = 0, valG = 0, valR = 0;
            int blurPixelCount = 0;

            for (int iyBlur = ymin; iyBlur <= ymax; iyBlur++) 
            {
                for (int ixBlur = xmin; ixBlur <= xmax; ixBlur++) 
                {
                    valB += image.at<cv::Vec3b>(iyBlur, ixBlur)[0];
                    valG += image.at<cv::Vec3b>(iyBlur, ixBlur)[1];
                    valR += image.at<cv::Vec3b>(iyBlur, ixBlur)[2];
                    blurPixelCount++;
                }
            }
            processedImage.at<cv::Vec3b>(iy, ix)[0] = static_cast<int>(std::floor(valB / blurPixelCount));
            processedImage.at<cv::Vec3b>(iy, ix)[1] = static_cast<int>(std::floor(valG / blurPixelCount));
            processedImage.at<cv::Vec3b>(iy, ix)[2] = static_cast<int>(std::floor(valR / blurPixelCount));
        }
    }
    image = processedImage;
}




using namespace std;

int main(int argc, char** argv)
{
    //--------------------------
    // Get Parameters from commandline
    //--------------------------
    cv::Mat image;
    if (argc < 3)
    {
        std::cerr << "Command line parameters must include the filename and the blur level" << std::endl;
    }
    
    string filename = argv[1];
    int blurlevel = atoi(argv[2]);
    //--------------------------
    // Create Local Images
    //--------------------------
    image = cv::imread(filename,1);
    
    if(! image.data )
    {
        std::cerr <<  "Could not open or find the image" << std::endl;
        return -1;
    }
    //--------------------------
    // Show Unblurred Image
    //--------------------------
   // std::string window_name1 = "Starter";
    //cv::imshow(window_name1, image);
   // cv::waitKey(0); // Wait for a key press to close the window

    //--------------------------
    // Blur Image
    //--------------------------
    blurImage(image, 5);

    //--------------------------
    // Show Blurred Image
    //--------------------------
    std::string window_name2 = "Blurred";
    cv::imshow(window_name2, image);
    cv::waitKey(0); // Wait for a key press to close the window
}
