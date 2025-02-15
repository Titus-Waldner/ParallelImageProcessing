//--------------------------------
//Code by Titus Waldner 7881218
// Lab2_Part4
// Parallel Processing - ECE 4530
//10/13/2023
//--------------------------------
//Code distributes an image with halos and then applys a blur/sharpen before merging the image back together
//--------------------------------

#include<iostream>
#include<vector>
#include<mpi.h>
#include<opencv4/opencv2/core/core.hpp>
#include<opencv4/opencv2/highgui/highgui.hpp>

//--------------------------
// apply blur to an image
//--------------------------
void blur(cv::Mat &image, int blurLevel) 
{
    // Create a new Mat
    cv::Mat processedImage(image.rows, image.cols, image.type());

    // Iterate over each row in the image
    for (int iy = 0; iy < image.rows; iy++) 
    {
        // Iterate over each pixel in the current row
        for (int ix = 0; ix < image.cols; ix++) 
        {
            // Define the boundaries of the neighborhood to be considered for blurring
            int ymin = std::max(0, iy - blurLevel); // Upper boundary
            int ymax = std::min(iy + blurLevel, image.rows - 1); // Lower boundary
            int xmin = std::max(0, ix - blurLevel); // Left boundary
            int xmax = std::min(ix + blurLevel, image.cols - 1); // Right boundary

            // Initialize variables to accumulate color values and count the pixels in the neighborhood
            double valB = 0, valG = 0, valR = 0;
            int blurPixelCount = 0;

            // Iterate over the pixels within the defined neighborhood
            for (int iyBlur = ymin; iyBlur <= ymax; iyBlur++) 
            {
                for (int ixBlur = xmin; ixBlur <= xmax; ixBlur++) 
                {
                    // Accumulate the color values of the pixels within the neighborhood
                    valB += image.at<cv::Vec3b>(iyBlur, ixBlur)[0];
                    valG += image.at<cv::Vec3b>(iyBlur, ixBlur)[1];
                    valR += image.at<cv::Vec3b>(iyBlur, ixBlur)[2];
                    // Increment the count of pixels in the neighborhood
                    blurPixelCount++;
                }
            }
            
            // Calculate the average color values for the neighborhood and store them in the processed image
            processedImage.at<cv::Vec3b>(iy, ix)[0] = static_cast<int>(std::floor(valB / blurPixelCount));
            processedImage.at<cv::Vec3b>(iy, ix)[1] = static_cast<int>(std::floor(valG / blurPixelCount));
            processedImage.at<cv::Vec3b>(iy, ix)[2] = static_cast<int>(std::floor(valR / blurPixelCount));
        }
    }
    // Replace the original image
    image = processedImage;
}


//--------------------------
// Sharpen the Image
//--------------------------

void sharpen(cv::Mat &image, int strength) 
{
    // Create a new image
    cv::Mat processedImage(image.rows, image.cols, image.type());

    // Define the Laplacian kernel for sharpening
    cv::Mat kernel = (cv::Mat_<float>(3, 3) << -1, -1, -1, -1, strength, -1, -1, -1, -1);

    // Iterate over each row 
    for (int iy = 0; iy < image.rows; iy++) 
    {
        // Iterate over each pixel in the current row
        for (int ix = 0; ix < image.cols; ix++) 
        {
            // Initialize variables to accumulate the weighted color values
            float valB = 0, valG = 0, valR = 0;

            // Iterate over the cells of the Laplacian kernel
            for (int kY = -1; kY <= 1; kY++)
            {
                for (int kX = -1; kX <= 1; kX++)
                {
                    // Calculate the coordinates of the neighbor pixel
                    int neighborY = std::max(0, std::min(iy + kY, image.rows - 1));
                    int neighborX = std::max(0, std::min(ix + kX, image.cols - 1));
                    
                    // Apply the Laplacian kernel to the neighbor pixel and accumulate the weighted color values
                    valB += image.at<cv::Vec3b>(neighborY, neighborX)[0] * kernel.at<float>(kY + 1, kX + 1);
                    valG += image.at<cv::Vec3b>(neighborY, neighborX)[1] * kernel.at<float>(kY + 1, kX + 1);
                    valR += image.at<cv::Vec3b>(neighborY, neighborX)[2] * kernel.at<float>(kY + 1, kX + 1);
                }
            }

            // Ensure color values are within the valid range and store them in the result image
            processedImage.at<cv::Vec3b>(iy, ix)[0] = static_cast<int>(std::max(0.0f, std::min(255.0f, valB)));
            processedImage.at<cv::Vec3b>(iy, ix)[1] = static_cast<int>(std::max(0.0f, std::min(255.0f, valG)));
            processedImage.at<cv::Vec3b>(iy, ix)[2] = static_cast<int>(std::max(0.0f, std::min(255.0f, valR)));
        }
    }
    
    // Update the original image
    image = processedImage;
}



//--------------------------
// Find local start and stop based on rank and global start and stop
//--------------------------
void parallel_range(int rank, int nproc, int global_start, int global_stop, int &local_start, int &local_stop)
{
	int global_size = global_stop - global_start;
	int local_size = global_size / nproc;
	int local_remainder = global_size % nproc;

    int offset = rank < local_remainder ? rank : local_remainder;
	
	local_start = rank*local_size + offset;
	local_stop = local_start + local_size;
	if(local_remainder > rank) 
    {
        local_stop++;
    }
	local_start += global_start;
	local_stop += global_start;
}

int main(int argc, char **argv)
{
    int m = 100;
    int rank, nproc;
    cv::Mat image, local_image;
    MPI_Comm comm = MPI_COMM_WORLD;
    MPI_Datatype Vec3b;
    
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &nproc);

    MPI_Type_contiguous(3, MPI_UNSIGNED_CHAR, &Vec3b);
    MPI_Type_commit(&Vec3b);
    MPI_Datatype type = Vec3b;

    int size[2];

    if (rank == 0) 
    {
        double startTime = MPI_Wtime();
        image = cv::imread("pic.jpg", 1);

        //--------------------------
        // Send Image Size to all Processors
        //--------------------------

        size[0] = image.rows;
        size[1] = image.cols;
        MPI_Bcast(size, 2, MPI_INT, 0, comm);

        //--------------------------
        // Define local_start and local_stop
        //--------------------------
        
        int local_start, local_stop;
        parallel_range(rank, nproc, 0, size[0], local_start, local_stop);
        local_stop = local_stop + m;
        int local_rows = local_stop - local_start;

        //--------------------------
        // Create Local Images
        //--------------------------
        local_image = cv::Mat(local_rows, size[1], CV_8UC3);
        

        //--------------------------
        // Gather to collect all secondcounts from all ranks
        //--------------------------
        int sendcounts[nproc];
        int senddispls[nproc];
        int recvcount = local_rows * size[1];
        
        MPI_Gather(&recvcount, 1, MPI_INT, sendcounts, 1, MPI_INT, 0, comm);
        
        
        //--------------------------
        // calculate senddispls
        //--------------------------

        senddispls[0] = 0;
        for(int irank=1; irank < nproc; irank++)
        {
            senddispls[irank] = senddispls[irank-1] + sendcounts[irank-1];
            senddispls[irank] -= 2*m * size[1];
        }

        //--------------------------
        // Send Images to All proccessors
        //--------------------------
        MPI_Scatterv(image.data, sendcounts, senddispls, type, local_image.data, recvcount, type, 0, comm);
  
        //--------------------------
        // Apply Blur/Sharpen
        //--------------------------

        sharpen(local_image, 9);
        //blur(local_image, 6);

        std::string window_name1 = "Local- Rank " + std::to_string(rank);
        cv::imshow(window_name1, local_image);
        cv::waitKey(0); // Wait for a key press to close the window

       // cv::imwrite("sub_image_" + std::to_string(rank) + ".jpg", local_image);

        MPI_Gatherv(local_image.data, recvcount, type,image.data, sendcounts, senddispls, type, 0, comm);

        std::string Final = "Final" + std::to_string(rank);
        cv::imshow(window_name1, image);
        cv::waitKey(0); // Wait for a key press to close the window

        //cv::imwrite("merged_image_" + std::to_string(rank) + ".jpg", image);

        //--------------------------
        // Get Run Time
        //--------------------------

        double endTime = MPI_Wtime();
        double elapsedTime = endTime - startTime;
        std::cout << "Wall time on rank 0: " << elapsedTime << " seconds" << std::endl;
    }

    else // all other ranks
    {
        //--------------------------
        // Receive size of images
        //--------------------------
        MPI_Bcast(size, 2, MPI_INT, 0, comm);
        

        //--------------------------
        // Define local_start and local_stop
        //--------------------------
        int local_start, local_stop;
        parallel_range(rank, nproc, 0, size[0], local_start, local_stop);
        if (rank == (nproc-1))
        {
            local_start= local_start- m;
        } 
        else
        {
            local_start = local_start - m; 
            local_stop = local_stop + m; 
        }
        int local_rows = local_stop - local_start;


        
        

        
        //--------------------------
        // Create Local Images
        //--------------------------
        local_image = cv::Mat(local_rows, size[1], CV_8UC3);

        //--------------------------
        // Gather to collect all secondcounts from all ranks
        //--------------------------
        int sendcounts[nproc], senddispls[nproc];
        int recvcount = local_rows * size[1];
        MPI_Gather(&recvcount, 1, MPI_INT, sendcounts, 1, MPI_INT, 0, comm);


        //--------------------------
        // Scatterv rank != 0
        //--------------------------
        MPI_Scatterv(NULL, NULL, NULL, NULL, local_image.data, recvcount, type, 0, comm);

        //--------------------------
        // blurImage/sharpenImage
        //--------------------------

        sharpen(local_image, 9);
        //blur(local_image, 6);

        std::string window_name1 = "Local- Rank " + std::to_string(rank);
        cv::imshow(window_name1, local_image);
        cv::waitKey(0); // Wait for a key press to close the window

        //cv::imwrite("sub_image_" + std::to_string(rank) + ".jpg", local_image);

        //--------------------------
        // Gatherv rank != 0
        //--------------------------
        MPI_Gatherv(local_image.data, recvcount, type,image.data, NULL, NULL, NULL, 0, comm);
        
    }

    MPI_Type_free(&Vec3b);
    MPI_Finalize();
    return 0;
}