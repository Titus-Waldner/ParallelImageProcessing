//--------------------------------
//Code by Titus Waldner 7881218
// Lab2_Part3
// Parallel Processing - ECE 4530
//10/13/2023
//--------------------------------
//Takes an image in and splits them up. Colors them based on rank, then stores them. Images have halos!!!
//--------------------------------

#include<iostream>
#include<vector>
#include<mpi.h>
#include<opencv4/opencv2/core/core.hpp>
#include<opencv4/opencv2/highgui/highgui.hpp>

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
        

        // Color Pixel depending on rank
        for (int row = 0; row < local_image.rows; row++) {
            for (int col = 0; col < local_image.cols; col++) {
                cv::Vec3b& pixel = local_image.at<cv::Vec3b>(row, col);
                pixel[0] = 255; // Set the blue channel to 255
            }
        }

        std::string window_name1 = "Local- Rank " + std::to_string(rank);
        cv::imshow(window_name1, local_image);
        cv::waitKey(0); // Wait for a key press to close the window

        

        MPI_Gatherv(local_image.data, recvcount, type,image.data, sendcounts, senddispls, type, 0, comm);

        std::string Final = "Final" + std::to_string(rank);
        cv::imshow(window_name1, image);
        cv::waitKey(0); // Wait for a key press to close the window
    }
    else
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
        if (rank == 1)
        {
            local_start = local_start - m; 
            local_stop = local_stop + m; 
        } 
        if (rank == 2)
        {
            local_start= local_start- m;
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
        
        if(rank == 1)
        {
            // Color Pixel depending on rank
            for (int row = 0; row < local_image.rows; row++) {
                for (int col = 0; col < local_image.cols; col++) {
                    cv::Vec3b& pixel = local_image.at<cv::Vec3b>(row, col);
                    pixel[1] = 255; // Set the Green channel to 255
                }
            }
        }
        else
        {
            // Color Pixel depending on rank
            for (int row = 0; row < local_image.rows; row++) {
                for (int col = 0; col < local_image.cols; col++) {
                    cv::Vec3b& pixel = local_image.at<cv::Vec3b>(row, col);
                    pixel[2] = 255; // Set the Red channel to 255
                }
            }
        }
        
        std::string window_name1 = "Local- Rank " + std::to_string(rank);
        cv::imshow(window_name1, local_image);
        cv::waitKey(0); // Wait for a key press to close the window
        MPI_Gatherv(local_image.data, recvcount, type,image.data, sendcounts, senddispls, type, 0, comm);
        
    }

    MPI_Type_free(&Vec3b);
    MPI_Finalize();
    return 0;
}