//--------------------------------
//Code by Titus Waldner 7881218
// Lab2_Part2
// Parallel Processing - ECE 4530
//10/13/2023
//--------------------------------
//Code shows how to properly distrubte data with halos and restore them using scatterv and gatherv
//Code will dynamically increase starting vector based on number of processors and all halo will increases as halo variable increases
//--------------------------------

#include <iostream>
#include <vector>
#include <mpi.h>

void parallelRange(int globalstart, int globalstop, int irank, int nproc, int& localstart, int& localstop, int& localcount)
{
    int nrows = globalstop - globalstart + 1;
    int divisor = nrows / nproc;
    int remainder = nrows % nproc;
    int offset;
    if (irank < remainder)
        offset = irank;
    else
        offset = remainder;

    localstart = irank * divisor + globalstart + offset;
    localstop = localstart + divisor - 1;
    if (remainder > irank)
        localstop += 1;
    localcount = localstop - localstart + 1;
}

bool areVectorsEqual(const std::vector<int>& vec1, const std::vector<int>& vec2)
{
    if (vec1.size() != vec2.size())
    {
        return false;
    }

    for (size_t i = 0; i < vec1.size(); i++)
    {
        if (vec1[i] != vec2[i])
        {
            return false;
        }
    }

    return true;
}

int main(int argc, char** argv)
{
    MPI_Init(&argc, &argv);

    int rank, nproc;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nproc);

    int halo = 5;
    int globalStart = 0;
    int globalStop = ((nproc*5)-1);

    int localStart;
    int localStop;
    int localCount;
    std::vector<int> localstart(nproc), localstop(nproc), localcount(nproc);
    std::vector<int> localData(5);
    std::vector<int> myVector(globalStop+1);
    std::vector<int> gatheredData(globalStop+1);

    if (rank == 0) // rank 0 **************************************************************************************************************************************************
    {
        //--------------------------
		// Create myVector and Fill Data
		//--------------------------
        for (int i = 0; i < (nproc*5); i = (i+5))
        {
                myVector[i]   = (i/5);
                myVector[i+1] = (i/5);
                myVector[i+2] = (i/5);
                myVector[i+3] = (i/5);
                myVector[i+4] = (i/5);
        }

        //--------------------------
        // Print Out Original Vector
        //--------------------------
        std::cout <<"Full Vector: ";
        for (int i = 0; i < myVector.size(); i++)
        {
           std::cout <<myVector[i] << " ";
        }
        std::cout << std::endl;
       
        std::cout << "LeveL: " << halo << std::endl;
        std::cout << std::endl;

        //--------------------------
		// Determine what to send
		//--------------------------

        for (unsigned int irank = 0; irank < nproc; irank++)
		{
			parallelRange(globalStart, globalStop, irank, nproc, localstart[irank], localstop[irank], localcount[irank]);
            //--------------------------
            // Add Halos
            //--------------------------  
            if(irank > 0)
            {
                localstart[irank] = (localstart[irank]-halo);
                
            }
            if(irank < nproc-1)
            {
                localstop[irank] = (localstop[irank]+halo);
            }
            if(irank < nproc-1 && irank > 0)
            {
                localcount[irank] = (localcount[irank]+2*halo);
            }
            else
            {
                localcount[irank] = (localcount[irank]+halo);
            }
		}
        // Allocate memory for localData
        localData.resize(localcount[0]);

        //--------------------------
        // Scatterv Rank 0
        //--------------------------
        MPI_Scatterv(myVector.data(), localcount.data(), localstart.data(), MPI_INT, localData.data(), localcount[0], MPI_INT, 0, MPI_COMM_WORLD);

        //--------------------------
        // Print Out Local Values
        //--------------------------
        std::cout <<"Rank: "<< rank <<": ";
        for (int i = 0; i < localData.size(); i++)
        {
            std::cout <<localData[i] << " ";
        }
        std::cout << std::endl;
        
        //--------------------------
        // Gatherv Rank 0
        //--------------------------
        MPI_Gatherv(localData.data(), localcount[0], MPI_INT, gatheredData.data(), localcount.data(), localstart.data(), MPI_INT, 0, MPI_COMM_WORLD);

        MPI_Barrier(MPI_COMM_WORLD);
        //--------------------------
        // Print Out Gathered Vector
        //--------------------------
        std::cout << std::endl;
        std::cout <<"Gathered Vector: ";
        for (int i = 0; i < gatheredData.size(); i++)
        {
           std::cout <<gatheredData[i] << " ";
        }
        std::cout << std::endl;

        //--------------------------
        // Check if myVector and Gathered vector are equal
        //--------------------------
        bool vectorsEqual = areVectorsEqual(myVector, gatheredData);
        if (rank == 0) {
            if (vectorsEqual) {
                std::cout << "myVector and gatheredData are equal: true" << std::endl;
            } else {
                std::cout << "myVector and gatheredData are equal: false" << std::endl;
            }
        }
    }


    else //all other ranks**************************************************************************************************************************************************
    {

        parallelRange(globalStart, globalStop, rank, nproc, localStart, localStop, localCount);
        //--------------------------
        // Add Halos
        //--------------------------
        if(rank < nproc-1)
        {
            localCount = (localCount+2*halo);
        }
        else
        {
            localCount = (localCount+halo);
        }
        
       // Allocate memory for localData
        localData.resize(localCount);

        //--------------------------
        // Scatterv Ranks != 0
        //--------------------------
        MPI_Scatterv(NULL, NULL, NULL, MPI_INT, localData.data(), localCount, MPI_INT, 0, MPI_COMM_WORLD);

        //--------------------------
        // Print Out Local Values
        //--------------------------
        std::cout <<"Rank: "<< rank <<": ";
        for (int i = 0; i < localData.size(); i++)
        {
            std::cout <<localData[i] << " ";
        }
        std::cout << std::endl;
        
        
        //--------------------------
        // Gatherv Ranks != 0
        //--------------------------
        MPI_Gatherv(localData.data(), localCount, MPI_INT, NULL, NULL, NULL, MPI_INT, 0, MPI_COMM_WORLD);

        MPI_Barrier(MPI_COMM_WORLD);

    }
    MPI_Finalize();
}
