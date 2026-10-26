#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <mpi.h>

int* Read_PGM(const char* filename, int* width, int* height, int* max_val);
void Write_PGM(const char* filename, int width, int height, int max_val, const int* image);
void Blur(int width, int local_height, int* local_image, int* local_buffer, int rank, int size, MPI_Comm comm);

// Read image from PGM file
int* Read_PGM(const char* filename, int* width, int* height, int* max_val) {
    FILE *file = fopen(filename, "r");
    if (!file) {
        perror("Error opening file");
        return NULL;
    }

    char format[3];
    if (fscanf(file, "%s", format) != 1) {
        fclose(file);
        return NULL;
    }

    if (format[0] != 'P' || format[1] != '2') {
        fclose(file);
        return NULL;
    }

    if (fscanf(file, "%d %d %d", width, height, max_val) != 3) {
        fclose(file);
        return NULL;
    }

    int* image = (int*)malloc((*width) * (*height) * sizeof(int));
    if (!image) {
        fclose(file);
        return NULL;
    }

    for (int i = 0; i < (*width) * (*height); i++) {
        if (fscanf(file, "%d", &image[i]) != 1) {
            free(image);
            fclose(file);
            return NULL;
        }
    }

    fclose(file);
    return image;
}

// Write image to PGM file
void Write_PGM(const char* filename, int width, int height, int max_val, const int* image) {
    FILE *file = fopen(filename, "w");
    if (!file) {
        perror("Error opening output file");
        return;
    }

    fprintf(file, "P2\n");
    fprintf(file, "%d %d\n", width, height);
    fprintf(file, "%d\n", max_val);

    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            fprintf(file, "%d ", image[i * width + j]);
        }
        fprintf(file, "\n");
    }

    fclose(file);
}

// Blur image using MPI
void Blur(int width, int local_height, int* local_image, int* local_buffer, 
          int rank, int size, MPI_Comm comm) {
    int* top_ghost = (int*)malloc(width * sizeof(int));
    int* bottom_ghost = (int*)malloc(width * sizeof(int));

    if (!top_ghost || !bottom_ghost) {
        printf("Failed to allocate ghost rows\n");
        MPI_Abort(comm, 1);
    }

    for (int it = 0; it < 20; it++) {
        MPI_Status status;

        // Exchange boundary data
        if (rank > 0) {
            MPI_Sendrecv(local_image, width, MPI_INT, rank - 1, 0,
                        top_ghost, width, MPI_INT, rank - 1, 1,
                        comm, &status);
        }

        if (rank < size - 1) {
            MPI_Sendrecv(local_image + (local_height-1) * width, width, MPI_INT, rank + 1, 1,
                        bottom_ghost, width, MPI_INT, rank + 1, 0,
                        comm, &status);
        }

        // Blur image
        for (int i = 0; i < local_height; i++) {
            if ((rank == 0 && i == 0)||(rank == size - 1 && i == local_height - 1)) {
                continue;
            }
            for (int j = 1; j < width - 1; j++) {
                int sum = 0;
                int count = 0;

                // Previous line
                int* top_row = (i > 0) ? &local_image[(i-1)*width] : top_ghost;
                sum += top_row[j-1] + top_row[j] + top_row[j+1];
                count += 3;
                // Current line
                sum += local_image[i*width + j-1] + local_image[i*width + j+1];
                count += 2;

                // Next line

                int* bottom_row = (i < local_height - 1) ? &local_image[(i+1)*width] : bottom_ghost;
                sum += bottom_row[j-1] + bottom_row[j] + bottom_row[j+1];
                count += 3;
                

                local_buffer[i*width + j] = sum / count;
            }
        }

        // Keep the boundary unchanged
        for (int i = 0; i < local_height; i++) {
            local_buffer[i * width] = local_image[i * width];
            local_buffer[i * width + width - 1] = local_image[i * width + width - 1];
        }
        if (rank == 0) {
            memcpy(local_buffer, local_image, width * sizeof(int));
        }
        if (rank == size - 1) {
            memcpy(local_buffer + (local_height - 1) * width,
                   local_image + (local_height - 1) * width,
                   width * sizeof(int));
        }

        // Exchange buffer
        int* temp = local_image;
        local_image = local_buffer;
        local_buffer = temp;

        // Synchronize all processes
        MPI_Barrier(comm);
    }

    free(top_ghost);
    free(bottom_ghost);
}

int main(int argc, char** argv) {
    int rank, size;
    int width, height, max_val;
    int* image = NULL;
    double start_time, end_time;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Main process reads the image
    if (rank == 0) {
        image = Read_PGM("im.pgm", &width, &height, &max_val);
        if (!image) {
            printf("Error reading image\n");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        printf("Image size: %dx%d\n", width, height);
    }

    // Broadcast image size
    MPI_Bcast(&width, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&height, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&max_val, 1, MPI_INT, 0, MPI_COMM_WORLD);

    // Calculate number of rows for each process
    int rows_per_proc = height / size;
    int remainder = height % size;
    int local_height = rows_per_proc + (rank < remainder ? 1 : 0);

    // Allocate local memory
    int* local_image = (int*)malloc(local_height * width * sizeof(int));
    int* local_buffer = (int*)malloc(local_height * width * sizeof(int));
    if (!local_image || !local_buffer) {
        printf("Rank %d: Failed to allocate memory\n", rank);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    // Calculate send count and displacement
    int* sendcounts = NULL;
    int* displs = NULL;
    if (rank == 0) {
        sendcounts = (int*)malloc(size * sizeof(int));
        displs = (int*)malloc(size * sizeof(int));
        if (!sendcounts || !displs) {
            printf("Failed to allocate sendcounts/displs\n");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }

        int offset = 0;
        for (int i = 0; i < size; i++) {
            // Basic Row Count
            int rows = rows_per_proc + (i < remainder ? 1 : 0);
        
            displs[i] = offset * width;
            
            sendcounts[i] = rows * width;
            
            // Update the offset considering only the actual data rows, excluding the ghost rows
            offset += rows_per_proc + (i < remainder ? 1 : 0);
        }
    }
    
    // Distribute data
    MPI_Scatterv(image, sendcounts, displs, MPI_INT,
                 local_image, local_height * width, MPI_INT,
                 0, MPI_COMM_WORLD);

    // Perform blurring
    start_time = MPI_Wtime();
    Blur(width, local_height, local_image, local_buffer, rank, size, MPI_COMM_WORLD);
    end_time = MPI_Wtime();

    // Recalculate send counts and displacements for gathering
    if (rank == 0) {
        int offset = 0;
        for (int i = 0; i < size; i++) {
            // Calculate actual data rows (excluding ghost rows)
            int rows = rows_per_proc + (i < remainder ? 1 : 0);
            sendcounts[i] = rows * width;
            displs[i] = offset * width;
            offset += rows;
        }
        for (size_t ijj = 0; ijj < size; ijj++)
        {
            printf("sendcounts[%d]: %d\n", ijj, sendcounts[ijj]/4000);
            printf("displs[%d]: %d\n", ijj, displs[ijj]/4000);
        }
        
    }

    // Calculate the actual number of rows this process handles (excluding ghost rows)
    int actual_rows = rows_per_proc + (rank < remainder ? 1 : 0);

    // Gather the processed data back to root
    MPI_Gatherv(
        local_image,  
        actual_rows * width,                   
        MPI_INT,                               
        image,                                 
        sendcounts,                            
        displs,                                
        MPI_INT,                               
        0,                                     
        MPI_COMM_WORLD                        
    );

    // Print results
    if (rank == 0) {
        Write_PGM("blur_mpi.pgm", width, height, max_val, image);
        printf("Processing time: %.4f seconds\n", end_time - start_time);
        free(image);
        free(sendcounts);
        free(displs);
    }

    // Clean up memory
    free(local_image);
    free(local_buffer);

    MPI_Finalize();
    return 0;
}