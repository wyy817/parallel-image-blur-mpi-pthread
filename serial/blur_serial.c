#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// Image blurring function
void Blur(int width, int height, int *image, int *buffer, int iterations) {
    for (int it = 0; it < iterations; it++) {
        for (int i = 1; i < height - 1; i++) {
            for (int j = 1; j < width - 1; j++) {
                int index = i * width + j;
                int top = (i - 1) * width;
                int middle = i * width;
                int bottom = (i + 1) * width;

                // ensure：only the mean of the values of the eight pixel points around the centre pixel point is calculated
                buffer[index] = (
                    image[top + j - 1] + image[top + j] + image[top + j + 1] +
                    image[middle + j - 1] + image[middle + j + 1] +
                    image[bottom + j - 1] + image[bottom + j] + image[bottom + j + 1]
                ) / 8;
            }
        }

        // Boundary pixels remain unchanged
        memcpy(buffer, image, width * sizeof(int));
        memcpy(buffer + (height - 1) * width, image + (height - 1) * width, width * sizeof(int));
        for (int i = 0; i < height; i++) {
            buffer[i * width] = image[i * width];
            buffer[i * width + (width - 1)] = image[i * width + (width - 1)];
        }

        // Exchange of buffer and source data
        int *temp = image;
        image = buffer;
        buffer = temp;
    }
}


// Reading PGM images
int *Read_PGM(const char *file_name, int *width, int *height, int *max_val) {
    FILE *file = fopen(file_name, "r");
    if (!file) {
        perror("Error opening file");
        exit(EXIT_FAILURE);
    }

    char format[3];
    fscanf(file, "%s", format);
    if (format[0] != 'P' || format[1] != '2') {
        fprintf(stderr, "Unsupported file format\n");
        exit(EXIT_FAILURE);
    }

    fscanf(file, "%d %d %d", width, height, max_val);

    int *image = (int *)malloc((*width) * (*height) * sizeof(int));
    for (int i = 0; i < (*width) * (*height); i++) {
        fscanf(file, "%d", &image[i]);
    }

    fclose(file);
    return image;
}

// Writing PGM images
void Write_PGM(const char *file_name, int width, int height, int max_val, int *image) {
    FILE *file = fopen(file_name, "w");
    if (!file) {
        perror("Error opening file");
        exit(EXIT_FAILURE);
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

// main function
int main() {
    int width, height, max_val;

    int *image = Read_PGM("im.pgm", &width, &height, &max_val);

    // Distribution buffer
    int *buffer = (int *)malloc(width * height * sizeof(int));
    if (!buffer) {
        perror("Error allocating buffer");
        free(image);
        exit(EXIT_FAILURE);
    }

    // Performs 20 blurring
    Blur(width, height, image, buffer, 20);
    Write_PGM("blur.pgm", width, height, max_val, image);

    // free up memory
    free(image);
    free(buffer);

    return 0;
}
