#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <pthread.h>
#include <time.h>
#include <stdint.h>
#include <immintrin.h>
#include <unistd.h>

#define BLOCK_SIZE 32
#define CACHE_LINE 64
#define ALIGNMENT 32

// Check AVX2 support
#ifdef __AVX2__
    #define USE_AVX2
#endif

typedef struct {
    int thread_id;
    int num_threads;
    int width;
    int height;
    int *input;
    int *output;
    char pad[CACHE_LINE];
} ThreadArgs;

pthread_barrier_t barrier;

// Read PGM image
int* read_pgm(const char* filename, int* width, int* height, int* maxval) {
    FILE* fp = fopen(filename, "r");
    if(!fp) {
        perror("Unable to open file");
        exit(1);
    }
    
    char format[3];
    if (fscanf(fp, "%s", format) != 1) {
        fclose(fp);
        fprintf(stderr, "Error reading format\n");
        exit(1);
    }
    
    if(format[0] != 'P' || format[1] != '2') {
        fclose(fp);
        fprintf(stderr, "Invalid image format (must be 'P2')\n");
        exit(1);
    }
    
    if (fscanf(fp, "%d %d", width, height) != 2) {
        fclose(fp);
        fprintf(stderr, "Error reading dimensions\n");
        exit(1);
    }
    
    if (fscanf(fp, "%d", maxval) != 1) {
        fclose(fp);
        fprintf(stderr, "Error reading maxval\n");
        exit(1);
    }
    
    int size = (*width) * (*height);
    int* image = aligned_alloc(ALIGNMENT, size * sizeof(int));
    if (!image) {
        fclose(fp);
        fprintf(stderr, "Failed to allocate memory for image\n");
        exit(1);
    }
    
    for(int i = 0; i < size; i++) {
        if (fscanf(fp, "%d", &image[i]) != 1) {
            free(image);
            fclose(fp);
            fprintf(stderr, "Error reading pixel data\n");
            exit(1);
        }
    }
    
    fclose(fp);
    return image;
}

// Write PGM image
void write_pgm(const char* filename, int* image, int width, int height, int maxval) {
    FILE* fp = fopen(filename, "w");
    if(!fp) {
        perror("Unable to open file for writing");
        return;
    }
    
    fprintf(fp, "P2\n");
    fprintf(fp, "%d %d\n", width, height);
    fprintf(fp, "%d\n", maxval);
    
    for(int i = 0; i < height; i++) {
        for(int j = 0; j < width; j++) {
            fprintf(fp, "%d ", image[i * width + j]);
        }
        fprintf(fp, "\n");
    }
    
    fclose(fp);
}

// Block processing
static void process_block_optimized(const int *input, int *output, 
                                  int x0, int y0, int block_w, int block_h,
                                  int width, int height) {
#ifdef USE_AVX2
    const int vec_size = 8; 
    
    for (int y = y0; y < y0 + block_h && y < height; y++) {
        if (y == 0 || y == height-1) {
            memcpy(&output[y * width + x0], 
                   &input[y * width + x0], 
                   block_w * sizeof(int));
            continue;
        }

        int x = x0;
        // Handle left boundary
        if (x0 == 0) {
            output[y * width] = input[y * width];
            x++;
        }

        // Use AVX2 to process the main part
        for (; x < x0 + block_w - vec_size && x < width - 1; x += vec_size) {
            if (x + vec_size < width - 1) {
                __m256i sum = _mm256_setzero_si256();
                
                // Previous row
                sum = _mm256_add_epi32(sum, _mm256_loadu_si256((__m256i*)&input[(y-1) * width + x - 1]));
                sum = _mm256_add_epi32(sum, _mm256_loadu_si256((__m256i*)&input[(y-1) * width + x]));
                sum = _mm256_add_epi32(sum, _mm256_loadu_si256((__m256i*)&input[(y-1) * width + x + 1]));
                
                // Current row
                sum = _mm256_add_epi32(sum, _mm256_loadu_si256((__m256i*)&input[y * width + x - 1]));
                sum = _mm256_add_epi32(sum, _mm256_loadu_si256((__m256i*)&input[y * width + x + 1]));
                
                // Next row
                sum = _mm256_add_epi32(sum, _mm256_loadu_si256((__m256i*)&input[(y+1) * width + x - 1]));
                sum = _mm256_add_epi32(sum, _mm256_loadu_si256((__m256i*)&input[(y+1) * width + x]));
                sum = _mm256_add_epi32(sum, _mm256_loadu_si256((__m256i*)&input[(y+1) * width + x + 1]));
                
                // Divide by 8
                __m256i result = _mm256_srli_epi32(sum, 3);
                _mm256_storeu_si256((__m256i*)&output[y * width + x], result);
            }
        }

        // Handle remaining pixels
        for (; x < x0 + block_w && x < width; x++) {
            if (x == width-1) {
                output[y * width + x] = input[y * width + x];
            } else {
                int sum = 0;
                sum += input[(y-1) * width + (x-1)];
                sum += input[(y-1) * width + x];
                sum += input[(y-1) * width + (x+1)];
                sum += input[y * width + (x-1)];
                sum += input[y * width + (x+1)];
                sum += input[(y+1) * width + (x-1)];
                sum += input[(y+1) * width + x];
                sum += input[(y+1) * width + (x+1)];
                output[y * width + x] = sum / 8;
            }
        }
    }
#else
    // Non-AVX2 implementation
    for (int y = y0; y < y0 + block_h && y < height; y++) {
        for (int x = x0; x < x0 + block_w && x < width; x++) {
            if (x == 0 || x == width-1 || y == 0 || y == height-1) {
                output[y * width + x] = input[y * width + x];
                continue;
            }
            
            int sum = 0;
            sum += input[(y-1) * width + (x-1)];
            sum += input[(y-1) * width + x];
            sum += input[(y-1) * width + (x+1)];
            sum += input[y * width + (x-1)];
            sum += input[y * width + (x+1)];
            sum += input[(y+1) * width + (x-1)];
            sum += input[(y+1) * width + x];
            sum += input[(y+1) * width + (x+1)];
            output[y * width + x] = sum / 8;
        }
    }
#endif
}

void* blur_thread(void *arg) {
    ThreadArgs *args = (ThreadArgs*)arg;
    const int width = args->width;
    const int height = args->height;
    
    const int rows_per_thread = (height + args->num_threads - 1) / args->num_threads;
    const int start_row = args->thread_id * rows_per_thread;
    const int end_row = (args->thread_id == args->num_threads - 1) ? 
                       height : start_row + rows_per_thread;

    #pragma GCC ivdep
    for (int y0 = start_row; y0 < end_row; y0 += BLOCK_SIZE) {
        for (int x0 = 0; x0 < width; x0 += BLOCK_SIZE) {
            __builtin_prefetch(&args->input[(y0 + BLOCK_SIZE) * width + x0], 0, 3);
            process_block_optimized(args->input, args->output, x0, y0,
                                  BLOCK_SIZE, BLOCK_SIZE, width, height);
        }
    }
    
    pthread_barrier_wait(&barrier);
    return NULL;
}

void single_blur(int *input, int *output, int width, int height, int num_threads) {
    pthread_t *threads = malloc(num_threads * sizeof(pthread_t));
    ThreadArgs *args = aligned_alloc(ALIGNMENT, num_threads * sizeof(ThreadArgs));
    
    pthread_barrier_init(&barrier, NULL, num_threads);
    
    for(int i = 0; i < num_threads; i++) {
        args[i].thread_id = i;
        args[i].num_threads = num_threads;
        args[i].width = width;
        args[i].height = height;
        args[i].input = input;
        args[i].output = output;
        
        if (pthread_create(&threads[i], NULL, blur_thread, &args[i]) != 0) {
            fprintf(stderr, "Failed to create thread %d\n", i);
            for(int j = 0; j < i; j++) {
                pthread_join(threads[j], NULL);
            }
            free(threads);
            free(args);
            exit(1);
        }
    }
    
    for(int i = 0; i < num_threads; i++) {
        pthread_join(threads[i], NULL);
    }
    
    pthread_barrier_destroy(&barrier);
    free(threads);
    free(args);
}

void blur_image(int *image, int width, int height, int iterations, int num_threads) {
    int *buffer1 = aligned_alloc(ALIGNMENT, width * height * sizeof(int));
    if (!buffer1) {
        fprintf(stderr, "Failed to allocate buffer1\n");
        exit(1);
    }
    
    int *buffer2 = aligned_alloc(ALIGNMENT, width * height * sizeof(int));
    if (!buffer2) {
        free(buffer1);
        fprintf(stderr, "Failed to allocate buffer2\n");
        exit(1);
    }
    
    memcpy(buffer1, image, width * height * sizeof(int));
    
    for(int i = 0; i < iterations; i++) {
        if(i % 2 == 0) {
            single_blur(buffer1, buffer2, width, height, num_threads);
        } else {
            single_blur(buffer2, buffer1, width, height, num_threads);
        }
    }
    
    if(iterations % 2 == 0) {
        memcpy(image, buffer1, width * height * sizeof(int));
    } else {
        memcpy(image, buffer2, width * height * sizeof(int));
    }
    
    free(buffer1);
    free(buffer2);
}

void test_thread_performance(const char* filename, int max_threads) {
    int width, height, maxval;
    int* original = read_pgm(filename, &width, &height, &maxval);
    if (!original) {
        return;
    }
    
    printf("\nThread Performance Test Results:\n");
    printf("================================\n");
    printf("Image size: %dx%d\n", width, height);
    printf("Threads\tTime(s)\tSpeedup\tEfficiency\n");
    
    double base_time = 0;
    
    for(int num_threads = 1; num_threads <= max_threads; num_threads *= 2) {
        int* image = aligned_alloc(ALIGNMENT, width * height * sizeof(int));
        if (!image) {
            free(original);
            return;
        }
        memcpy(image, original, width * height * sizeof(int));
        
        struct timespec start, end;
        clock_gettime(CLOCK_MONOTONIC, &start);
        
        blur_image(image, width, height, 20, num_threads);
        
        clock_gettime(CLOCK_MONOTONIC, &end);
        double time_spent = (end.tv_sec - start.tv_sec) + 
                          (end.tv_nsec - start.tv_nsec) / 1e9;
        
        if (num_threads == 1) {
            base_time = time_spent;
        }
        
        double speedup = base_time / time_spent;
        double efficiency = speedup / num_threads;
        
        printf("%d\t%.4f\t%.4f\t%.4f\n", 
               num_threads, time_spent, speedup, efficiency);
        
        // Create output filename with thread count
        char output_filename[256];
        snprintf(output_filename, sizeof(output_filename), 
                "blur_thread_%d.pgm", num_threads);
        
        // Save the output for each thread configuration
        write_pgm(output_filename, image, width, height, maxval);
        
        free(image);
    }
    
    free(original);
}

int main(int argc, char* argv[]) {
    if(argc != 2) {
        fprintf(stderr, "Usage: %s <input_image.pgm>\n", argv[0]);
        return 1;
    }
    
    int max_threads = sysconf(_SC_NPROCESSORS_ONLN);
    printf("Detected %d CPU cores\n", max_threads);
    
    test_thread_performance(argv[1], max_threads);
    
    return 0;
}