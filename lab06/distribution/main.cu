#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "distribution.cuh"

typedef enum {
    METHOD_NAIVE,
    METHOD_BATCH,
    METHOD_UNKNOWN
} MethodType;

MethodType get_method(const char* str) {
    if (strcmp(str, "naive") == 0) return METHOD_NAIVE;
    if (strcmp(str, "batch") == 0) return METHOD_BATCH;
    return METHOD_UNKNOWN;
}

int main(int argc, char** argv)
{
    if (argc != 4) {
        printf("Usage: %s <n> <method> <save>\n", argv[0]);
        printf("Example: %s 1000 naive 1\n", argv[0]);
        return 1;
    }

    int n = atoi(argv[1]);
    char *method_str = argv[2];
    int save = atoi(argv[3]);

    const float TAU = 2.5f;
    const float DT = 0.01f;

    float *x = NULL;
    switch (get_method(method_str)) {
        case METHOD_NAIVE:
            x = generate_naive(n, TAU, DT);
            break;

        case METHOD_BATCH:
            x = generate_batchrand(n, TAU, DT);
            break;

        default:
            printf("Pick method from [naive, batch]\n");
            return 1;
    }

    if (save && x != NULL) {
        FILE *f = fopen("data/results_distribution.bin", "wb");
        if (f) {
            fwrite(x, sizeof(float), n, f); 
            fclose(f);
        }
    }

    return 0;
}