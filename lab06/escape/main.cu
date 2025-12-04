#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "escape.cuh"

int main(int argc, char** argv)
{
    if (argc != 3) {
        printf("Usage: %s <n> <save>\n", argv[0]);
        printf("Example: %s 1000 1\n", argv[0]);
        return 1;
    }

    int n = atoi(argv[1]);
    int save = atoi(argv[2]);

    const float DT = 0.0001f;
    const int T = 50000;

    int *x = simulate(n, DT, T);

    if (save && x != NULL) {
        FILE *f = fopen("data/results_escape.bin", "wb");
        if (f) {
            fwrite(x, sizeof(int), n, f); 
            fclose(f);
        }
    }

    return 0;
}