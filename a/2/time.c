#include <stdio.h>
#include <limits.h>

float a[3396][3][224][224];

int main()
{/* expected time to traverse array of this size */
    a[1000][2][199][199] = 12345678;
    int max = INT_MIN;
    for(int i = 0; i < 3396; ++i)
        for(int j = 0; j < 3; ++j)
            for(int k = 0; k < 224; ++k)
                for(int l = 0; l < 224; ++l)
                    if(a[i][j][k][l] > max)
                        max = a[i][j][k][l];
    printf("%d\n", max);
}
