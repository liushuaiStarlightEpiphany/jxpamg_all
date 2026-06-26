#include <stdio.h>

#define TOSTRING(x) #x

#ifdef USE_COLOR_PRINTF
#define _RESET "\033[0m"
#define _RED "\033[31m"
#define _GREEN "\033[32m"
#define _YELLOW "\033[33m"
#define _BLUE "\033[34m"
#define _MAGENTA "\033[35m"
#define _CYAN "\033[36m"
#define _WHITE "\033[37m"
#define _BOLD "\033[1m"
#define _UNDERLINE "\033[4m"
#define _BLINK "\033[5m"
#define _REVERSE "\033[7m"
#define _HIDE "\033[8m"
#define _CLEAR "\033[2J"
#else
#define _RESET ""
#define _RED ""
#define _GREEN ""
#define _YELLOW ""
#define _BLUE ""
#define _MAGENTA ""
#define _CYAN ""
#define _WHITE ""
#define _BOLD ""
#define _UNDERLINE ""
#define _BLINK ""
#define _REVERSE ""
#define _HIDE ""
#define _CLEAR ""
#endif

/*----------------------------------------------------------------------------*/
/* printf functions                                                           */
/*----------------------------------------------------------------------------*/
#define WARN_PRINTF(...)                                                                                                                                       \
    printf(_YELLOW _BOLD "warning: " _RESET _YELLOW);                                                                                                          \
    printf(__VA_ARGS__);                                                                                                                                       \
    printf(_RESET)
#define ERROR_PRINTF(...)                                                                                                                                      \
    printf(_RED _BOLD "error: " _RESET _RED);                                                                                                                  \
    printf(__VA_ARGS__);                                                                                                                                       \
    printf(_RESET)
#define SUCCESS_PRINTF(...)                                                                                                                                    \
    printf(_GREEN _BOLD "success: " _RESET _GREEN);                                                                                                            \
    printf(__VA_ARGS__);                                                                                                                                       \
    printf(_RESET)
#define INFO_PRINTF(...)                                                                                                                                       \
    printf(_BLUE _BOLD "info: " _RESET _BLUE);                                                                                                                 \
    printf(__VA_ARGS__);                                                                                                                                       \
    printf(_RESET)
#define DEBUG_PRINTF(...)                                                                                                                                      \
    printf(_MAGENTA _BOLD "debug: " _RESET _MAGENTA __FILE__ ":%s:%d: " _RESET, __func__, __LINE__);                                                           \
    printf(__VA_ARGS__)
#define JX_PRINTF printf
#define JX_FPRINTF fprintf
#define jx_printf printf
#define jx_fprintf fprintf

#define jx_fwrite fwrite
// #define jx_fread fread
#define jx_fread(ptr, size, count, stream)                                                                                                                 \
    do {                                                                                                                                                       \
        size_t __result = fread(ptr, size, count, stream);                                                                                                     \
        if (__result != count) {                                                                                                                               \
            fprintf(stderr, "fread failed at %s:%d\n", __FILE__, __LINE__);                                                                                    \
            exit(1);                                                                                                                                           \
        }                                                                                                                                                      \
    } while (0)

// conditional printf for mpi
#define PRINTF_MPI0(...)                                                                                                                                       \
    if (iam == 0) {                                                                                                                                            \
        printf(__VA_ARGS__);                                                                                                                                   \
    }
#define PRINTF_MPI1(...)                                                                                                                                       \
    if (iam == 1) {                                                                                                                                            \
        printf(__VA_ARGS__);                                                                                                                                   \
    }

// MPI init macro for iam, np
#define MPI_CLAIM(comm)                                                                                                                                        \
    int iam, np;                                                                                                                                               \
    jx_MPI_Comm_rank(comm, &iam);                                                                                                                          \
    jx_MPI_Comm_size(comm, &np);

// you'd better use { } to wrap the code block following the macro to use multiple lines
#define IAM_0 if (iam == 0)
#define IAM_1 if (iam == 1)