
#ifndef COMMON_CUSTOM_H
#define COMMON_CUSTOM_H

#ifdef __cplusplus
extern "C" {
#endif


#include <stdbool.h>

void* heap_alloc();
void heap_free(void* heap);

bool heap_empty(void* heap);
int heap_remove(void* heap);
void heap_insert(void* heap, int value, float weight);

inline int queue_remove(int *queue, int *queue_start, int *queue_end) {
    *queue_start += 1;
    return queue[*queue_start - 1];
}

inline void queue_insert(int *queue, int *queue_start, int *queue_end, int value) {
    queue[*queue_end] = value;
    *queue_end += 1;
}


#ifdef __cplusplus
} // extern C
#endif

#endif // COMMON_CUSTOM_H
