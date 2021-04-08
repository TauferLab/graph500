
#ifndef COMMON_CUSTOM_H
#define COMMON_CUSTOM_H


inline int heap_remove(int *heap, int *n, float *weights) {
    int nn = *n;
    int best_i = 0;
    int best = heap[0];
    float min = weights[best];

    for (int i = 1; i < nn; ++i) {
        int entry = heap[i];
        if (weights[entry] < min) {
            min = weights[entry];
            best = entry;
            best_i = i;
        }
    }
    *n -= 1;
    heap[best_i] = heap[*n];
    return best;
}

inline void heap_insert(int *heap, int *n, int value) {
    heap[*n] = value;
    *n += 1;
}

inline int queue_remove(int *queue, int *queue_start, int *queue_end) {
    *queue_start += 1;
    return queue[*queue_start - 1];
}

inline void queue_insert(int *queue, int *queue_start, int *queue_end, int value) {
    queue[*queue_end] = value;
    *queue_end += 1;
}



#endif // COMMON_CUSTOM_H
